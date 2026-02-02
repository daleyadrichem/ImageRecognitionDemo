# detection_demo.py
from __future__ import annotations

from typing import List
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from image_recognition.utils import Demo, draw_bounding_box


class ObjectDetectionDemo(Demo):
    """
    Object detection demo using Faster R-CNN with a ResNet-50 backbone.

    Detected objects are drawn as bounding boxes with class labels and
    confidence scores.
    """

    name: str = "Object Detection (Faster R-CNN)"

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        categories: List[str],
        score_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the object detection demo.

        Parameters
        ----------
        device : torch.device
            Device on which the model should be executed.
        model : torch.nn.Module
            Detection model.
        categories : list of str
            List of class names.
        score_threshold : float, optional
            Minimum confidence score required to draw a detection.
        """
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError("score_threshold must be in [0, 1]")

        self.device = device
        self.model = model.to(device)
        self.categories = categories
        self.score_threshold = score_threshold

        self.model.eval()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run object detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.

        Returns
        -------
        np.ndarray
            Annotated output frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR image with 3 channels")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        output = frame.copy()

        for box, label, score in zip(
            outputs["boxes"],
            outputs["labels"],
            outputs["scores"],
        ):
            score_val = float(score.item())
            if score_val < self.score_threshold:
                continue

            x1, y1, x2, y2 = box.int().tolist()
            class_idx = int(label.item())

            class_name = (
                self.categories[class_idx]
                if class_idx < len(self.categories)
                else f"class_{class_idx}"
            )

            draw_bounding_box(
                output,
                (x1, y1, x2, y2),
                class_name,
                score_val,
            )

        return output


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, List[str]]:
    """
    Rebuild Faster R-CNN model from a training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file.
    device : torch.device
        Torch device.

    Returns
    -------
    model : torch.nn.Module
        Reconstructed detection model.
    categories : list of str
        Category names.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model" not in ckpt or "num_classes" not in ckpt:
        raise KeyError("Checkpoint must contain 'model' and 'num_classes'")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        ckpt["num_classes"],
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    categories = ckpt.get(
        "categories",
        [f"class_{i}" for i in range(ckpt["num_classes"])],
    )

    return model, categories


def build_coco_model(
    device: torch.device,
) -> tuple[torch.nn.Module, List[str]]:
    """
    Load pretrained COCO Faster R-CNN model.

    Returns
    -------
    model : torch.nn.Module
        COCO pretrained model.
    categories : list of str
        COCO category names.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model, weights.meta["categories"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run object detection on a video.
    """
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN object detection on a video."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--output",
        default="detected_output.mp4",
        help="Output video path (used with --save-video)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------
    # Model loading
    # -------------------------------------------------

    if args.model.lower() == "coco":
        print("Using COCO pretrained model")
        model, categories = build_coco_model(device)
    else:
        print(f"Loading model from checkpoint: {args.model}")
        model, categories = build_model_from_checkpoint(
            Path(args.model),
            device,
        )

    demo = ObjectDetectionDemo(
        device=device,
        model=model,
        categories=categories,
        score_threshold=args.score_threshold,
    )

    # -------------------------------------------------
    # Video setup
    # -------------------------------------------------

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, fps, (width, height)
        )

    progress = None if args.live else tqdm(
        total=total_frames,
        unit="frame",
    )

    if args.live:
        print("Running LIVE mode (press 'q' to quit)")
    else:
        print("Running OFFLINE mode")

    # -------------------------------------------------
    # Processing loop
    # -------------------------------------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = demo.process_frame(frame)

        if args.live:
            cv2.imshow("Object Detection Demo", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            progress.update(1)

        if writer is not None:
            writer.write(output)

    cap.release()

    if writer is not None:
        writer.release()

    if args.live:
        cv2.destroyAllWindows()

    if progress is not None:
        progress.close()

    print("Done ✅")


if __name__ == "__main__":
    main()
