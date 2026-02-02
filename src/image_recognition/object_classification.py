from __future__ import annotations

from typing import List
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)
from tqdm import tqdm

from image_recognition.utils import Demo, draw_classification_label


class ImageClassificationDemo(Demo):
    """
    Image classification demo using a custom-trained ResNet classifier.

    Each video frame is treated as a single image and classified into
    one of the custom dataset classes.
    """

    name: str = "Image Classification Demo"

    def __init__(self, model_path: Path, device: torch.device) -> None:
        """
        Initialize the image classification demo.

        Args:
            model_path: Path to the trained model checkpoint (.pth).
            device: Torch device to run inference on.
        """
        self.device = device
        self.model_path = model_path

        self.model, self.preprocess, self.categories = (
            self._load_model_and_metadata()
        )

        self.model.to(self.device)
        self.model.eval()

    def _load_model_and_metadata(
        self,
    ) -> tuple[nn.Module, nn.Module, List[str]]:
        """
        Load model checkpoint, rebuild architecture, and load metadata.

        Returns:
            model: The reconstructed classification model.
            preprocess: Input preprocessing transform.
            classes: List of class names.
        """
        checkpoint = torch.load(self.model_path, map_location="cpu")

        classes: List[str] = checkpoint["classes"]
        num_classes = len(classes)

        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["model"])

        preprocess = weights.transforms()

        return model, preprocess, classes

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run classification on a single video frame.

        Args:
            frame: Input BGR frame from OpenCV.

        Returns:
            Annotated frame with predicted class and confidence.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR image with 3 channels")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        input_tensor = (
            self.preprocess(pil_image)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        score, idx = torch.max(probs, dim=0)
        label = self.categories[idx.item()]
        confidence = float(score.item())

        output = frame.copy()
        draw_classification_label(output, label, confidence)
        return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    """
    Entry point for running image classification on a video.

    Supports:
    - Live display mode
    - Offline processing with progress bar
    - Optional saving of annotated output video
    """
    parser = argparse.ArgumentParser(
        description="Run image classification demo on a video."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show live classification window",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output video",
    )
    parser.add_argument(
        "--output",
        default="classified_output.mp4",
        help="Path to output video (used with --save-video)",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    demo = ImageClassificationDemo(
        model_path=Path(args.model),
        device=device,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    # Video metadata
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

    if args.live:
        print("Running in LIVE mode (press 'q' to quit)")
    else:
        print("Running in OFFLINE mode")

    progress = None if args.live else tqdm(
        total=total_frames, unit="frame"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = demo.process_frame(frame)

        if args.live:
            cv2.imshow("Image Classification Demo", output)
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
