from __future__ import annotations

from typing import List
import argparse
from pathlib import Path
import yaml

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from image_recognition.utils import Demo, draw_bounding_box


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------

class ObjectDetectionDemoYOLO(Demo):
    """
    Object detection demo using YOLOv8.

    Detected objects are drawn as bounding boxes with class labels
    and confidence scores.
    """

    name: str = "Object Detection (YOLOv8)"

    def __init__(
        self,
        device: torch.device,
        model: YOLO,
        categories: List[str],
        score_threshold: float = 0.5,
    ) -> None:
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError("score_threshold must be in [0, 1]")

        self.device = device
        self.model = model
        self.categories = categories
        self.score_threshold = score_threshold

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLOv8 object detection on a single frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR image with 3 channels")

        # YOLOv8 expects BGR np.ndarray directly
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.score_threshold,
            verbose=False,
        )[0]

        output = frame.copy()

        if results.boxes is None:
            return output

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls_id in zip(
            boxes, scores, labels
        ):
            class_name = (
                self.categories[cls_id]
                if cls_id < len(self.categories)
                else f"class_{cls_id}"
            )

            if score < 0.5:
                continue

            draw_bounding_box(
                output,
                (int(x1), int(y1), int(x2), int(y2)),
                class_name,
                float(score),
            )

        return output


def build_yolo_model(
    checkpoint_path: Path,
    data_yaml: Path,
    device: torch.device,
) -> tuple[YOLO, List[str]]:

    model = YOLO(checkpoint_path)
    model.to(device)

    # Always take categories from the model itself
    categories = list(model.names.values())

    return model, categories



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 object detection on a video."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
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
        default=0.25,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------
    # Model loading
    # -------------------------------------------------

    print(f"Loading YOLO model from: {args.model}")
    model, categories = build_yolo_model(
        Path(args.model),
        Path(args.data),
        device,
    )

    demo = ObjectDetectionDemoYOLO(
        device=device,
        model=model,
        categories=categories,
        score_threshold=args.score_threshold,
    )

    # -------------------------------------------------
    # Video setup (same as your Faster R-CNN demo)
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
            cv2.imshow("Object Detection Demo (YOLOv8)", output)
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
