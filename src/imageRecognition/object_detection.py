# detection_demo.py
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from utils import Demo, draw_bounding_box


class ObjectDetectionDemo(Demo):
    """
    Object detection demo using Faster R-CNN with a ResNet-50 backbone.

    Detected objects are drawn as bounding boxes with class labels and
    confidence scores.

    Attributes
    ----------
    name : str
        Human-readable demo name.
    device : torch.device
        Device on which the model is executed.
    model : torch.nn.Module
        Loaded detection model.
    categories : list of str
        List of COCO class names.
    score_threshold : float
        Minimum confidence score required to display a detection.
    """

    name: str = "Object Detection (Faster R-CNN, COCO)"

    def __init__(self, device: torch.device, score_threshold: float = 0.5) -> None:
        """
        Initialize the object detection demo.

        Parameters
        ----------
        device : torch.device
            Device on which the model should be executed.
        score_threshold : float, optional
            Minimum confidence score required to draw a detection,
            by default 0.5.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model weights cannot be loaded.
        ValueError
            If `score_threshold` is not in the range [0, 1].
        """
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError("Expected score_threshold in the range [0, 1].")

        self.device = device
        self.score_threshold = score_threshold

        self.model, self.categories = self._load_model_and_metadata()
        self.model.to(self.device)
        self.model.eval()

    def _load_model_and_metadata(self) -> tuple[torch.nn.Module, List[str]]:
        """
        Load the pretrained detection model and category labels.

        Returns
        -------
        tuple of (torch.nn.Module, list of str)
            The detection model and list of COCO class names.

        Raises
        ------
        RuntimeError
            If loading pretrained weights fails.
        """
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        categories = weights.meta["categories"]
        return model, categories

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run object detection on the given frame and draw bounding boxes.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        numpy.ndarray
            Output image in BGR format with bounding boxes and labels drawn.

        Raises
        ------
        ValueError
            If `frame` is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

        # Convert BGR -> RGB and then to tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        output_frame = frame.copy()

        for box, label_idx, score in zip(boxes, labels, scores):
            score_val = float(score.item())
            if score_val < self.score_threshold:
                continue

            x1, y1, x2, y2 = box.int().tolist()
            class_name = self.categories[int(label_idx.item())]
            draw_bounding_box(
                output_frame,
                (x1, y1, x2, y2),
                class_name,
                score_val,
            )

        return output_frame
