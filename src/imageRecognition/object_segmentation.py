# segmentation_demo.py
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from utils import Demo


class SegmentationDemo(Demo):
    """
    Semantic segmentation demo using DeepLabV3 with a ResNet-50 backbone.

    Produces a colored mask of class labels and overlays it on top
    of the original frame.

    Attributes
    ----------
    name : str
        Human-readable demo name.
    device : torch.device
        Device on which the model is executed.
    model : torch.nn.Module
        Loaded segmentation model.
    preprocess : torch.nn.Module
        Preprocessing transform pipeline provided by the weights.
    categories : list of str
        List of semantic class names.
    color_map : numpy.ndarray
        Color lookup table for each class index.
    alpha : float
        Blend factor between the original frame and the segmentation mask.
    """

    name: str = "Semantic Segmentation (DeepLabV3, COCO/VOC)"

    def __init__(self, device: torch.device, alpha: float = 0.5) -> None:
        """
        Initialize the semantic segmentation demo.

        Parameters
        ----------
        device : torch.device
            Device on which the model should be executed.
        alpha : float, optional
            Blend factor between original frame and segmentation mask
            in the range [0, 1]. A value of 0 uses only the original frame,
            a value of 1 uses only the mask. By default 0.5.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model weights cannot be loaded.
        ValueError
            If `alpha` is not in the range [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Expected alpha in the range [0, 1].")

        self.device = device
        self.alpha = alpha

        self.model, self.preprocess, self.categories = self._load_model_and_metadata()
        self.model.to(self.device)
        self.model.eval()

        self.color_map = self._create_color_map(len(self.categories))

    def _load_model_and_metadata(self):
        """
        Load the pretrained segmentation model, preprocessing transforms
        and labels.

        Returns
        -------
        tuple
            (model, preprocess_transform, categories) where:

            * model : torch.nn.Module
                Loaded DeepLabV3 model.
            * preprocess_transform : torch.nn.Module
                Transform pipeline expected by the model.
            * categories : list of str
                Semantic class names.

        Raises
        ------
        RuntimeError
            If loading pretrained weights fails.
        """
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = deeplabv3_resnet50(weights=weights)
        preprocess = weights.transforms()
        categories: List[str] = weights.meta["categories"]
        return model, preprocess, categories

    def _create_color_map(self, num_classes: int) -> np.ndarray:
        """
        Create a deterministic color map for classes.

        Parameters
        ----------
        num_classes : int
            Number of semantic classes.

        Returns
        -------
        numpy.ndarray
            Array of shape (num_classes, 3) with dtype uint8 containing
            BGR colors for each class index.

        Raises
        ------
        ValueError
            If `num_classes` is not positive.
        """
        if num_classes <= 0:
            raise ValueError("Expected num_classes to be a positive integer.")

        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = 0  # background usually 0 -> black
        return colors

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation on the given frame and overlay masks.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        numpy.ndarray
            Output image in BGR format with segmentation masks overlaid.

        Raises
        ------
        ValueError
            If `frame` is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

        # Convert BGR -> RGB -> PIL for model-specific transforms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]

        # output shape: [num_classes, H, W]
        class_map = output.argmax(0).cpu().numpy().astype(np.uint8)

        # Resize class map back to original frame size if needed
        if class_map.shape != frame.shape[:2]:
            class_map = cv2.resize(
                class_map,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Map class indices to colors
        color_mask = self.color_map[class_map]  # (H, W, 3), uint8

        # Blend with original frame
        blended = cv2.addWeighted(frame, 1 - self.alpha, color_mask, self.alpha, 0.0)
        return blended
