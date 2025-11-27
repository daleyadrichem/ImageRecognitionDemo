# classification_demo.py
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from utils import Demo, draw_classification_label


class ImageClassificationDemo(Demo):
    """
    Image classification demo using a ResNet-50 model pretrained on ImageNet.

    The entire frame is treated as a single image and classified into one of
    1000 ImageNet classes.

    Attributes
    ----------
    name : str
        Human-readable demo name.
    device : torch.device
        Device on which the model is executed.
    model : torch.nn.Module
        Loaded ResNet-50 model.
    preprocess : torch.nn.Module
        Preprocessing transform pipeline provided by the weights.
    categories : list of str
        List of ImageNet class names.
    """

    name: str = "Image Classification (ResNet-50, ImageNet)"

    def __init__(self, device: torch.device) -> None:
        """
        Initialize the image classification demo.

        Parameters
        ----------
        device : torch.device
            Device on which the model should be executed.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model weights cannot be loaded.
        """
        self.device = device
        self.model: nn.Module
        self.model, self.preprocess, self.categories = self._load_model_and_metadata()
        self.model.to(self.device)
        self.model.eval()

    def _load_model_and_metadata(self) -> tuple[nn.Module, nn.Module, List[str]]:
        """
        Load the pretrained model, preprocessing transforms, and labels.

        Returns
        -------
        tuple of (torch.nn.Module, torch.nn.Module, list of str)
            The model, the preprocessing transform, and the list of class names.

        Raises
        ------
        RuntimeError
            If loading pretrained weights fails.
        """
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        categories = weights.meta["categories"]
        return model, preprocess, categories

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run classification on the given frame and draw the top prediction.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        numpy.ndarray
            Output image in BGR format with the top-1 prediction drawn.

        Raises
        ------
        ValueError
            If `frame` is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

        # Convert BGR (OpenCV) -> RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_prob, top_idx = torch.max(probabilities, dim=0)
        label = self.categories[top_idx.item()]
        confidence = float(top_prob.item())

        output_frame = frame.copy()
        draw_classification_label(output_frame, label, confidence)
        return output_frame

