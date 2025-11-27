# style_transfer_demo.py
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from utils import Demo


class StyleTransferDemo(Demo):
    """
    Real-time style transfer demo using a fast neural style model.

    This demo applies the aesthetic style learned from a reference style
    image (e.g. a Van Gogh or Picasso painting) to each incoming frame.
    The model has been trained offline on that style image; at runtime
    the style is applied in a single forward pass, enabling near real-time
    stylization.

    Attributes
    ----------
    name : str
        Human-readable demo name.
    device : torch.device
        Device on which the model is executed.
    model : torch.nn.Module
        Loaded fast style-transfer model.
    style_name : str
        Name of the style used by the model (e.g. ``"candy"``).
    """

    name: str = "Neural Style Transfer (Fast, pre-trained)"

    def __init__(
        self,
        device: torch.device,
        style_name: str = "candy",
        hub_repo: str = "pytorch/vision:v0.10.0",
    ) -> None:
        """
        Initialize the style transfer demo.

        Parameters
        ----------
        device : torch.device
            Device on which the model should be executed.
        style_name : str, optional
            Name of the pre-trained style to use, by default "candy".
            Common options (depending on the installed torchvision version)
            include "candy", "mosaic", "picasso", "udnie".
        hub_repo : str, optional
            Torch Hub repository spec for loading the model,
            by default "pytorch/vision:v0.10.0".

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the style-transfer model or its weights cannot be loaded.
        """
        self.device = device
        self.style_name = style_name

        # Lazy import torch.hub here to avoid importing it if not needed.
        # The first call will download the model weights if they are not
        # yet cached on the machine.
        self.model = torch.hub.load(
            hub_repo,
            "fast_neural_style",
            model=self.style_name,
        ).to(self.device)
        self.model.eval()

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert a BGR OpenCV frame to a 4D tensor suitable for the model.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (uint8, shape (H, W, 3)).

        Returns
        -------
        torch.Tensor
            Tensor of shape (1, 3, H, W) with dtype float32 and values
            in the range [0, 255].

        Raises
        ------
        ValueError
            If `frame` is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor back to a BGR OpenCV frame.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape (1, 3, H, W) with values typically in [0, 255].

        Returns
        -------
        numpy.ndarray
            Styled image in BGR format (uint8, shape (H, W, 3)).

        Raises
        ------
        ValueError
            If `tensor` does not have the expected shape.
        """
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(
                "Expected tensor of shape (1, 3, H, W) as model output."
            )

        # Move to CPU, remove batch dimension, clamp to valid range
        img = tensor.detach().cpu().squeeze(0)
        img = img.clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)

        # Convert back RGB -> BGR for OpenCV
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return bgr

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply neural style transfer to a single frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        numpy.ndarray
            Output image in BGR format with the artistic style applied.

        Raises
        ------
        ValueError
            If `frame` is not a 3-channel image.
        RuntimeError
            If the model fails during the forward pass.
        """
        content_tensor = self._frame_to_tensor(frame)

        with torch.no_grad():
            styled_tensor = self.model(content_tensor)

        output_frame = self._tensor_to_frame(styled_tensor)
        return output_frame
