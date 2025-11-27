# utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import time

import cv2
import numpy as np
import torch


@runtime_checkable
class Demo(Protocol):
    """
    Protocol that all demo classes should implement.

    A demo takes a BGR frame from OpenCV, processes it and returns a
    BGR frame with visualizations drawn on top.

    Attributes
    ----------
    name : str
        Human-readable demo name shown on the output frame.
    """

    name: str

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image in BGR format (as provided by OpenCV).

        Returns
        -------
        numpy.ndarray
            Output image in BGR format with visualizations drawn on top.

        Raises
        ------
        ValueError
            If `frame` does not have the expected shape or dtype.
        """
        ...


@dataclass
class FPSCounter:
    """
    Utility for tracking frames-per-second in a video loop.

    Attributes
    ----------
    last_time : float
        Time of the last update call in seconds since the epoch.
    fps : float
        Current estimate of frames-per-second.
    smoothing : float
        Smoothing factor for the exponential moving average (0–1).
    """

    last_time: float = time.time()
    fps: float = 0.0
    smoothing: float = 0.9  # exponential moving average

    def update(self) -> float:
        """
        Update internal FPS estimate and return the current value.

        Returns
        -------
        float
            Updated frames-per-second estimate.

        Raises
        ------
        None
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            return self.fps

        current_fps = 1.0 / dt
        if self.fps == 0.0:
            self.fps = current_fps
        else:
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps
        return self.fps


def get_device() -> torch.device:
    """
    Select an appropriate torch.device for running models.

    Prefers CUDA if available, otherwise Metal (MPS) on macOS, otherwise CPU.

    Returns
    -------
    torch.device
        Device on which models should be executed.

    Raises
    ------
    None
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS Metal
        return torch.device("mps")
    return torch.device("cpu")


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Draw FPS information in the top-left corner of the frame (in-place).

    Parameters
    ----------
    frame : numpy.ndarray
        Image buffer in BGR format to be modified in-place.
    fps : float
        Frames-per-second value to render.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `frame` is not a 3-channel image.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_title(frame: np.ndarray, title: str) -> None:
    """
    Draw the current demo title at the bottom-left of the frame (in-place).

    Parameters
    ----------
    frame : numpy.ndarray
        Image buffer in BGR format to be modified in-place.
    title : str
        Title text to draw.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `frame` is not a 3-channel image.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")

    (text_width, text_height), baseline = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    x, y = 10, frame.shape[0] - 10

    # Draw background rectangle for readability
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - baseline - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        frame,
        title,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_classification_label(
    frame: np.ndarray,
    label: str,
    confidence: float,
    position: tuple[int, int] = (10, 50),
) -> None:
    """
    Draw classification label and confidence on the frame (in-place).

    Parameters
    ----------
    frame : numpy.ndarray
        Image buffer in BGR format to be modified in-place.
    label : str
        Class label name.
    confidence : float
        Confidence score in the range [0, 1].
    position : tuple of int, optional
        (x, y) coordinates of the text baseline, by default (10, 50).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `frame` is not a 3-channel image or `confidence` is outside [0, 1].
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("Expected confidence in the range [0, 1].")

    text = f"{label} ({confidence * 100:.1f}%)"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_bounding_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    score: float,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """
    Draw a bounding box and label with confidence on the frame (in-place).

    Parameters
    ----------
    frame : numpy.ndarray
        Image buffer in BGR format to be modified in-place.
    box : tuple of int
        Bounding box coordinates (x1, y1, x2, y2) in pixel units.
    label : str
        Class label name.
    score : float
        Confidence score in the range [0, 1].
    color : tuple of int, optional
        BGR color of the bounding box and label background,
        by default (0, 255, 0).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `frame` is not a 3-channel image or `score` is outside [0, 1].
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3) in BGR format.")
    if not (0.0 <= score <= 1.0):
        raise ValueError("Expected score in the range [0, 1].")

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {score * 100:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 4),
        (x1 + text_width + 4, y1),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
