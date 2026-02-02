# src/image_recognition/sources.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import cv2
import numpy as np


class FrameSource(Protocol):
    """
    Protocol for any source that can provide video frames.

    Implementations can be a physical camera, a video file, or a static image.
    """

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read a single frame from the source.

        Returns
        -------
        tuple of (bool, numpy.ndarray)
            A pair (ok, frame) where `ok` indicates whether a frame was
            successfully read, and `frame` is the image in BGR format.

        Raises
        ------
        None
        """
        ...

    def release(self) -> None:
        """
        Release any resources held by the source.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        ...


@dataclass
class CameraSource:
    """
    Frame source that reads frames from a physical camera via OpenCV.

    Attributes
    ----------
    camera_index : int
        Index of the camera device (as used by OpenCV).
    """

    camera_index: int = 0

    def __post_init__(self) -> None:
        """
        Initialize the underlying VideoCapture object.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.camera_index}."
            )

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read a frame from the camera.

        Returns
        -------
        tuple of (bool, numpy.ndarray)
            A pair (ok, frame) where `ok` indicates whether a frame was
            successfully read, and `frame` is the image in BGR format.

        Raises
        ------
        None
        """
        return self._cap.read()

    def release(self) -> None:
        """
        Release the underlying camera resource.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._cap.release()

@dataclass
class MediaFileSource:
    """
    Frame source that reads from a video file or static image.

    If the path points to a video file, frames are read sequentially.
    If the path points to an image, the same image is returned for every read.

    Supported:
    - Images: .png, .jpg, .jpeg
    - Video:  .mp4
    
    Attributes
    ----------
    path : str
        Filesystem path to the image or video file.
    loop_image : bool
        If True and the path is an image, the image is returned indefinitely.
    """

    path: str
    loop_image: bool = True

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    VIDEO_EXTS = {".mp4"}

    def __post_init__(self) -> None:
        """
        Initialize the media source based on file extension.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the extension is unsupported.
        """
        import os

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Media file does not exist: {self.path!r}")

        ext = os.path.splitext(self.path)[1].lower()

        if ext in self.IMAGE_EXTS:
            # Load as static image
            img = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Failed to load image file: {self.path!r}"
                )
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Image must have 3 channels (BGR).")

            self._image = img
            self._cap = None
            return

        elif ext in self.VIDEO_EXTS:
            # Load as video
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise FileNotFoundError(
                    f"Failed to open video file: {self.path!r}"
                )
            self._cap = cap
            self._image = None
            return

        else:
            raise ValueError(
                f"Unsupported file extension: {ext!r}. "
                f"Supported images: {self.IMAGE_EXTS}, videos: {self.VIDEO_EXTS}"
            )

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read a frame from the media file.

        Returns
        -------
        tuple of (bool, numpy.ndarray)
            If the source is a video, returns successive frames until the video
            ends, after which `ok` will be False.

            If the source is an image, returns the image on every call when
            `loop_image` is True. If `loop_image` is False, returns the image
            once and then (False, last_frame) afterwards.

        Raises
        ------
        None
        """
        if self._cap is not None:
            return self._cap.read()

        # Static image mode
        if self._image is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        if self.loop_image:
            return True, self._image.copy()

        if getattr(self, "_served_once", False):
            return False, self._image.copy()

        self._served_once = True
        return True, self._image.copy()

    def release(self) -> None:

        """
        Release any underlying resources.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if self._cap is not None:
            self._cap.release()
