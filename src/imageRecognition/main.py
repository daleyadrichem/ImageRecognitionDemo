# main.py
from __future__ import annotations

import cv2
import numpy as np
import torch

from utils import Demo, FPSCounter, get_device, draw_fps, draw_title
from classification_demo import ImageClassificationDemo
from detection_demo import ObjectDetectionDemo
from segmentation_demo import SegmentationDemo
from style_transfer_demo import StyleTransferDemo


WINDOW_NAME = (
    "AI Vision Demos "
    "(1: Classification, 2: Detection, 3: Segmentation, 4: Style, q: Quit)"
)


def build_demos(device: torch.device) -> list[Demo]:
    """
    Instantiate all demos that can be switched between at runtime.

    Parameters
    ----------
    device : torch.device
        Device on which all models should be executed.

    Returns
    -------
    list of Demo
        List of instantiated demo objects implementing the `Demo` protocol.

    Raises
    ------
    RuntimeError
        If any of the demos fail to initialize their models.
    """
    return [
        ImageClassificationDemo(device),
        ObjectDetectionDemo(device),
        SegmentationDemo(device),
        StyleTransferDemo(device, style_name="candy"),
    ]


def run() -> None:
    """
    Main loop for the vision demos.

    Grabs frames from the default webcam, runs the currently active demo,
    and displays the result in an OpenCV window. Demos can be switched
    using the keyboard hotkeys.

    Controls
    --------
    1 : Image classification
    2 : Object detection
    3 : Semantic segmentation
    4 : Style transfer (artistic filter)
    q : Quit

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the default camera cannot be opened.
    """
    device = get_device()
    print(f"Using device: {device}")

    demos = build_demos(device)
    current_index = 0
    current_demo = demos[current_index]

    print("Controls:")
    print("  1 - Image classification")
    print("  2 - Object detection")
    print("  3 - Semantic segmentation")
    print("  4 - Neural style transfer")
    print("  q - Quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open default camera (index 0).")

    fps_counter = FPSCounter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Optionally resize for performance
            frame = cv2.resize(frame, (640, 480))

            # Process with current demo
            output_frame = current_demo.process_frame(frame)

            # Update and draw FPS
            fps = fps_counter.update()
            draw_fps(output_frame, fps)

            # Draw the current demo name
            draw_title(output_frame, current_demo.name)

            cv2.imshow(WINDOW_NAME, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                current_index = 0
            elif key == ord("2"):
                current_index = 1
            elif key == ord("3"):
                current_index = 2
            elif key == ord("4"):
                current_index = 3

            current_demo = demos[current_index]

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
