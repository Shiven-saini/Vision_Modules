# rvm/io/loader.py
"""
Loader utilities for images, videos, and webcam.
"""

import cv2
from pathlib import Path


def load_image(path: str):
    """
    Load an image from file.
    Args:
        path (str): Path to image.
    Returns:
        ndarray: Loaded image (BGR).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def load_video(video_path: str, out_path: str, fps: float = None):
    """
    Load a video and prepare writer for output.
    Args:
        video_path (str): Input video file path.
        out_path (str): Output video file path.
        fps (float): Optional FPS override, otherwise inferred.
    Returns:
        tuple: (VideoCapture, VideoWriter)
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    # Infer video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter for annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    return cap, writer


def load_webcam(index: int = 0):
    """
    Open webcam stream.
    Args:
        index (int): Webcam index (default=0).
    Returns:
        VideoCapture
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap
