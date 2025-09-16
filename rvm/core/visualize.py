# rvm/core/visualize.py
"""
Visualization utilities for detection/segmentation results.

Functions:
- draw_boxes(image, boxes)
- draw_masks(image, masks)
- draw_markers(image, markers)
"""

import cv2
import numpy as np
from typing import List, Tuple

from rvm.core.types import Box, Mask, Marker
import random

_COLOR_MAP = {}


def _get_color_for_id(obj_id):
    """
    Return a consistent color for a given object ID.
    """
    if obj_id not in _COLOR_MAP:
        # Generate a random color
        random.seed(obj_id)
        _COLOR_MAP[obj_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return _COLOR_MAP[obj_id]


def draw_boxes(image, boxes):
    """
    Draw bounding boxes on the image.
    Each box gets a unique color based on its coordinates hash.
    """
    annotated = image.copy()
    for box in boxes:
        # Hash box coordinates to get a stable color
        obj_id = hash((box.x1, box.y1, box.x2, box.y2)) % 10000
        color = _get_color_for_id(obj_id)

        cv2.rectangle(
            annotated,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            2,
        )
        label = f"{box.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (box.x1, max(0, box.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def draw_masks(image: np.ndarray, masks: List[Mask], alpha: float = 0.4) -> np.ndarray:
    """ 
    Olay masks on the image with transparency.
    Each mask gets a unique color based on its segmentation hash.
    """
    img = image.copy()
    overlay = img.copy()

    for mask in masks:
        if len(mask.segmentation) == 0:
            continue
        obj_id = hash(tuple(map(tuple, mask.segmentation))) % 10000
        color = _get_color_for_id(obj_id)

        pts = np.array(mask.segmentation, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, f"{mask.confidence:.2f}",
                    (pts[0][0][0], pts[0][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_markers(image: np.ndarray, markers: List[Marker]) -> np.ndarray:
    """
    Draw detected markers on the image.
    Each marker gets a unique color based on its ID or corners hash.
    """
    img = image.copy()
    for marker in markers:
        # Use marker ID if available, else hash corners for color
        obj_id = marker.id if marker.id is not None else hash(tuple(map(tuple, marker.corners))) % 10000
        color = _get_color_for_id(obj_id)

        corners = np.array(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [corners], True, color, 2)
        cX = int(np.mean([pt[0] for pt in marker.corners]))
        cY = int(np.mean([pt[1] for pt in marker.corners]))
        cv2.putText(img, str(marker.id), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img
