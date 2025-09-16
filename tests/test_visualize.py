# tests/test_visualize.py
import numpy as np
import cv2

from rvm.core.visualize import draw_boxes, draw_masks, draw_markers
from rvm.core.types import Box, Mask, Marker


def test_draw_boxes():
    """Test that draw_boxes modifies the image when boxes are provided."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a simple box
    boxes = [Box(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=1)]

    out = draw_boxes(img, boxes)

    # Shape should remain unchanged
    assert out.shape == img.shape
    # Some pixels should change
    assert np.any(out != img)


def test_draw_masks():
    """Test that draw_masks overlays mask polygons correctly."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Simple square polygon
    segmentation = [[10, 10], [50, 10], [50, 50], [10, 50]]
    masks = [Mask(segmentation=segmentation, confidence=0.8, class_id=1)]

    out = draw_masks(img, masks)

    assert out.shape == img.shape
    assert np.any(out != img)


def test_draw_markers():
    """Test that draw_markers annotates markers correctly."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Square marker with id=1
    corners = [(20, 20), (40, 20), (40, 40), (20, 40)]
    markers = [Marker(id=1, corners=corners)]

    out = draw_markers(img, markers)

    assert out.shape == img.shape
    assert np.any(out != img)
