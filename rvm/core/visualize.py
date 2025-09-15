import cv2
import numpy as np
from rvm.core.types import Box

def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(image, f"{box.class_id}:{box.confidence:.2f}",
                    (box.x1, box.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image

def random_color(seed: int):
    """
    Generate a stable random color for each object ID using a fixed seed.
    This ensures the same object keeps the same color across frames.
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, 3).tolist()


def draw_masks(image, masks, alpha=0.5):
    """
    Efficient mask overlay with unique colors per object.
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    # Pre-generate colors for all masks
    num_masks = len(masks)
    colors = np.random.randint(0, 255, size=(num_masks, 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        # Convert segmentation list -> numpy polygon
        pts = np.array(mask.segmentation, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color=tuple(int(c) for c in colors[i]))

    # Blend once
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended