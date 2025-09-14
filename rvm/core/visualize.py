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
