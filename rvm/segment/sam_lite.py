# rvm/segment/sam_lite.py
"""
SAM-Lite wrapper (using FastSAM-s.pt):
 - class SamLiteSegmenter
 - method: segment(image: np.ndarray, point_coords=None, point_labels=None) -> List[Mask]

If FastSAM is not installed, this will fall back to a lightweight
stub that returns a central rectangular mask (useful for tests / CI).
"""
from typing import List, Optional
import numpy as np
import cv2

from rvm.core.types import Mask


class SamLiteSegmenter:
    def __init__(self, model_path: str = "FastSAM-s.pt", device: str = "cpu"):
        """
        Try to load FastSAM model if available. If not, keep a flag to use fallback.
        Args:
            model_path: path to FastSAM checkpoint (default: FastSAM-s.pt)
            device: 'cpu', 'cuda', or 'mps'
        """
        self.device = device
        self._available = False
        try:
            from ultralytics import FastSAM  # type: ignore
            self.model = FastSAM(model_path)
            self._available = True
        except Exception as e:
            print(f"[WARN] FastSAM not available, fallback mode: {e}")
            self._available = False
            self.model = None

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """
        Convert binary mask (H,W) to polygon (list of [x,y] ints).
        """
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            ys, xs = np.where(mask == 1)
            return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

    def segment(self, image: np.ndarray, point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None) -> List[Mask]:
        """
        Run segmentation on an image using FastSAM if available.

        Args:
            image: BGR image as numpy array.
            point_coords / point_labels: ignored (not used in FastSAM).

        Returns:
            List[Mask]: list of Mask dataclasses
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array (H,W,3)")

        h, w = image.shape[:2]

        if self._available:
            try:
                results = self.model.predict(image, device=self.device, retina_masks=True, imgsz=512)
                masks_list: List[Mask] = []
                for r in results:
                    if not hasattr(r, "masks") or r.masks is None:
                        continue
                    for m in r.masks:
                        for poly in m.xy:  # polygon points [[x, y], ...]
                            masks_list.append(
                                Mask(
                                    segmentation=[list(map(int, p)) for p in poly],
                                    confidence=1.0,  # FastSAM doesn't return per-mask score
                                    class_id=0
                                )
                            )
                return masks_list
            except Exception as e:
                print(f"[WARN] FastSAM prediction failed, falling back: {e}")

        # Fallback: simple centered rectangle mask
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [Mask(segmentation=polygon, confidence=1.0, class_id=0)]
