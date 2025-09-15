import os
import cv2
from dataclasses import asdict
from ultralytics import FastSAM
from rvm.core.types import Mask
from rvm.io.writer import save_json
import numpy as np

class SamSegmenter:
    def __init__(self, model_name="FastSAM-s.pt", device="cpu"):
        """
        Initialize FastSAM lightweight segmenter.
        model_name: one of ["FastSAM-s.pt", "FastSAM-x.pt"] or local path
        device: "cpu" (default), "cuda", or "mps"
        """
        self.device = device
        self.model = FastSAM(model_name)

    def run(self, image_path):
        """
        Run segmentation on a single image using FastSAM.
        Returns: list[Mask]
        """
        results = self.model(image_path, device=self.device, retina_masks=True, imgsz=512)
        masks = []

        for r in results:
            if not hasattr(r, "masks") or r.masks is None:
                continue
            for m in r.masks:
                for poly in m.xy:  # polygon points [[x, y], ...]
                    masks.append(
                        Mask(
                            segmentation=[list(map(int, p)) for p in poly],
                            confidence=1.0,  # FastSAM does not output per-mask scores
                            class_id=0
                        )
                    )
        return masks

    def save(self, image_path, results, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        image = cv2.imread(image_path)

        overlay = image.copy()
        num_masks = len(results)
        colors = np.random.randint(0, 255, size=(num_masks, 3), dtype=np.uint8)

        for i, m in enumerate(results):
            pts = np.array(m.segmentation, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=tuple(int(c) for c in colors[i]))

        blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        out_img = os.path.join(out_dir, "overlay.png")
        cv2.imwrite(out_img, blended)

        # Save JSON (bắt buộc)
        masks_dict = [asdict(m) for m in results]
        out_json = os.path.join(out_dir, "masks.json")
        save_json(masks_dict, out_json)

        print(f"[INFO] Saved overlay to {out_img}")
        print(f"[INFO] Saved masks to {out_json}")
