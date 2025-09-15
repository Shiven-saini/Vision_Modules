import sys
import os
from pathlib import Path

import cv2

from rvm.markers.aruco import ArucoDetector
from rvm.core.visualize import draw_markers
from rvm.io.loader import load_image
from rvm.io.writer import save_json, save_image

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m demos.markers_image input.jpg output_dir/")
        return

    image_path = sys.argv[1]
    out_dir = sys.argv[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load input image
    img = load_image(image_path)

    # Run marker detection
    detector = ArucoDetector()
    corners, ids = detector.detect(img)

    # Draw results
    out_img = draw_markers(img, corners, ids)

    # Save outputs
    save_image(out_img, out_dir, "markers_result.jpg")

    # Save JSON
    if ids is not None:
        data = [
            {"id": int(marker_id), "corners": corner.tolist()}
            for marker_id, corner in zip(ids.flatten(), corners)
        ]
        save_json(data, os.path.join(out_dir, "markers_result.json"))

    print(f"[INFO] Detected {0 if ids is None else len(ids)} markers")
    print(f"[INFO] Results saved to: {out_dir}")

if __name__ == "__main__":
    main()