import json
import os
import cv2 as cv

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_video(output_path, fps, width, height):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # codec mp4
    return cv.VideoWriter(output_path, fourcc, fps, (width, height))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(image, out_dir, filename):
    """Save an image to the given output directory."""
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    cv.imwrite(out_path, image)
    return out_path