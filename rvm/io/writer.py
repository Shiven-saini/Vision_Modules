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
