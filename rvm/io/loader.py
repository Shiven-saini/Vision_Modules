import cv2 as cv

def load_image(path):
    return cv.imread(path)


def load_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height


def load_webcam(index=0):
    cap = cv.VideoCapture(index)
    return cap
