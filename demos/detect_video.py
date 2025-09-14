import cv2
import sys
import os
from rvm.detect.yolo import YoloDetector
from rvm.core.visualize import draw_boxes
from rvm.io.loader import load_video
from rvm.io.writer import save_video, save_json

def main():
    if len(sys.argv) < 3:
        print("Usage: python demos/detect_video.py input.mp4 output_dir/")
        return

    video_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    detector = YoloDetector("yolov8n.pt")
    cap, fps, width, height = load_video(video_path)

    out_video = save_video(os.path.join(out_dir, "annotated.mp4"), fps, width, height)
    all_results = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.run(frame)
        annotated = draw_boxes(frame, results)
        out_video.write(annotated)

        frame_data = [b.to_dict() for b in results]
        all_results.append({"frame_id": frame_id, "detections": frame_data})

        frame_id += 1

    cap.release()
    out_video.release()

    save_json(all_results, os.path.join(out_dir, "detections.json"))
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
