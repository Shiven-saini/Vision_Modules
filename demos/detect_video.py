# demo/detect_video.py
import argparse
from rvm.api import detect

def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    result = detect(args.video, model=args.model, out_dir=args.out)
    print(f"Annotated video saved at: {result}")

if __name__ == "__main__":
    main()