# demo/detect_webcam.py
import argparse
from rvm.api import detect

def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on webcam")
    parser.add_argument("--device", default="0", help="Webcam device ID")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    detect(source=args.device, model=args.model, out_dir=args.out, realtime=True)

if __name__ == "__main__":
    main()
