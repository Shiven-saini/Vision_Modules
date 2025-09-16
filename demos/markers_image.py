# demo/markers_image.py
import argparse
from rvm.api import detect_markers

def main():
    parser = argparse.ArgumentParser(description="Detect ArUco markers in an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    markers = detect_markers(args.image, out_dir=args.out)
    print("Markers detected:", markers)

if __name__ == "__main__":
    main()