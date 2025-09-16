# rvm/cli/markers.py
import argparse
from rvm.api import detect_markers

def main():
    parser = argparse.ArgumentParser(description="Run marker/QR detection")
    parser.add_argument("--source", required=True, help="Path to image")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    detect_markers(args.source, args.out)

if __name__ == "__main__":
    main()
