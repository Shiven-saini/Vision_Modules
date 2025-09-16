# rvm/cli/segment.py
import argparse
from rvm.api import segment_image

def main():
    parser = argparse.ArgumentParser(description="Run image segmentation")
    parser.add_argument("--source", required=True, help="Path to image")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    segment_image(args.source, args.out)

if __name__ == "__main__":
    main()
