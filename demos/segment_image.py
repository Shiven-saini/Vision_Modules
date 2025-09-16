# demo/segment_image.py
import argparse
from rvm.api import segment_image

def main():
    parser = argparse.ArgumentParser(description="Run segmentation on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    masks = segment_image(args.image, out_dir=args.out)
    print("Segmentation results:", masks)

if __name__ == "__main__":
    main()
