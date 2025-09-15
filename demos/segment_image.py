# demos/segment_image.py
import sys
import os
from rvm.segment.sam_lite import SamSegmenter

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m demos.segment_image input.jpg output_dir/")
        return

    image_path = sys.argv[1]
    out_dir = sys.argv[2]

    os.makedirs(out_dir, exist_ok=True)

    seg = SamSegmenter()
    results = seg.run(image_path)
    seg.save(image_path, results, out_dir)

if __name__ == "__main__":
    main()