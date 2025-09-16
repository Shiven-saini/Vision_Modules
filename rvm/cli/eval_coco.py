# rvm/cli/eval_coco.py
import argparse
from rvm.eval.coco_eval import evaluate_coco

def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO detections")
    parser.add_argument("--images", required=True, help="Images directory (with preds.json inside)")
    parser.add_argument("--ann", required=True, help="Path to COCO annotations file")
    parser.add_argument("--out", default="reports", help="Output directory")
    args = parser.parse_args()

    metrics = evaluate_coco(f"{args.images}/preds.json", args.ann, args.out)
    print("Evaluation finished:", metrics)

if __name__ == "__main__":
    main()
