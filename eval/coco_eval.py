# rvm/eval/coco_eval.py
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def evaluate_coco(pred_file: str, ann_file: str, out_dir: str) -> Dict[str, float]:
    """
    Evaluate detection results on a COCO-style dataset.

    Args:
        pred_file (str): Path to prediction JSON file.
        ann_file (str): Path to COCO ground-truth annotations.
        out_dir (str): Directory to save report.

    Returns:
        dict: {"precision": float, "recall": float}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(ann_file)

    with open(pred_file, "r") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list of dicts in COCO format.")

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    try:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"[ERROR] COCO evaluation failed: {e}")
        return {"precision": 0.0, "recall": 0.0}

    precision = float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0
    recall = float(coco_eval.stats[8]) if coco_eval.stats is not None else 0.0

    # Save HTML report
    report_path = out_dir / "report.html"
    with open(report_path, "w") as f:
        f.write("<html><head><title>COCO Eval Report</title></head><body>")
        f.write("<h1>COCO Evaluation Report</h1>")
        f.write("<ul>")
        f.write(f"<li>Precision (AP@[0.5:0.95]): {precision:.3f}</li>")
        f.write(f"<li>Recall (AR@100): {recall:.3f}</li>")
        f.write("</ul>")
        f.write("</body></html>")

    # Save PR curve
    try:
        plt.figure()
        plt.plot(coco_eval.eval["recall"], coco_eval.eval["precision"], label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / "pr_curve.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not plot PR curve: {e}")

    return {"precision": precision, "recall": recall}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="COCO Evaluation")
    parser.add_argument("--images", type=str, required=True, help="Directory containing preds.json")
    parser.add_argument("--ann", type=str, required=True, help="Path to annotation JSON")
    parser.add_argument("--out", type=str, default="reports", help="Output directory")

    args = parser.parse_args()
    pred_file = Path(args.images) / "preds.json"
    results = evaluate_coco(str(pred_file), args.ann, args.out)

    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
