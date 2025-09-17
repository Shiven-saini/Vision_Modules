import os
import json
import pytest
from rvm.api import coco_eval

@pytest.mark.integration
def test_coco_eval(tmp_path):
    # Setup fake COCO data
    ann_file = tmp_path / "annotations.json"
    pred_file = tmp_path / "preds.json"
    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Minimal COCO annotation (1 image, 1 bbox)
    annotations = {
        "images": [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "test"}],
    }
    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    # Minimal prediction in COCO format
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}]
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    # Run eval
    results = coco_eval(
        pred_file=str(pred_file),
        ann_file=str(ann_file),
        out_dir=str(out_dir),
    )

    # Validate output
    html_files = [f for f in os.listdir(out_dir) if f.endswith(".html")]
    assert html_files, "No HTML report generated"
    assert "precision" in results and "recall" in results
