import os
import pytest
from rvm.api import coco_eval

@pytest.mark.integration
def test_coco_eval(tmp_path):
    """
    Integration test: run COCO evaluation function and check outputs.
    Requires:
      - A preds.json file inside data/images
      - A COCO annotation JSON file at data/annotations.json
    """
    images_dir = "data/images"  # must contain preds.json
    ann_file = "data/annotations.json"  # COCO ground-truth annotations
    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    coco_eval(
        images_dir=images_dir,
        ann_file=ann_file,
        out_dir=str(out_dir),
    )

    # Reports directory must contain an HTML report
    html_files = [f for f in os.listdir(out_dir) if f.endswith(".html")]
    assert html_files, "No HTML report generated"

    # Report should contain metrics
    report_path = out_dir / html_files[0]
    with open(report_path, "r") as f:
        html_content = f.read()
    assert "Precision" in html_content
    assert "Recall" in html_content
