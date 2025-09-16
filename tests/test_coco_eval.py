import os
import subprocess
import pytest


@pytest.mark.integration
def test_coco_eval_cli(tmp_path):
    """
    Integration test: run COCO evaluation CLI and check outputs.
    Requires:
      - A preds.json file inside data/images
      - A COCO annotation JSON file at data/annotations.json
    """
    images_dir = "data/images"  # must contain preds.json
    ann_file = "data/annotations.json"  # COCO ground-truth annotations
    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Run CLI via module import
    result = subprocess.run(
        [
            "python",
            "-m", "rvm.eval.coco_eval",
            "--images", images_dir,
            "--ann", ann_file,
            "--out", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    # Debug logs
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # CLI must finish successfully
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Reports directory must contain an HTML report
    html_files = [f for f in os.listdir(out_dir) if f.endswith(".html")]
    assert html_files, "No HTML report generated"

    # Report should contain metrics
    report_path = out_dir / html_files[0]
    with open(report_path, "r") as f:
        html_content = f.read()
    assert "Precision" in html_content
    assert "Recall" in html_content
