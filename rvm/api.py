# rvm/api.py
"""
Unified high-level API for Robora Vision Modules (RVM).
Provides:
- Object detection (image, video, webcam)
- Segmentation
- Marker detection
- COCO evaluation
"""

from pathlib import Path
from typing import List, Dict, Any

from rvm.detect.yolo import YOLODetector
from rvm.segment.sam_lite import SamLiteSegmenter
from rvm.markers.aruco import ArucoDetector
from rvm.core.visualize import draw_boxes, draw_masks, draw_markers
from rvm.io.loader import load_image, load_video, load_webcam
from rvm.io.writer import save_image, save_json
from eval.coco_eval import evaluate_coco


# -----------------------------
# Detection
# -----------------------------
def detect(
    source: str,
    model: str = "yolov8n.pt",
    out_dir: str = "results",
    realtime: bool = False
) -> List[Dict[str, Any]]:
    """
    Run object detection on an image, video, or webcam.

    Args:
        source (str): Path to image/video or webcam index (e.g., "0").
        model (str): YOLO model weights.
        out_dir (str): Directory to save results.
        realtime (bool): If True and source is webcam, display results in real-time.

    Returns:
        list of dict: Detection results (boxes, scores, labels).
    """
    detector = YOLODetector(model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Webcam
    if source.isdigit():
        cap = load_webcam(int(source))
        all_results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            annotated = draw_boxes(frame, detections)

            if realtime:
                import cv2
                cv2.imshow("RVM Detection (Press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            for d in detections:
                result = d.to_dict()
                result["frame"] = frame_idx
                all_results.append(result)
            frame_idx += 1

        cap.release()
        save_json(all_results, out_dir / "detect_webcam.json")
        return all_results

    # Image
    elif source.lower().endswith((".jpg", ".jpeg", ".png")):
        img = load_image(source)
        detections = detector.detect(img)
        annotated = draw_boxes(img, detections)
        save_image(annotated, out_dir, "detect_result.jpg")
        save_json([d.to_dict() for d in detections], out_dir / "detect_result.json")
        return [d.to_dict() for d in detections]

    # Video
    elif source.lower().endswith((".mp4", ".mov", ".avi")):
        cap, writer = load_video(source, out_dir / "detect_result.mp4")
        all_results = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            annotated = draw_boxes(frame, detections)
            writer.write(annotated)

            for d in detections:
                result = d.to_dict()
                result["frame"] = frame_idx
                all_results.append(result)
            frame_idx += 1

        cap.release()
        writer.release()
        save_json(all_results, out_dir / "detect_result.json")
        return all_results

    else:
        raise ValueError(f"Unsupported source type: {source}")


# -----------------------------
# Segmentation
# -----------------------------
def segment_image(image_path: str, out_dir: str = "results") -> List[Dict[str, Any]]:
    img = load_image(image_path)
    segmenter = SamLiteSegmenter()
    masks = segmenter.segment(img)

    annotated = draw_masks(img, masks)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(annotated, out_dir, "segment_result.jpg")
    save_json([m.to_dict() for m in masks], out_dir / "segment_result.json")
    return [m.to_dict() for m in masks]


# -----------------------------
# Marker detection
# -----------------------------
def detect_markers(image_path: str, out_dir: str = "results") -> List[Dict[str, Any]]:
    img = load_image(image_path)
    detector = ArucoDetector()
    markers = detector.detect(img)

    annotated = draw_markers(img, markers)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(annotated, out_dir, "markers_result.jpg")
    save_json([m.to_dict() for m in markers], out_dir / "markers_result.json")
    return [m.to_dict() for m in markers]


# -----------------------------
# COCO Evaluation
# -----------------------------
def coco_eval(pred_file: str, ann_file: str, out_dir: str = "reports") -> Dict[str, float]:
    """
    Run COCO-style evaluation on predictions.

    Args:
        pred_file (str): Path to predictions JSON file.
        ann_file (str): Path to COCO annotation JSON file.
        out_dir (str): Directory to save reports.

    Returns:
        dict: {"precision": float, "recall": float}
    """
    return evaluate_coco(pred_file, ann_file, out_dir)
