import os
import json
import cv2
from ultralytics import YOLO
from rvm.core.types import Box

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def run(self, source):
        results = self.model(source)
        output = []
        for r in results:
            for box in r.boxes:
                b = Box(
                    x1=int(box.xyxy[0][0]),
                    y1=int(box.xyxy[0][1]),
                    x2=int(box.xyxy[0][2]),
                    y2=int(box.xyxy[0][3]),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0])
                )
                output.append(b)
        return output

    def save(self, source, results, out_dir):
        """
        Save detection results as JSON and annotated image/video.
        
        Args:
            source (str): Path to the input image or video.
            results (List[Box]): Detection results from run().
            out_dir (str): Directory to save outputs.
        """
        os.makedirs(out_dir, exist_ok=True)

        # 1. Save JSON
        json_path = os.path.join(out_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump([b.__dict__ for b in results], f, indent=4)
        print(f"Saved JSON results to {json_path}")

        # 2. Annotate and save image/video
        if source.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(source)
            for b in results:
                cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                cv2.putText(img, f"{b.class_id}:{b.confidence:.2f}",
                            (b.x1, b.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

            out_img_path = os.path.join(out_dir, "annotated.jpg")
            cv2.imwrite(out_img_path, img)
            print(f"Saved annotated image to {out_img_path}")

        elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            cap = cv2.VideoCapture(source)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video_path = os.path.join(out_dir, "annotated.mp4")
            out = cv2.VideoWriter(out_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                for b in results:
                    cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{b.class_id}:{b.confidence:.2f}",
                                (b.x1, b.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()
            print(f"Saved annotated video to {out_video_path}")
