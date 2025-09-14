import cv2
import time
from rvm.detect.yolo import YoloDetector
from rvm.core.visualize import draw_boxes
from rvm.io.loader import load_webcam

def main():
    detector = YoloDetector("yolov8n.pt")
    cap = load_webcam(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = detector.run(frame)
        fps = 1.0 / (time.time() - start)

        # Draw boxes
        annotated = draw_boxes(frame, results)

        cv2.imshow("Webcam Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
