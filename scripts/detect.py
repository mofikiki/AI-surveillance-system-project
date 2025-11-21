# scripts/detect.py
import cv2
from ultralytics import YOLO

WEBCAM_INDEX = 1
YOLO_MODEL = "yolov8n.pt"
CONF = 0.35

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        return

    model = YOLO(YOLO_MODEL)
    print("[INFO] Detection running (webcam). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        results = model(frame, conf=CONF, classes=[0])

        for r in results:
            for box in r.boxes:
                x1 = int(box.xyxy[0][0]); y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2]); y2 = int(box.xyxy[0][3])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("NightWatch - Detection (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
