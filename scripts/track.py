# scripts/track.py
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

WEBCAM_INDEX = 1
YOLO_MODEL = "yolov8n.pt"
CONF = 0.35

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        return

    model = YOLO(YOLO_MODEL)
    tracker = DeepSort(max_age=10, n_init=3)
    print("[INFO] Tracking running (webcam). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        results = model(frame, conf=CONF, classes=[0])

        ds_inputs = []
        for r in results:
            for d in r.boxes:
                x1 = int(d.xyxy[0][0]); y1 = int(d.xyxy[0][1])
                x2 = int(d.xyxy[0][2]); y2 = int(d.xyxy[0][3])
                conf_score = float(d.conf[0])
                ds_inputs.append([[x1, y1, x2, y2], conf_score, "person"])

        tracks = tracker.update_tracks(ds_inputs, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

        cv2.imshow("NightWatch - Track (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
