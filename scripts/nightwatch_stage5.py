import time
import cv2
import json
import os
import pygame
from shapely.geometry import Point, Polygon
from tkinter import Tk, filedialog
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ---------------- CONFIG ----------------
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.35
ALARM_PATH = "assets/alarm.wav"
ALARM_COOLDOWN_SECONDS = 10
INTRUSION_DISPLAY_SECONDS = 2.5
ZONE_CONFIG_PATH = "config/zones.json"
# ----------------------------------------


def choose_video():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return path


def load_zones():
    if not os.path.exists(ZONE_CONFIG_PATH):
        print("No zones.json found!")
        return []

    with open(ZONE_CONFIG_PATH, "r") as f:
        data = json.load(f)

    zones = []
    for z in data["zones"]:
        pts = [(int(x), int(y)) for x, y in z["points"]]
        poly = Polygon(pts)
        zones.append({"name": z["name"], "points": pts, "polygon": poly})
    return zones


def init_alarm():
    if not os.path.exists(ALARM_PATH):
        print("[WARN] No alarm.wav found. Sound disabled.")
        return None
    pygame.mixer.init()
    return pygame.mixer.Sound(ALARM_PATH)


def np_array(coords):
    return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))


def main():

    source = choose_video()
    if not source:
        print("No video selected.")
        return

    zones = load_zones()
    if not zones:
        print("No zones loaded.")
        return

    alarm = init_alarm()

    model = YOLO(YOLO_MODEL)
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    last_alarm = {}
    last_intrusion_time = 0

    print("Running NightWatch Stage 5...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        # Draw polygon zone
        for zone in zones:
            pts = zone["points"]
            cv2.polylines(frame, [np_array(pts)], True, (0, 0, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np_array(pts)], (0, 0, 255))
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Detect people
        results = model(frame, conf=YOLO_CONF, classes=[0])

        detections = []
        for r in results:
            for det in r.boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                conf = float(det.conf[0])
                detections.append([[x1, y1, x2, y2], conf, "person"])

        # Track
        tracks = tracker.update_tracks(detections, frame=frame)

        now = time.time()

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Compute centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

            centroid_point = Point(cx, cy)

            # Check each zone
            for zone in zones:
                inside = zone["polygon"].contains(centroid_point)

                if inside:
                    cv2.putText(frame, "INSIDE ZONE", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,255,0), 2)

                    # Trigger intrusion if cooldown passed
                    last_t = last_alarm.get(tid, 0)
                    if now - last_t >= ALARM_COOLDOWN_SECONDS:
                        print(f"[INTRUSION] ID={tid} entered {zone['name']}")

                        if alarm:
                            alarm.play()

                        last_alarm[tid] = now
                        last_intrusion_time = now
                else:
                    # For debugging
                    cv2.putText(frame, "OUTSIDE", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255,255,0), 2)

        # Show intrusion banner if recent
        if now - last_intrusion_time <= INTRUSION_DISPLAY_SECONDS:
            cv2.putText(frame, "INTRUSION DETECTED", (40, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,0,255), 3)

        cv2.imshow("NightWatch - Intrusion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
