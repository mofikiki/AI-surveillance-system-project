"""
NightWatch - Stage 6.1 (No Blinking Mode)
YOLO runs every N frames, but overlays update every frame.

Your choices applied:
- Instant intrusion trigger
- Ignore small detections
- Intrusion banner = 5 seconds
- Cooldown = 10 seconds
- FRAME_SKIP = 3 (but overlays do not blink)
"""

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

ZONE_CONFIG_PATH = "config/zones.json"
ALARM_PATH = "assets/alarm.wav"

DEEPSORT_MAX_AGE = 30
DEEPSORT_N_INIT = 3

FRAME_SKIP = 3  # YOLO runs every 3 frames

MIN_AREA = 1500
MIN_W = 20
MIN_H = 20

ALARM_COOLDOWN = 10
BANNER_SECONDS = 5
# ----------------------------------------


def choose_video():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )


def load_zones():
    if not os.path.exists(ZONE_CONFIG_PATH):
        print("Zone file missing.")
        return []

    with open(ZONE_CONFIG_PATH) as f:
        data = json.load(f)

    zones = []
    for z in data["zones"]:
        pts = [(int(x), int(y)) for x, y in z["points"]]
        zones.append({
            "name": z["name"],
            "points": pts,
            "polygon": Polygon(pts)
        })
    return zones


def init_alarm():
    if not os.path.exists(ALARM_PATH):
        print("[WARN] alarm.wav not found.")
        return None

    pygame.mixer.init()
    return pygame.mixer.Sound(ALARM_PATH)


def np_arr(coords):
    return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))


def bbox_area(x1, y1, x2, y2):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h, w, h


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
    tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening video.")
        return

    last_alarm_time = {}
    last_intrusion_time = 0

    frame_index = 0
    tracks_cache = []  # last valid tracks

    print("NightWatch No-Blink Mode running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Draw zones on EVERY frame (smooth)
        for zone in zones:
            pts = zone["points"]
            cv2.polylines(frame, [np_arr(pts)], True, (0, 0, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np_arr(pts)], (0, 0, 255))
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        do_yolo = (frame_index % FRAME_SKIP == 0)

        if do_yolo:
            results = model(frame, conf=YOLO_CONF, classes=[0])

            ds_inputs = []
            for r in results:
                for det in r.boxes:
                    xyxy = det.xyxy[0].tolist()
                    conf = float(det.conf[0])
                    x1, y1, x2, y2 = map(int, xyxy)

                    area, w, h = bbox_area(x1, y1, x2, y2)
                    if area < MIN_AREA or w < MIN_W or h < MIN_H:
                        continue

                    ds_inputs.append([[x1, y1, x2, y2], conf, "person"])

            tracks = tracker.update_tracks(ds_inputs, frame=frame)
            tracks_cache = tracks  # update cache with new data
        else:
            # Reuse last YOLO/DeepSORT outputs
            tracks = tracks_cache

        now = time.time()

        # Process tracks and draw overlays every frame
        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

            centroid = Point(cx, cy)

            # Check zones
            for zone in zones:
                poly = zone["polygon"]
                inside = poly.contains(centroid) or poly.touches(centroid)

                if inside:
                    cv2.putText(frame, "INSIDE", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,255,0), 2)

                    last_t = last_alarm_time.get(tid, 0)
                    if now - last_t >= ALARM_COOLDOWN:
                        print(f"[ALARM] ID={tid} entered {zone['name']}")
                        if alarm:
                            alarm.play()
                        last_alarm_time[tid] = now
                        last_intrusion_time = now

                    break
                else:
                    cv2.putText(frame, "OUTSIDE", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255,255,0), 1)

        # Show banner
        if now - last_intrusion_time <= BANNER_SECONDS:
            cv2.putText(frame, "INTRUSION DETECTED",
                        (40, 60), cv2.FONT_HERSHEY_DUPLEX,
                        1.3, (0,0,255), 3)

        # HUD
        confirmed_count = len([t for t in tracks if t.is_confirmed()])
        cv2.putText(frame, f"Tracks: {confirmed_count}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (200,200,200), 2)

        cv2.imshow("NightWatch - NoBlink", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
