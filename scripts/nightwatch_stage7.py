"""
NightWatch - Stage 7.1 (Maximum Stability Edition - Webcam Version)

This is your exact working code, but updated to:
✔ Remove file picker
✔ Use webcam index 1
✔ Use winsound for guaranteed alarm playback
✔ Preserve all your detections, tracking, stability fixes
"""

import os
import time
import csv
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import platform
import winsound
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- CONFIG ----------------

WEBCAM_INDEX = 1                    # your Iriun camera
YOLO_MODEL   = "yolov8n.pt"
YOLO_CONF    = 0.35

ZONE_CONFIG  = "config/zones.json"
ALARM_PATH   = "assets/alarm.wav"

FRAME_SKIP   = 2
MIN_AREA     = 300
MAX_AGE      = 10
N_INIT       = 3

ALARM_COOLDOWN = 10
BANNER_SECONDS = 5

# ----------------------------------------


# ----------------------------------------
# ALARM SYSTEM (winsound = guaranteed)
# ----------------------------------------
def play_alarm():
    try:
        winsound.PlaySound(ALARM_PATH,
                           winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print("[WARN] Alarm failed:", e)
# ----------------------------------------


# ----------------------------------------
# ZONE LOADING
# ----------------------------------------
def load_zones():
    if not os.path.exists(ZONE_CONFIG):
        print("[ERROR] zones.json missing.")
        return []

    with open(ZONE_CONFIG) as f:
        data = json.load(f)

    zones = []
    for z in data.get("zones", []):
        pts = [(int(x), int(y)) for x, y in z["points"]]
        zones.append({
            "name": z["name"],
            "points": pts,
            "polygon": Polygon(pts)
        })
    return zones


# ----------------------------------------
# HELPERS
# ----------------------------------------
def np_arr(coords):
    return np.array(coords, dtype=np.int32).reshape((-1,1,2))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def day_folder(base="logs"):
    today = datetime.now().date().isoformat()
    folder = os.path.join(base, today)
    ensure_dir(folder)
    ensure_dir(os.path.join(folder, "snaps"))
    return folder


def open_csv(csv_path):
    exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)

    if not exists:
        w.writerow([
            "timestamp_utc",
            "track_id",
            "zone",
            "centroid_x",
            "centroid_y",
            "frame_idx",
            "video_path",
            "snapshot_path"
        ])
    return f, w


def append_json(path, record):
    try:
        with open(path, "r", encoding="utf-8") as fj:
            data = json.load(fj)
        if not isinstance(data, list):
            data = [data]
    except:
        data = []

    data.append(record)
    with open(path, "w", encoding="utf-8") as fj:
        json.dump(data, fj, indent=2)


def crop_and_save(frame, bbox, out_path):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    padx = int((x2 - x1) * 0.05)
    pady = int((y2 - y1) * 0.05)

    xa = max(0, x1 - padx)
    ya = max(0, y1 - pady)
    xb = min(w-1, x2 + padx)
    yb = min(h-1, y2 + pady)

    crop = frame[ya:yb, xa:xb]
    cv2.imwrite(out_path, crop)
    return out_path


def valid_bbox(x1, y1, x2, y2, frame_w, frame_h):
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0:
        return False
    if x2 > frame_w * 1.5 or y2 > frame_h * 1.5:
        return False
    return True


# ----------------------------------------
# MAIN
# ----------------------------------------
def main():
    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}")
        return

    zones = load_zones()
    if not zones:
        print("[ERROR] No zones configured!")
        return

    print("[INFO] Loading YOLO…")
    model = YOLO(YOLO_MODEL)

    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT)

    frame_idx = 0
    tracks_cache = []
    last_alarm = {}
    last_intrusion_time = 0

    current_day = None
    csv_file = None
    csv_writer = None
    json_path = None
    day_path = None

    print("[INFO] NightWatch Stage 7.1 (Webcam Mode) running… Press q to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam disconnected or empty frame.")
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            # rotate logs daily
            today = datetime.now().date().isoformat()
            if today != current_day:
                if csv_file:
                    csv_file.close()

                current_day = today
                day_path = day_folder()

                csv_path = os.path.join(day_path, "intrusions.csv")
                json_path = os.path.join(day_path, "intrusions.json")

                csv_file, csv_writer = open_csv(csv_path)

            # draw zones
            for z in zones:
                pts = z["points"]
                cv2.polylines(frame, [np_arr(pts)], True, (0,0,255), 2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np_arr(pts)], (0,0,255))
                cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

            # YOLO every N frames
            run_yolo = (frame_idx % FRAME_SKIP == 0)

            if run_yolo:
                results = model(frame, conf=YOLO_CONF, classes=[0])
                ds_inputs = []

                for r in results:
                    for det in r.boxes:
                        x1 = int(det.xyxy[0][0])
                        y1 = int(det.xyxy[0][1])
                        x2 = int(det.xyxy[0][2])
                        y2 = int(det.xyxy[0][3])

                        if not valid_bbox(x1, y1, x2, y2, w, h):
                            continue

                        area = (x2 - x1) * (y2 - y1)
                        if area < MIN_AREA:
                            continue

                        ds_inputs.append([
                            [x1, y1, x2, y2],
                            float(det.conf[0]),
                            "person"
                        ])

                tracks = tracker.update_tracks(ds_inputs, frame=frame)
                tracks_cache = tracks
            else:
                tracks = tracks_cache

            now = time.time()

            # track loop
            for t in tracks:
                if not t.is_confirmed():
                    continue

                tid = t.track_id
                x1, y1, x2, y2 = map(int, t.to_ltrb())

                if not valid_bbox(x1, y1, x2, y2, w, h):
                    continue

                # visuals
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"ID:{tid}",
                            (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 2)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

                p = Point(cx, cy)

                for z in zones:
                    inside = z["polygon"].contains(p) or z["polygon"].touches(p)

                    if inside:
                        cv2.putText(frame, "INSIDE", (x1, y2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,0), 2)

                        last_t = last_alarm.get(tid, 0)

                        if now - last_t >= ALARM_COOLDOWN:
                            last_alarm[tid] = now
                            last_intrusion_time = now

                            print(f"[ALARM] ID={tid} entered {z['name']}")

                            play_alarm()

                            timestamp = datetime.utcnow().replace(
                                tzinfo=timezone.utc
                            ).isoformat().replace(":", "-")

                            snap_name = f"{timestamp}_id{tid}.jpg"
                            snap_rel  = os.path.join("snaps", snap_name)
                            snap_path = os.path.join(day_path, snap_rel)

                            ensure_dir(os.path.dirname(snap_path))
                            crop_and_save(frame, (x1,y1,x2,y2), snap_path)

                            csv_writer.writerow([
                                timestamp,
                                tid,
                                z["name"],
                                cx, cy,
                                frame_idx,
                                "webcam_1",
                                snap_rel
                            ])
                            csv_file.flush()

                            append_json(json_path, {
                                "timestamp_utc": timestamp,
                                "track_id": tid,
                                "zone": z["name"],
                                "centroid": [cx, cy],
                                "frame_idx": frame_idx,
                                "video": "webcam_1",
                                "snapshot": snap_rel
                            })

                        break

                    else:
                        cv2.putText(frame, "OUTSIDE", (x1, y2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,0), 1)

            # banner
            if time.time() - last_intrusion_time <= BANNER_SECONDS:
                cv2.putText(frame,
                            "INTRUSION DETECTED",
                            (40,60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.3, (0,0,255), 3)

            cv2.imshow("NightWatch - Stable (Webcam)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if csv_file:
            csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
