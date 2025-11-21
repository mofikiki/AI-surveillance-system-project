# scripts/zone_ui.py
import cv2
import json
import os

ZONE_CONFIG = "config/zones.json"
WEBCAM_INDEX = 1

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Cannot grab frame from webcam.")
        return

    cv2.namedWindow("Draw Zone - NightWatch")
    cv2.setMouseCallback("Draw Zone - NightWatch", mouse_callback)

    print("INSTRUCTIONS:")
    print(" - Click to add polygon points.")
    print(" - Press 'r' to reset.")
    print(" - Press ENTER to save.")
    print(" - Press 'q' to quit without saving.")

    while True:
        tmp = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(tmp, p, 4, (0,255,0), -1)
            if i > 0:
                cv2.line(tmp, points[i-1], p, (0,255,0), 2)
        if len(points) > 2:
            cv2.line(tmp, points[-1], points[0], (0,255,0), 2)

        cv2.imshow("Draw Zone - NightWatch", tmp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points.clear()
            print("[INFO] Points reset.")
        if key == ord('q'):
            print("Exit without saving.")
            break
        if key == 13:  # ENTER
            if len(points) < 3:
                print("Need at least 3 points.")
                continue
            os.makedirs("config", exist_ok=True)
            zone_data = {"zones":[{"name":"main_zone","points":points}]}
            with open(ZONE_CONFIG, "w") as f:
                json.dump(zone_data, f, indent=4)
            print(f"Zone saved to {ZONE_CONFIG}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
