import cv2
from ultralytics import YOLO

print("---- NightWatch Smoke Test ----")

# Check OpenCV
print("OpenCV version:", cv2.__version__)

# Try loading a YOLO model
try:
    model = YOLO("yolov8n.pt")  # will auto-download if missing
    print("YOLO loaded successfully.")
except Exception as e:
    print("YOLO failed to load:", e)
    exit()

# Try opening webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam NOT detected. Try using a video file instead.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam.")
    exit()

print("Webcam frame captured successfully:", frame.shape)

# Show the frame briefly
cv2.imshow("Smoke Test", frame)
cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()

print("Smoke test complete.")

