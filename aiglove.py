import cv2
import time
from ultralytics import YOLO
import pyttsx3

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
CAM_INDEX = 1
CONF_THRES = 0.45
SPEAK_DELAY = 6  # seconds between narrations
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

# Camera
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Camera not working")

print("AI Narration started")

last_spoken = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = model(frame, conf=CONF_THRES, verbose=False)
    boxes = results[0].boxes

    if boxes:
        box = boxes[0]
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        if cx < w * 0.33:
            position = "towards the left side"
        elif cx > w * 0.66:
            position = "towards the right side"
        else:
            position = "directly in front"

        # Simple color estimation
        obj_region = frame[y1:y2, x1:x2]
        avg_color = obj_region.mean(axis=(0,1))
        color_desc = "light colored" if avg_color.mean() > 140 else "dark colored"

        narration = f"""
I have detected an object in your surroundings.
The object appears to be a {label}.
It is located {position} of your current position.
The object looks {color_desc} in appearance.
The environment seems to be an indoor or controlled area.
"Please be aware of the object and proceed carefully.
"""  

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if time.time() - last_spoken > SPEAK_DELAY:
            print("Speaking narration...")
            engine.say(narration)
            engine.runAndWait()   # 🔴 LONG BLOCKING SPEECH (IMPORTANT)
            last_spoken = time.time()

    cv2.imshow("AI Environment Narration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()

