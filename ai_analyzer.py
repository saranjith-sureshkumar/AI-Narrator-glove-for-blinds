import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time
import pygame  # Your audio mixer

# Init pygame mixer (your code)
pygame.mixer.init()

st.set_page_config(layout="wide")
st.title("🧤🔊 AI Object Analyzer - Live Webcam + Voice")

@st.cache_resource
def load_model():
    try:
        return YOLO("./best.pt")  # Your custom
    except:
        return YOLO("yolov8n.pt")

model = load_model()
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Sidebar: Webcam test + select
st.sidebar.header("📹 Webcam Setup")
test_indices = st.sidebar.button("🔍 Test All Cameras")

if test_indices:
    st.sidebar.write("**Scanning cameras...**")
    for i in range(4):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        status = "✅ LIVE" if ret else "❌ None"
        st.sidebar.write(f"**Index {i}:** {status}")
        cap.release()

cam_index = st.sidebar.slider("Select Webcam", 0, 3, 1)  # USB usually 1
st.sidebar.success("Laptop speakers auto-used")

# MAIN LIVE DETECTION (Your robust loop)
if st.button("🚀 START LIVE STREAM + DETECT", use_container_width=True):
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    object_count = st.empty()
    
    prev_objects = set()
    last_speech = 0
    
    st.balloons()
    status_placeholder.info("👀 Point objects → Live detection + voice!")
    
    while st.button("⏹️ STOP STREAM"):
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("No video. Replug USB / change index.")
            break
        
        # YOLO Detection (Your model)
        results = model(frame, conf=0.45, verbose=False)
        current_objects = []
        h, w = frame.shape[:2]
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cls_id = int(detection.cls[0])
            conf = float(detection.conf[0])
            obj_name = model.names[cls_id]
            
            # Position
            center_x = (x1 + x2) // 2
            position = "left" if center_x < w//3 else "right" if center_x > 2*w//3 else "center"
            current_objects.append(f"{obj_name} {position}")
            
            # Custom label
            label_text = f"{obj_name} {conf:.1f} {position}"
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Voice new objects (pyttsx3 + pygame ready)
        new_detected = set(current_objects) - prev_objects
        now = time.time()
        if new_detected and now - last_speech > 1.8:
            speech_text = f"Object ahead: {' and '.join(new_detected)}."
            status_placeholder.success(speech_text)
            engine.say(speech_text)
            engine.runAndWait()
            # pygame.mixer.music.play()  # Uncomment for MP3 alert
            last_speech = now
        
        prev_objects = set(current_objects)
        
        # Streamlit live display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption=f"Live Detection: {len(results[0].boxes)} objects", 
                               channels="RGB", use_column_width=True)
        object_count.metric("Objects", len(results[0].boxes))
    
    cap.release()
    st.success("✅ Stream stopped.")

st.markdown("""
### 🎯 Status
- **Model:** best.pt (your notebook/desk) or yolov8n fallback
- **Speech:** Laptop speakers (pyttsx3)
- **Cam:** Index test + select

**Demo Flow:**
1. TEST CAMERAS → Find USB (usually 1)
2. Select index → START LIVE  
3. Point chair/notebook → **Green boxes + "Object ahead: notebook center" SPEAKS!**
""")
