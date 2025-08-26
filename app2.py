import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict
from PIL import Image

# Load model
model = load_model("emotion_class.h5", compile=True)

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotions = ['happy', 'sad', 'neutral']
img_size = 48

st.set_page_config(page_title="Facial Emotion Detection", layout="wide")
st.title("😊 Facial Emotion Detection")

# Session state
if "emotion_counter" not in st.session_state:
    st.session_state.emotion_counter = defaultdict(int)
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Input Source")
    option = st.radio("Choose input:", ["📷 Webcam", "⬆️ Upload Image"])

    if option == "📷 Webcam":
        img_file = st.camera_input("Take a picture")
    else:
        img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📊 Emotion Analysis Summary")

    if img_file is not None:
        # Read image
        img = Image.open(img_file)
        frame = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (img_size, img_size))
            roi = roi.reshape(1, img_size, img_size, 1) / 255.0

            prediction = model.predict(roi, verbose=0)
            emotion = emotions[np.argmax(prediction)]
            st.session_state.emotion_counter[emotion] += 1
            st.session_state.total_frames += 1

            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        st.image(frame, channels="RGB", caption="Processed Image")

        # Show percentages
        if st.session_state.total_frames > 0:
            for emotion in emotions:
                count = st.session_state.emotion_counter[emotion]
                percent = (count / st.session_state.total_frames) * 100 if st.session_state.total_frames > 0 else 0
                st.write(f"**{emotion.capitalize()}**: {percent:.2f}%")

            st.bar_chart({e: st.session_state.emotion_counter[e] for e in emotions})
    else:
        st.info("👈 Please take a picture or upload an image to see results.")
