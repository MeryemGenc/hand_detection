import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from io import BytesIO

st.set_page_config(page_title="El Tespiti", page_icon="🖐️")
st.title("🖐️ El Tespiti Uygulaması")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----------------------------------------
# 📷 FOTOĞRAF İŞLEME
# ----------------------------------------
st.header("📸 Görsel ile El Tespiti")
uploaded_image = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    st.image(image, caption="Tespit Edilen Eller", use_column_width=True)

# ----------------------------------------
# 🎥 VİDEO İŞLEME
# ----------------------------------------
st.header("🎥 Video ile El Tespiti")
uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_input.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25  # fallback FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Yeni video çıkışı için temp dosya
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    st.info("Video işleniyor, lütfen bekleyin...")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()

    st.success(f"Video başarıyla işlendi! Toplam kare: {frame_count}")

    # ✅ Videoyu BytesIO olarak göster
    with open(temp_output.name, 'rb') as f:
        video_bytes = f.read()
        st.video(video_bytes)
