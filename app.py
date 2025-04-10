import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="El Tespiti", page_icon="🖐️")
st.title("🖐️ El Tespiti Uygulaması")
st.write("Görsel ya da video yükleyin, içindeki elleri tespit edelim!")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def process_image(image):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(image, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return image, True
    return image, False


# Görsel işle
st.header("🖼️ Görsel Yükleyerek El Tespiti")
uploaded_file = st.file_uploader("Bir görsel seçin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processed_img, found = process_image(img)

    if found:
        st.success("El başarıyla tespit edildi!")
    else:
        st.warning("Görselde el bulunamadı.")

    st.image(processed_img, caption="İşlenmiş Görsel", use_column_width=True)

# Video işle
st.header("🎥 Video Yükleyerek El Tespiti")
uploaded_video = st.file_uploader("Bir video seçin", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    st.info("Videoyu işliyoruz, lütfen bekleyin...")

    # İşlenmiş videoyu kaydetmek için
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        # Videoyu RGB'den BGR'ye çevirip yazıyoruz
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    st.success("Video işleme tamamlandı!")

    # İşlenmiş videoyu oynat
    st.video(out_path)

