import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("🖐️ El Tespiti Uygulaması")
st.write("Bir görsel yükleyin, biz de içindeki eli tespit edip anahtar noktalarını gösterelim.")

uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli oku
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mediapipe el modeli
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

    # İşleme
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)

        st.success("El başarıyla tespit edildi!")
    else:
        st.warning("Görselde el bulunamadı.")

    # Görseli göster
    st.image(img, caption="İşlenmiş Görsel", use_column_width=True)
