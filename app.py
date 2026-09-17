import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

# Konfigurasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

st.title("AR Dynamic Studio Web")

# Fungsi untuk memproses setiap frame video dari kamera browser
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    # Deteksi Tangan
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    
    if res.multi_hand_landmarks:
        for lms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Jalankan stream kamera di browser
webrtc_streamer(key="ar-filter", video_frame_callback=video_frame_callback)