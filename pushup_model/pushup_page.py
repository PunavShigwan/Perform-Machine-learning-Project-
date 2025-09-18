import streamlit as st
import time
import random
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pickle

# ----------------------
# Motivational tips
# ----------------------
tips = [
    "Keep your core tight for stability!",
    "Great pace—stay consistent!",
    "Lower a bit slower to maximize strength.",
    "Breathe out when you push up.",
    "You’re doing amazing—stay focused!",
    "Push through the burn, you got this!"
]

# ----------------------
# Page config + CSS
# ----------------------
st.set_page_config(page_title="Pushup Trainer", layout="wide")

st.markdown("""
    <style>
        /* Kill Streamlit gray padding */
        .block-container {
            padding: 0rem 2rem 0rem 2rem;
            max-width: 100% !important;
        }
        body {
            background-color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        header, footer, #MainMenu {visibility: hidden;}
        
        /* Header */
        .custom-header {
            background-color: #FFD60A;
            padding: 1.5rem;
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            color: #000;
            border-radius: 0 0 16px 16px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.15);
        }

        /* Footer */
        .custom-footer {
            background-color: #FFD60A;
            padding: 1rem;
            text-align: center;
            font-size: 1rem;
            color: #000;
            margin-top: 2rem;
            border-radius: 16px 16px 0 0;
            box-shadow: 0px -2px 8px rgba(0,0,0,0.1);
        }

        /* Metric Cards */
        .metric-box {
            background-color: #fff8d6;
            padding: 1.2rem;
            border-radius: 14px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            color: #000;
            margin-bottom: 1rem;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
            transition: all 0.2s ease-in-out;
        }
        .metric-box:hover {
            transform: translateY(-3px);
            box-shadow: 0px 6px 12px rgba(0,0,0,0.12);
        }

        /* Section headers */
        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #111;
            border-left: 6px solid #FFD60A;
            padding-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Header
# ----------------------
st.markdown("<div class='custom-header'>🏋️ Pushup Prediction Trainer</div>", unsafe_allow_html=True)


# ===== Load Gradient Boosting Model =====
MODEL_PATH = r"C:\major_project\pushup_model\saved_models\GradientBoosting.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===== Mediapipe Pose setup =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ===== Helper Functions =====
def extract_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks.landmark, results
    return None, None

def landmarks_to_features(landmarks):
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(coords).reshape(1, -1)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def form_score(landmarks):
    shoulder = landmarks[11]
    hip = landmarks[23]
    ankle = landmarks[27]
    angle = calculate_angle(shoulder, hip, ankle)
    score = max(0, 100 - abs(180 - angle) * 2)
    return int(score)

def elbow_angle(landmarks):
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]
    return calculate_angle(shoulder, elbow, wrist)

def update_prediction(last_form, last_duration, current_pred):
    if last_duration is None:
        return current_pred
    form_factor = (last_form - 70) / 30
    speed_factor = 1.5 if last_duration < 1 else (0.8 if last_duration > 3 else 1.0)
    adjustment = int(form_factor * 2 * speed_factor)
    new_pred = max(5, current_pred + adjustment)
    return new_pred


# ----------------------
# Layout
# ----------------------
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("<div class='section-title'>🎥 Pushup Recording</div>", unsafe_allow_html=True)

    mode = st.radio("Choose input mode:", ["Upload Video", "Use Camera"], horizontal=True)

    source = None
    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload your push-up video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            source = tfile.name

    elif mode == "Use Camera":
        source = 0  # Webcam index

    # Trackers
    pushup_count = 0
    predicted_max = 15
    state = "UP"
    last_rep_time = None

    # Placeholders for video + metrics
    stframe = st.empty()
    rep_box = col2.empty()
    max_box = col2.empty()
    tip_box = col2.empty()

    if source is not None:
        cap = cv2.VideoCapture(source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, results = extract_landmarks(frame)
            if landmarks:
                features = landmarks_to_features(landmarks)
                score = form_score(landmarks)
                elbow = elbow_angle(landmarks)
                elbow_label = 0 if elbow < 90 else 1

                label = model.predict(features)[0]
                proba = model.predict_proba(features)[0].max()

                if proba < 0.6:
                    label = elbow_label

                if state == "UP" and label == 0:
                    state = "DOWN"
                elif state == "DOWN" and label == 1:
                    pushup_count += 1
                    state = "UP"

                    now = time.time()
                    rep_duration = None
                    if last_rep_time:
                        rep_duration = now - last_rep_time
                    last_rep_time = now

                    predicted_max = update_prediction(score, rep_duration, predicted_max)

                    # Show new motivational tip for 1 sec
                    tip_box.markdown(f"<div class='metric-box'>💡 {random.choice(tips)}</div>", unsafe_allow_html=True)
                    time.sleep(1)
                    tip_box.empty()

                # Draw overlay on video
                cv2.putText(frame, f'Pushups: {pushup_count} (Form={score}%)',
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update video
            stframe.image(frame, channels="BGR")

            # Update metrics
            rep_box.markdown(f"<div class='metric-box'>✅ Reps Completed: {pushup_count}</div>", unsafe_allow_html=True)
            max_box.markdown(f"<div class='metric-box'>📈 Predicted Max Pushups: {predicted_max}</div>", unsafe_allow_html=True)

        cap.release()


# ----------------------
# Footer
# ----------------------
st.markdown("<div class='custom-footer'>© 2025 Fitness Project | Train Smart, Stay Strong</div>", unsafe_allow_html=True)
