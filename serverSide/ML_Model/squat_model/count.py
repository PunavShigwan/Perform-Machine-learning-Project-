import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque

print("📦 Loading squat_service...")

# =====================================================
# MODEL PATH
# =====================================================
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\squat_model\saved_models_v2\GradientBoosting.pkl"

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("❌ Squat model not found")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print("✅ Squat model loaded")
    return _model

# =====================================================
# MEDIAPIPE
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# =====================================================
# ANGLE FUNCTION
# =====================================================
def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    ang = abs(np.degrees(radians))
    return 360 - ang if ang > 180 else ang

# =====================================================
# SQUAT METRICS
# =====================================================
def knee_angle(lm):
    return angle(lm[23], lm[25], lm[27])  # hip-knee-ankle

def hip_angle(lm):
    return angle(lm[11], lm[23], lm[25])  # shoulder-hip-knee

def form_score(lm):
    k = knee_angle(lm)
    h = hip_angle(lm)

    score = 100

    # depth
    if k > 100:
        score -= 30

    # hip position
    if h < 70:
        score -= 20

    return int(np.clip(score, 0, 100))

# =====================================================
# MAIN FUNCTION
# =====================================================
def analyze_squat_video(input_path, output_path):

    model = get_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video")

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5)) or 25

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"avc1"),
                          fps, (w, h))

    pose = mp_pose.Pose()

    state = "UP"
    squats = 0
    predicted_max = 5

    knee_buf = deque(maxlen=5)
    down_frames = 0
    up_frames = 0

    rep_log = []
    last_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            k = knee_angle(lm)
            knee_buf.append(k)
            smooth_k = np.mean(knee_buf)

            score = form_score(lm)

            # =====================
            # SQUAT LOGIC
            # =====================
            if smooth_k < 90:
                down_frames += 1
                up_frames = 0
            elif smooth_k > 160:
                up_frames += 1
                down_frames = 0

            if state == "UP" and down_frames > 2:
                state = "DOWN"

            if state == "DOWN" and up_frames > 2:
                squats += 1

                now = time.time()
                rep_time = now - last_time if last_time else 1
                last_time = now

                rep_log.append({
                    "rep": squats,
                    "form": score,
                    "time": round(rep_time, 2)
                })

                # fatigue logic
                if score < 60:
                    predicted_max -= 1
                else:
                    predicted_max += 1

                predicted_max = max(5, predicted_max)

                state = "UP"

            # =====================
            # DRAW
            # =====================
            cv2.putText(frame, f"Squats: {squats}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Form: {score}%",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        out.write(frame)

    cap.release()
    out.release()
    pose.close()

    return {
        "squat_count": squats,
        "estimated_max": predicted_max,
        "reps": rep_log,
        "output_video_path": os.path.abspath(output_path)
    }