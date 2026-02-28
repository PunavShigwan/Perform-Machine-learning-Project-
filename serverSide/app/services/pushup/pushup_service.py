import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque

print("📦 Loading pushup_service...")

# =====================================================
# MODEL PATH
# =====================================================
# =====================================================
# MODEL PATH (DYNAMIC & SAFE)
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

MODEL_PATH = r"C:\Users\rauls\Desktop\Perform-Machine-learning-Project-\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"

# =====================================================
# CONFIG
# =====================================================
ELBOW_DOWN = 90
ELBOW_UP = 155
SMOOTHING_WINDOW = 5
FRAME_CONFIRM = 2

# =====================================================
# LAZY MODEL LOAD
# =====================================================
_model = None

def get_model():
    global _model
    if _model is None:
        print("📦 Loading ML model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print("✅ Pushup ML model loaded")
    return _model

# =====================================================
# MEDIAPIPE
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(np.degrees(radians))
    return 360 - ang if ang > 180 else ang


def elbow_angle(lm):
    return angle(lm[11], lm[13], lm[15])


def body_angle(lm):
    return angle(lm[11], lm[23], lm[27])


def is_pushup_position(lm):
    body_ang = body_angle(lm)
    shoulder, hip, wrist = lm[11], lm[23], lm[15]
    return (
        160 <= body_ang <= 200 and
        wrist.y > shoulder.y and
        abs(hip.y - shoulder.y) < 0.15
    )


def form_score(lm):
    body_ang = body_angle(lm)
    return int(np.clip(100 - abs(180 - body_ang) * 1.8, 0, 100))


def rep_rating(score):
    if score >= 85:
        return "Excellent", 2
    elif score >= 70:
        return "Good", 1
    elif score >= 50:
        return "Poor", 0
    else:
        return "Bad", -2


def fatigue_index(log):
    if len(log) < 2:
        return 0

    form_drop = max(0, log[0]["form"] - log[-1]["form"])
    time_increase = max(0, log[-1]["time"] - log[0]["time"]) * 10
    bad_ratio = sum(1 for r in log if r["form"] < 50) / len(log) * 100

    fatigue = (
        0.5 * form_drop +
        0.3 * time_increase +
        0.2 * bad_ratio
    )
    return int(np.clip(fatigue, 0, 100))


def fatigue_level(v):
    if v < 30:
        return "LOW"
    elif v < 60:
        return "MODERATE"
    else:
        return "HIGH"

# =====================================================
# MAIN SERVICE FUNCTION
# =====================================================
def analyze_pushup_video(input_path, output_path):

    get_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25

    print("🎬 Video opened successfully")
    print("🎞 Resolution:", w, "x", h, "FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open")

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    state = "UP"
    pushups = 0
    predicted_max = 5

    elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
    down_frames = 0
    up_frames = 0
    down_scores = []
    valid_down = False

    # ✅ NEW: per-rep detailed log
    rep_log = []
    last_rep_time = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            elbow_buf.append(elbow_angle(lm))
            smooth_elbow = np.mean(elbow_buf)
            score = form_score(lm)

            if smooth_elbow < ELBOW_DOWN:
                down_frames += 1
                up_frames = 0
            elif smooth_elbow > ELBOW_UP:
                up_frames += 1
                down_frames = 0

            if state == "UP" and down_frames >= FRAME_CONFIRM:
                state = "DOWN"
                down_scores = []
                valid_down = is_pushup_position(lm)

            if state == "DOWN":
                down_scores.append(score)

            if state == "DOWN" and up_frames >= FRAME_CONFIRM:
                if valid_down:
                    pushups += 1

                    now = time.time()
                    rep_time = now - last_rep_time if last_rep_time else 1.0
                    last_rep_time = now

                    rep_form = int((np.mean(down_scores) + score) / 2)
                    rating, delta = rep_rating(rep_form)
                    predicted_max = max(5, predicted_max + delta)

                    # ✅ STORE FORM % PER REP
                    rep_log.append({
                        "rep": pushups,
                        "form": rep_form,
                        "rating": rating,
                        "time": round(rep_time, 2)
                    })

                state = "UP"

            fatigue = fatigue_index(rep_log)

            cv2.putText(frame, f"Pushups: {pushups}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Fatigue: {fatigue}% ({fatigue_level(fatigue)})",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        out.write(frame)

    cap.release()
    out.release()
    pose.close()

    final_fatigue = fatigue_index(rep_log)

    return {
        "pushup_count": pushups,
        "fatigue": final_fatigue,
        "fatigue_level": fatigue_level(final_fatigue),
        "estimated_range": f"{predicted_max-2} - {predicted_max+2}",
        "reps": rep_log,                 # ✅ NEW FIELD
        "output_video_path": os.path.abspath(output_path)

    }
