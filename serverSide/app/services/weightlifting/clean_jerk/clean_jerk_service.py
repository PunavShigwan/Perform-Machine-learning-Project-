import cv2
import numpy as np
import mediapipe as mp
import joblib
import os

# ================================
# LOAD MODEL (FIXED)
# ================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ML_Model",
    "clean_and_jerk_model",
    "models",
    "best_model.pkl"
)

LABEL_MAP_PATH = os.path.join(
    BASE_DIR,
    "ML_Model",
    "clean_and_jerk_model",
    "models",
    "label_map.pkl"
)

print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

print("LABEL PATH:", LABEL_MAP_PATH)
print("LABEL EXISTS:", os.path.exists(LABEL_MAP_PATH))

model = joblib.load(MODEL_PATH)
LABEL_MAP = joblib.load(LABEL_MAP_PATH)

model = joblib.load(MODEL_PATH)
LABEL_MAP = joblib.load(LABEL_MAP_PATH)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def pressure_level(val):
    if val < 15:
        return "LOW"
    elif val < 35:
        return "MODERATE"
    else:
        return "HIGH"


def evaluate_form(phase, knee_angle, hip_angle, shoulder_angle, elbow_angle):
    issues = []

    if phase in ["P1_first_pull", "P3_second_pull"]:
        if knee_angle < 130 or knee_angle > 170:
            issues.append("KNEE")
        if hip_angle < 140:
            issues.append("HIP")

    if phase in ["P4_catch", "P6_dip"]:
        if shoulder_angle < 150:
            issues.append("SHOULDER")
        if elbow_angle < 160:
            issues.append("ELBOW")

    return issues


# =====================================================
# MAIN SERVICE FUNCTION
# =====================================================
def analyze_clean_jerk_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev_knee_angle = None
    prev_hip_y = None

    total_frames = 0
    bad_frames = 0
    phase_counter = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            if prev_knee_angle is None:
                knee_vel = 0
                hip_vel = 0
            else:
                knee_vel = knee_angle - prev_knee_angle
                hip_vel = hip[1] - prev_hip_y

            prev_knee_angle = knee_angle
            prev_hip_y = hip[1]

            features = np.array([[knee_angle, hip_vel, knee_vel]])
            phase_id = model.predict(features)[0]
            phase = LABEL_MAP[phase_id]

            phase_counter[phase] = phase_counter.get(phase, 0) + 1

            issues = evaluate_form(phase, knee_angle, hip_angle, shoulder_angle, elbow_angle)
            if issues:
                bad_frames += 1

            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()

    form_accuracy = round((1 - bad_frames / max(total_frames, 1)) * 100, 2)

    return {
        "total_frames": total_frames,
        "bad_frames": bad_frames,
        "form_accuracy_percent": form_accuracy,
        "phase_distribution": phase_counter,
        "processed_video_path": os.path.abspath(output_path)   # ✅ FIX
    }