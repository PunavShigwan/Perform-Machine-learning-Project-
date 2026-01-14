import cv2
import numpy as np
import mediapipe as mp
import joblib
import os

# =====================================================
# CONFIG
# =====================================================
VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\sample_videos\op1.mp4"
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\best_model.pkl"
LABEL_MAP_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\label_map.pkl"

# =====================================================
# LOAD MODEL
# =====================================================
print("📦 Loading model...")
model = joblib.load(MODEL_PATH)
LABEL_MAP = joblib.load(LABEL_MAP_PATH)
print("✅ Model loaded")

# =====================================================
# MEDIAPIPE INIT
# =====================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# UTILS
# =====================================================
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


def joint_color(joint, issues):
    return (0, 0, 255) if joint in issues else (0, 255, 0)


def draw_joint_tag(frame, text, point, color):
    x = int(point[0] * frame.shape[1])
    y = int(point[1] * frame.shape[0])
    cv2.circle(frame, (x, y), 6, color, -1)
    cv2.putText(frame, text, (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# =====================================================
# VIDEO STREAM
# =====================================================
print("🎥 Loading video...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video not found or cannot be opened")
    exit()

print("▶ Playing video... Press Q to quit")

prev_knee_angle = None
prev_hip_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ℹ️ End of video")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    overlay_color = (0, 255, 0)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # ===============================
        # JOINT COORDINATES
        # ===============================
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

        # ===============================
        # ANGLES
        # ===============================
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # ===============================
        # VELOCITY FEATURES
        # ===============================
        if prev_knee_angle is None:
            knee_vel = 0
            hip_vel = 0
        else:
            knee_vel = knee_angle - prev_knee_angle
            hip_vel = hip[1] - prev_hip_y

        prev_knee_angle = knee_angle
        prev_hip_y = hip[1]

        # ===============================
        # MODEL PREDICTION
        # ===============================
        features = np.array([[knee_angle, hip_vel, knee_vel]])
        phase_id = model.predict(features)[0]
        phase = LABEL_MAP[phase_id]

        # ===============================
        # PRESSURE (PROXY)
        # ===============================
        knee_pressure = abs(knee_vel) * (180 - knee_angle)
        hip_pressure = abs(hip_vel) * 1000

        knee_p = pressure_level(knee_pressure)
        hip_p = pressure_level(hip_pressure)

        # ===============================
        # FORM EVALUATION
        # ===============================
        issues = evaluate_form(
            phase,
            knee_angle,
            hip_angle,
            shoulder_angle,
            elbow_angle
        )

        form_correct = len(issues) == 0
        quality = "FORM CORRECT" if form_correct else "FORM INCORRECT"
        overlay_color = (0, 255, 0) if form_correct else (0, 0, 255)

        # ===============================
        # DRAW POSE
        # ===============================
        mp_draw.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # ===============================
        # DRAW JOINT TAGS
        # ===============================
        draw_joint_tag(frame, "KNEE", knee, joint_color("KNEE", issues))
        draw_joint_tag(frame, "HIP", hip, joint_color("HIP", issues))
        draw_joint_tag(frame, "ANKLE", ankle, joint_color("ANKLE", issues))
        draw_joint_tag(frame, "SHOULDER", shoulder, joint_color("SHOULDER", issues))
        draw_joint_tag(frame, "ELBOW", elbow, joint_color("ELBOW", issues))

        # ===============================
        # UI PANEL
        # ===============================
        cv2.rectangle(frame, (10, 10), (500, 190), (0, 0, 0), -1)

        cv2.putText(frame, f"Phase: {phase}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)

        cv2.putText(frame, f"Form: {quality}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, overlay_color, 2)

        cv2.putText(frame, f"Knee Pressure: {knee_p}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Hip Pressure: {hip_p}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not form_correct:
            cv2.putText(frame, f"Issue: {', '.join(issues)}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Clean & Jerk – AI Form Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 User stopped video")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video closed successfully")
