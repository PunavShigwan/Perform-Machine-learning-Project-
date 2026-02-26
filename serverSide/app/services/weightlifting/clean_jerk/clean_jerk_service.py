import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
from collections import deque

# ================================
# LOAD MODEL
# ================================
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\best_model.pkl"
LABEL_MAP_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\models\label_map.pkl"

model = joblib.load(MODEL_PATH)
LABEL_MAP = joblib.load(LABEL_MAP_PATH)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ================================
# 3-STAGE LIFT LOGIC (SIMPLIFIED)
# ================================
STAGES = ["CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"]

STAGE_LABELS = {
    "CLEAN_TO_SHOULDER": "① Clean to Shoulder",
    "JERK_OVERHEAD":     "② Jerk Over Head",
    "RELEASE_FINISH":    "③ Release / Lockout"
}

SUCCESS_STAGES = {"CLEAN_TO_SHOULDER", "JERK_OVERHEAD", "RELEASE_FINISH"}

# ================================
# JOINT LANDMARK CONFIG (NO FACE)
# ================================
JOINT_LANDMARKS = {
    "L.Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "R.Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "L.Elbow":    mp_pose.PoseLandmark.LEFT_ELBOW,
    "R.Elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW,
    "L.Wrist":    mp_pose.PoseLandmark.LEFT_WRIST,
    "R.Wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
    "L.Hip":      mp_pose.PoseLandmark.LEFT_HIP,
    "R.Hip":      mp_pose.PoseLandmark.RIGHT_HIP,
    "L.Knee":     mp_pose.PoseLandmark.LEFT_KNEE,
    "R.Knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
    "L.Ankle":    mp_pose.PoseLandmark.LEFT_ANKLE,
    "R.Ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# ================================
# HELPERS
# ================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def evaluate_stage(stage, knee_angle, hip_angle, shoulder_angle, elbow_angle):
    issues = []
    if stage == "CLEAN_TO_SHOULDER":
        if knee_angle < 120:
            issues.append(f"Knee extension low ({knee_angle:.0f}°)")
        if hip_angle < 130:
            issues.append(f"Hip extension low ({hip_angle:.0f}°)")
    elif stage == "JERK_OVERHEAD":
        if shoulder_angle < 150:
            issues.append(f"Shoulder lockout low ({shoulder_angle:.0f}°)")
        if elbow_angle < 160:
            issues.append(f"Elbow lockout low ({elbow_angle:.0f}°)")
    elif stage == "RELEASE_FINISH":
        if shoulder_angle < 150:
            issues.append("Incomplete overhead release")
    return issues


def stage_from_angles(shoulder_angle, elbow_angle, wrist_y, shoulder_y):
    """
    Heuristic stage detection:
    - Clean: bar near shoulders, elbows bent
    - Jerk: arms extending overhead
    - Release: arms fully locked out overhead
    """
    if shoulder_angle < 120 and elbow_angle < 120:
        return "CLEAN_TO_SHOULDER"
    elif shoulder_angle > 140 and elbow_angle > 140 and wrist_y < shoulder_y:
        return "JERK_OVERHEAD"
    elif shoulder_angle > 160 and elbow_angle > 160:
        return "RELEASE_FINISH"
    return None


def draw_joint_labels(frame, lm, width, height, angles):
    for name, landmark_enum in JOINT_LANDMARKS.items():
        lmk = lm[landmark_enum.value]
        if lmk.visibility < 0.4:
            continue
        cx, cy = int(lmk.x * width), int(lmk.y * height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, name, (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


# ================================
# MAIN ANALYSIS FUNCTION
# ================================
def analyze_clean_jerk_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    stage_counter = {}
    stage_issues_log = {}
    stage_window = deque(maxlen=5)

    total_frames = 0
    bad_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        current_issues = []

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            def pt(lm_enum):
                l = lm[lm_enum.value]
                return [l.x, l.y]

            hip = pt(mp_pose.PoseLandmark.LEFT_HIP)
            knee = pt(mp_pose.PoseLandmark.LEFT_KNEE)
            ankle = pt(mp_pose.PoseLandmark.LEFT_ANKLE)
            shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = pt(mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = pt(mp_pose.PoseLandmark.LEFT_WRIST)

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            shoulder_angle = 180 - calculate_angle(elbow, shoulder, hip)  # fixed outer angle
            elbow_angle = 180 - calculate_angle(shoulder, elbow, wrist)  # fixed outer angle

            stage = stage_from_angles(shoulder_angle, elbow_angle, wrist[1], shoulder[1])
            if stage:
                stage_window.append(stage)
                stage = max(set(stage_window), key=list(stage_window).count)

                stage_counter[stage] = stage_counter.get(stage, 0) + 1
                current_issues = evaluate_stage(stage, knee_angle, hip_angle, shoulder_angle, elbow_angle)

                if current_issues:
                    bad_frames += 1
                    stage_issues_log.setdefault(stage, []).append(current_issues)

            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_joint_labels(frame, lm, width, height, {})

            if stage:
                cv2.putText(frame, STAGE_LABELS[stage], (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    detected_stages = set(stage_counter.keys())
    missing = SUCCESS_STAGES - detected_stages
    form_acc = round((1 - bad_frames / max(total_frames, 1)) * 100, 2)

    if missing:
        verdict = "FAILED"
        reason = f"Missing stages: {', '.join(missing)}"
    elif form_acc < 70:
        verdict = "MARGINAL"
        reason = f"Form accuracy low ({form_acc}%)"
    else:
        verdict = "SUCCESSFUL"
        reason = f"All stages detected, form accuracy {form_acc}%"

    stage_summary = {}
    for st in STAGES:
        cnt = stage_counter.get(st, 0)
        issues = []
        for il in stage_issues_log.get(st, []):
            issues.extend(il)
        stage_summary[st] = {
            "label": STAGE_LABELS[st],
            "frame_count": cnt,
            "percentage": round(cnt / max(total_frames, 1) * 100, 1),
            "detected": cnt > 0,
            "issues": list(dict.fromkeys(issues)),
            "form_ok": len(issues) == 0
        }

    return {
        "lift_verdict": verdict,
        "verdict_reason": reason,
        "form_accuracy_percent": form_acc,
        "total_frames": total_frames,
        "bad_frames": bad_frames,
        "stage_summary": stage_summary,
        "processed_video_path": output_path
    }