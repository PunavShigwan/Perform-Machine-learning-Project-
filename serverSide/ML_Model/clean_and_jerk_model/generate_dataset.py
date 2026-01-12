import cv2
import csv
import os
import mediapipe as mp
import numpy as np
from math import acos, degrees

# ===============================
# Utility functions
# ===============================

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine, -1.0, 1.0)))
    return angle


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


# ===============================
# MediaPipe setup
# ===============================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# Paths
# ===============================

VIDEO_DIR = "../../../data/clean_jerk/sample_videos/"
OUTPUT_CSV = "../../../data/clean_jerk/extracted_features/clean_jerk_dataset.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===============================
# CSV Header
# ===============================

header = [
    "elbow_angle",
    "knee_angle",
    "hip_angle",
    "stance_width",
    "bar_height_ratio",
    "stability_time",
    "symmetry_score",
    "label"  # 1 = PASS, 0 = FAIL
]

# ===============================
# Dataset Generation
# ===============================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)

        stable_frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark

            # LEFT side landmarks
            shoulder = [lm[11].x, lm[11].y]
            elbow = [lm[13].x, lm[13].y]
            wrist = [lm[15].x, lm[15].y]

            hip = [lm[23].x, lm[23].y]
            knee = [lm[25].x, lm[25].y]
            ankle = [lm[27].x, lm[27].y]

            # RIGHT side
            r_ankle = [lm[28].x, lm[28].y]

            # Angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)

            # Stance width
            stance_width = abs(ankle[0] - r_ankle[0])

            # Barbell proxy (wrist vs shoulder)
            bar_height_ratio = wrist[1] / shoulder[1]

            # Stability (lockout)
            if elbow_angle > 170 and knee_angle > 160:
                stable_frames += 1

        stability_time = stable_frames / fps

        # Symmetry score (left-right ankle)
        symmetry_score = 1 - abs(ankle[0] - r_ankle[0])

        # 🔴 MANUAL LABELING (IMPORTANT)
        label = 1 if "pass" in video_file.lower() else 0

        writer.writerow([
            round(elbow_angle, 2),
            round(knee_angle, 2),
            round(hip_angle, 2),
            round(stance_width, 3),
            round(bar_height_ratio, 3),
            round(stability_time, 2),
            round(symmetry_score, 2),
            label
        ])

        cap.release()

print("✅ Dataset generated using MediaPipe")
