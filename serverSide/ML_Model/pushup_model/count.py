import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import sys
from collections import deque

# ===============================
# PATHS
# ===============================
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"
VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\testing_video\test.mp4"

# ===============================
# CONFIG
# ===============================
ELBOW_DOWN_THRESH = 90
ELBOW_UP_THRESH = 155
SMOOTHING_WINDOW = 5
FRAME_CONFIRM = 2
MIN_REP_TIME = 0.6

# ===============================
# LOAD MODEL
# ===============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===============================
# MEDIAPIPE
# ===============================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# ===============================
# HELPERS
# ===============================
def angle(a, b, c):
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    ang = abs(np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) -
                          np.arctan2(a[1]-b[1], a[0]-b[0])))
    return 360-ang if ang > 180 else ang

def elbow_angle(lm):
    return angle(lm[11], lm[13], lm[15])

def form_score(lm):
    shoulder, hip, ankle = lm[11], lm[23], lm[27]
    body_angle = angle(shoulder, hip, ankle)
    return int(np.clip(100 - abs(180 - body_angle) * 1.8, 0, 100))

def rep_rating(score):
    if score >= 85:
        return "Excellent", 2
    elif score >= 70:
        return "Good", 1
    elif score >= 50:
        return "Poor", 0
    else:
        return "Bad", -2

# ===============================
# VIDEO
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)

# ===============================
# STATE
# ===============================
state = "UP"
pushup_count = 0
predicted_max = 15
last_rep_time = None

elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
down_frames = 0
up_frames = 0

# Rep-level form tracking
down_form_scores = []
up_form_scores = []

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        elbow = elbow_angle(lm)
        elbow_buf.append(elbow)
        smooth_elbow = np.mean(elbow_buf)

        score = form_score(lm)

        # ===============================
        # FSM WITH FORM COLLECTION
        # ===============================
        if smooth_elbow < ELBOW_DOWN_THRESH:
            down_frames += 1
            up_frames = 0
            if state == "DOWN":
                down_form_scores.append(score)

        elif smooth_elbow > ELBOW_UP_THRESH:
            up_frames += 1
            down_frames = 0
            if state == "UP":
                up_form_scores.append(score)

        else:
            down_frames = up_frames = 0

        if state == "UP" and down_frames >= FRAME_CONFIRM:
            state = "DOWN"
            down_form_scores = []

        elif state == "DOWN" and up_frames >= FRAME_CONFIRM:
            now = time.time()
            rep_time = now - last_rep_time if last_rep_time else None

            # ===============================
            # COMPLETE REP
            # ===============================
            pushup_count += 1
            last_rep_time = now

            avg_down = np.mean(down_form_scores) if down_form_scores else score
            avg_up = np.mean(up_form_scores) if up_form_scores else score
            rep_form = int((avg_down + avg_up) / 2)

            rating, delta = rep_rating(rep_form)
            predicted_max = max(5, predicted_max + delta)

            print(
                f"Rep {pushup_count} | "
                f"Form {rep_form}% ({rating}) | "
                f"Range ~ {predicted_max-2} - {predicted_max+2}"
            )

            up_form_scores = []
            state = "UP"

        # ===============================
        # UI
        # ===============================
        cv2.putText(frame, f"Pushups: {pushup_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"State: {state}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Est. Max: {predicted_max-2} - {predicted_max+2}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,215,0), 2)

        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Push-up Rep Rating", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ SESSION COMPLETE")
print("Total Pushups:", pushup_count)
print(f"Final Estimated Range: {predicted_max-2} - {predicted_max+2}")
