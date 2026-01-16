import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import sys

# ===============================
# PATHS (CHANGE ONLY IF NEEDED)
# ===============================
MODEL_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"
VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\testing_video\sanchuw.mp4"

# ===============================
# LOAD MODEL
# ===============================
print("📦 Loading model...")
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found:", MODEL_PATH)
    sys.exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded successfully")

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

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
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def form_score(landmarks):
    shoulder = landmarks[11]  # left shoulder
    hip = landmarks[23]       # left hip
    ankle = landmarks[27]     # left ankle
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

    return max(5, current_pred + adjustment)

# ===============================
# VIDEO OPEN
# ===============================
print("🎥 Opening video...")
if not os.path.exists(VIDEO_PATH):
    print("❌ Video not found:", VIDEO_PATH)
    sys.exit()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Failed to open video")
    sys.exit()

print("✅ Video loaded successfully")

# ===============================
# TRACKING VARIABLES
# ===============================
pushup_count = 0
state = "UP"
last_rep_time = None
predicted_max = 15
pose_detected_once = False
prev_time = time.time()

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("ℹ Video ended or frame not read")
        break

    landmarks, results = extract_landmarks(frame)

    if landmarks:
        if not pose_detected_once:
            print("🧍 Person detected – starting analysis")
            pose_detected_once = True

        features = landmarks_to_features(landmarks)

        if features.shape[1] != model.n_features_in_:
            print("❌ Feature mismatch:", features.shape[1])
            continue

        score = form_score(landmarks)
        elbow = elbow_angle(landmarks)
        elbow_label = 0 if elbow < 90 else 1

        label = model.predict(features)[0]
        proba = model.predict_proba(features)[0].max()

        if proba < 0.6:
            label = elbow_label

        # FSM rep counting
        if state == "UP" and label == 0:
            state = "DOWN"
        elif state == "DOWN" and label == 1:
            pushup_count += 1
            state = "UP"

            now = time.time()
            rep_duration = None
            if last_rep_time:
                rep_duration = now - last_rep_time
                print(f"Rep {pushup_count} | {rep_duration:.2f}s | Form {score}%")
            last_rep_time = now

            predicted_max = update_prediction(score, rep_duration, predicted_max)

        # FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # UI
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.putText(frame, f'Pushups: {pushup_count} | Form: {score}%',
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f'Estimated Max: {max(1, predicted_max-2)} - {predicted_max+2}',
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

        cv2.putText(frame, f'FPS: {fps}',
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        cv2.putText(frame, "⚠ No person detected",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.namedWindow("Push-up Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Push-up Counter", 900, 600)
    cv2.imshow("Push-up Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()

print("\n✅ SESSION COMPLETE")
print("Total Pushups:", pushup_count)
print(f"Predicted Range: {max(1, predicted_max-2)} - {predicted_max+2}")
