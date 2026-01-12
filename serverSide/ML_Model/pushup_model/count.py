import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# ===== Load only Gradient Boosting model =====
MODEL_PATH = r"C:\major_project\pushup_model\saved_models\GradientBoosting.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===== Mediapipe Pose setup =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ===== Helper functions =====
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
    """Shoulder-hip-ankle straightness = form %"""
    shoulder = landmarks[11]  # left shoulder
    hip = landmarks[23]       # left hip
    ankle = landmarks[27]     # left ankle
    angle = calculate_angle(shoulder, hip, ankle)
    score = max(0, 100 - abs(180 - angle) * 2)  # penalty if not straight
    return int(score)

def elbow_angle(landmarks):
    """Check elbow bend to confirm up/down"""
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]
    return calculate_angle(shoulder, elbow, wrist)

# ===== Prediction Update Function =====
def update_prediction(last_form, last_duration, current_pred):
    """
    Adjust prediction after each rep:
    - Good form + reasonable speed = increase endurance
    - Poor form or very slow/fast = decrease
    """
    if last_duration is None:
        return current_pred

    # Normalize factors
    form_factor = (last_form - 70) / 30   # >70 good, <70 penalize
    speed_factor = 1.5 if last_duration < 1 else (0.8 if last_duration > 3 else 1.0)

    adjustment = int(form_factor * 2 * speed_factor)  # +/- 2 reps influence

    new_pred = max(5, current_pred + adjustment)  # clamp to at least 5
    return new_pred

# ===== Open video =====
video_path = "testing_video/sanchuw.mp4"
cap = cv2.VideoCapture(video_path)

# Tracking
pushup_count = 0
state = "UP"
last_rep_time = None
predicted_max = 15  # initial baseline

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, results = extract_landmarks(frame)
    if landmarks:
        features = landmarks_to_features(landmarks)
        score = form_score(landmarks)

        # Optional elbow fallback
        elbow = elbow_angle(landmarks)
        elbow_label = 0 if elbow < 90 else 1

        label = model.predict(features)[0]   # 0=down, 1=up
        proba = model.predict_proba(features)[0].max()

        if proba < 0.6:
            label = elbow_label

        # ===== FSM Rep Counting =====
        if state == "UP" and label == 0:
            state = "DOWN"
        elif state == "DOWN" and label == 1:
            pushup_count += 1
            state = "UP"

            now = time.time()
            rep_duration = None
            if last_rep_time:
                rep_duration = now - last_rep_time
                print(f"Rep {pushup_count} duration: {rep_duration:.2f} sec | Form: {score}%")
            last_rep_time = now

            # Update prediction dynamically
            predicted_max = update_prediction(score, rep_duration, predicted_max)

        # ===== Show Info on Screen =====
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.putText(frame, f'Pushups: {pushup_count} (Form={score}%)',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show prediction as a range
        pred_low = max(1, predicted_max - 2)
        pred_high = predicted_max + 2
        cv2.putText(frame, f'Estimated Max: {pred_low}-{pred_high}',
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)

        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        cv2.putText(frame, "No person detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Push-up Counter (Gradient Boosting)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===== Final Results =====
print("\nFinal Push-up Count:", pushup_count)
print(f"Predicted Push-up Range: {max(1, predicted_max-2)} - {predicted_max+2}")
