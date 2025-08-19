import cv2
import mediapipe as mp
import numpy as np
import time

# ===== User input =====
age = int(input("Enter age: "))
weight = float(input("Enter weight (kg): "))
height = float(input("Enter height (cm): "))

# ===== MediaPipe setup =====
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

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def form_score(landmarks):
    """Score form based on body alignment"""
    shoulder = landmarks[11]  # left shoulder
    hip = landmarks[23]       # left hip
    ankle = landmarks[27]     # left ankle
    angle = calculate_angle(shoulder, hip, ankle)
    score = max(0, 100 - abs(180 - angle) * 2)  # perfect straight line = 180°
    return score

def count_pushups(shoulder_y_positions, threshold=0.005):
    """Count push-ups from vertical shoulder movement"""
    count = 0
    state = "up"
    mean_y = np.mean(shoulder_y_positions)
    for y in shoulder_y_positions:
        if state == "up" and y > mean_y + threshold:
            state = "down"
        elif state == "down" and y < mean_y - threshold:
            state = "up"
            count += 1
    return count

# ===== Open video =====
video_path = "testing_video/test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# ===== Tracking variables =====
shoulder_y_positions = []
pushup_count = 0
last_rep_time = None
estimated_target = 10  # starting target

# ===== Smoothing parameters =====
SMOOTHING_FACTOR = 0.2  # smaller = smoother changes
MAX_GAP = 3  # max allowed gap between estimate and real count

# ===== Main loop =====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, results = extract_landmarks(frame)
    if landmarks:
        score = form_score(landmarks)
        shoulder_y = landmarks[11].y
        shoulder_y_positions.append(shoulder_y)

        # Count pushups
        new_count = count_pushups(shoulder_y_positions)
        target_change = 0  # default change

        if new_count > pushup_count:
            now = time.time()

            # Speed check
            if last_rep_time is not None:
                rep_time = now - last_rep_time
                if rep_time < 1:
                    target_change -= 0.5
                elif rep_time > 4:
                    target_change -= 0.2
                else:
                    target_change += 0.3
            last_rep_time = now

            # Count effect
            target_change += 0.2
            pushup_count = new_count

        # Form effect
        if score > 85:
            target_change += 0.3
        elif score < 70:
            target_change -= 0.3
        if score < 50:
            target_change -= 3  # huge penalty for bad form

        # Smooth the change
        new_estimate = estimated_target + target_change
        # Keep estimate within MAX_GAP of real count
        new_estimate = max(pushup_count, min(new_estimate, pushup_count + MAX_GAP))
        # Apply smoothing factor
        estimated_target += (new_estimate - estimated_target) * SMOOTHING_FACTOR
        estimated_target = max(5, estimated_target)

        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Text color based on form
        color = (0, 255, 0) if score >= 70 else (0, 0, 255)

        # Overlay text
        cv2.putText(frame, f'Push-ups: {pushup_count}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f'Score: {int(score)}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f'Estimated Target: {int(estimated_target)}', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        cv2.putText(frame, "No person detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Push-up Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
