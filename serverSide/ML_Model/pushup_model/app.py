import os
import cv2
import time
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

# ===== Flask setup =====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== MediaPipe setup =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ===== Helper functions (from count.py) =====
def extract_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return results.pose_landmarks.landmark, results
    return None, None

def calculate_angle(a, b, c):
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
    shoulder = landmarks[11]  # left shoulder
    hip = landmarks[23]       # left hip
    ankle = landmarks[27]     # left ankle
    angle = calculate_angle(shoulder, hip, ankle)
    score = max(0, 100 - abs(180 - angle) * 2)
    return score

def count_pushups(shoulder_y_positions, threshold=0.005):
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

# ===== Core video analysis function =====
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    shoulder_y_positions = []
    pushup_count = 0
    last_rep_time = None
    estimated_target = 10

    SMOOTHING_FACTOR = 0.2
    MAX_GAP = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, results = extract_landmarks(frame)
        if landmarks:
            score = form_score(landmarks)
            shoulder_y = landmarks[11].y
            shoulder_y_positions.append(shoulder_y)

            new_count = count_pushups(shoulder_y_positions)
            target_change = 0

            if new_count > pushup_count:
                now = time.time()
                if last_rep_time is not None:
                    rep_time = now - last_rep_time
                    if rep_time < 1:
                        target_change -= 0.5
                    elif rep_time > 4:
                        target_change -= 0.2
                    else:
                        target_change += 0.3
                last_rep_time = now
                target_change += 0.2
                pushup_count = new_count

            if score > 85:
                target_change += 0.3
            elif score < 70:
                target_change -= 0.3
            if score < 50:
                target_change -= 3

            new_estimate = estimated_target + target_change
            new_estimate = max(pushup_count, min(new_estimate, pushup_count + MAX_GAP))
            estimated_target += (new_estimate - estimated_target) * SMOOTHING_FACTOR
            estimated_target = max(5, estimated_target)

        # ⚠️ No cv2.imshow here (server mode)
    cap.release()
    return pushup_count, int(estimated_target)

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        pushups, estimated_target = analyze_video(filepath)

        return jsonify({
            "status": "success",
            "filename": filename,
            "pushups": pushups,
            "estimated_target": estimated_target
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
