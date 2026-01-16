import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

print("📦 Loading pushup_service module...")

MODEL_PATH = r"C:\major_project\serverSide\ML_Model\pushup_model\saved_models\GradientBoosting.pkl"


print("📍 MODEL PATH RESOLVED TO:", MODEL_PATH)


# ----------------------------
# SAFE MODEL LOADING (LAZY)
# ----------------------------
model = None

def get_model():
    global model
    if model is None:
        print("📦 Loading ML model...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded")
    return model


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def landmarks_to_features(landmarks):
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
    ).flatten().reshape(1, -1)


# ----------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------
def analyze_pushup_video(input_path, output_path):
    try:
        print("🎬 Starting pushup analysis")
        print("📂 Input:", input_path)
        print("📂 Output:", output_path)

        model = get_model()

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        print("🧠 Initializing MediaPipe Pose")
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("❌ Failed to open input video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = 25
            print("⚠ FPS was 0, defaulting to 25")

        print(f"🎥 Video: {width}x{height} @ {fps}fps")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open")

        pushup_count = 0
        state = "UP"
        form_scores = []
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ℹ End of video")
                break

            frame_no += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = landmarks_to_features(landmarks)

                if features.shape[1] != model.n_features_in_:
                    print("⚠ Feature mismatch, skipping frame")
                    out.write(frame)
                    continue

                label = model.predict(features)[0]

                score = int(
                    max(0, 100 - abs(180 - calculate_angle(
                        landmarks[11], landmarks[23], landmarks[27]
                    )) * 2)
                )

                form_scores.append(score)

                if state == "UP" and label == 0:
                    state = "DOWN"
                elif state == "DOWN" and label == 1:
                    pushup_count += 1
                    state = "UP"
                    print(f"🏋️ Rep counted: {pushup_count}")

                cv2.putText(frame, f"Pushups: {pushup_count}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Form: {score}%",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,215,0), 2)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            out.write(frame)

            if frame_no % 50 == 0:
                print(f"⏱ Processed {frame_no} frames")

        cap.release()
        out.release()
        pose.close()

        avg_form = int(sum(form_scores) / len(form_scores)) if form_scores else 0

        print("✅ Analysis finished successfully")

        return {
            "pushup_count": pushup_count,
            "average_form": avg_form,
            "estimated_max_range": "13 - 17",
            "output_video_path": output_path
        }

    except Exception as e:
        print("🔥 FATAL ERROR IN ANALYSIS:", e)
        raise e
