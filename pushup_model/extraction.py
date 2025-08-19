import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
import logging

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO for less verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode

frames_path = "pushup_frames/wrong sequence"  # Changed for wrong sequence

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_pose_keypoints(image):
    print("[DEBUG] Extracting pose keypoints...")
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        print("[DEBUG] Pose landmarks detected.")
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints)
    print("[DEBUG] No pose landmarks found.")
    return None

data = []
labels = []

images = [f for f in os.listdir(frames_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
logging.info(f"📂 Found {len(images)} frames in 'wrong sequence'.")
print(f"[INFO] 📂 Found {len(images)} frames in 'wrong sequence'.")

for idx, file in enumerate(images, start=1):
    print(f"[DEBUG] Reading file {file} ({idx}/{len(images)})...")
    img_path = os.path.join(frames_path, file)
    image = cv2.imread(img_path)

    if image is None:
        logging.warning(f"Skipping {file} — failed to read.")
        print(f"[WARNING] Skipping {file} — failed to read.")
        continue

    print("[DEBUG] Resizing image...")
    image = cv2.resize(image, (640, 480))

    keypoints = extract_pose_keypoints(image)
    if keypoints is not None:
        print("[DEBUG] Normalizing coordinates...")
        xyz = keypoints.reshape(-1, 4)[:, :3]
        min_vals = xyz.min(axis=0)
        max_vals = xyz.max(axis=0)
        norm_xyz = (xyz - min_vals) / (max_vals - min_vals + 1e-8)

        norm_keypoints = np.hstack([norm_xyz, keypoints.reshape(-1, 4)[:, 3:4]]).flatten()
        data.append(norm_keypoints)
        labels.append(0)  # Label for wrong pushup
        print("[DEBUG] Keypoints appended.")
    else:
        print("[DEBUG] Skipped appending due to no keypoints.")

    if idx % 50 == 0 or idx == len(images):
        logging.debug(f"Processed {idx}/{len(images)} frames.")
        print(f"[DEBUG] Processed {idx}/{len(images)} frames.")

print("[DEBUG] Converting lists to NumPy arrays...")
X = np.array(data, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print("[DEBUG] Splitting data into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[DEBUG] Saving datasets...")
np.save("X_train_wrong.npy", X_train)
np.save("X_test_wrong.npy", X_test)
np.save("y_train_wrong.npy", y_train)
np.save("y_test_wrong.npy", y_test)

logging.info("✅ Processing complete for WRONG sequence!")
logging.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
print("✅ Processing complete for WRONG sequence!")
print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
