import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split

# === Paths ===
frame_folder = r"C:\major_project\squat_model\frames"
output_folder = r"C:\major_project\squat_model\features"

# Create output folders
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# === Mediapipe Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# === Collect features ===
all_features = []
all_labels = []

frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(('.jpg', '.png'))]

print(f"🔎 Found {len(frame_files)} frames in dataset.")

for idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frame_folder, frame_file)
    image = cv2.imread(frame_path)

    if image is None:
        print(f"⚠️ Skipping unreadable file: {frame_file}")
        continue

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract x, y, z, visibility for each landmark (132 features)
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])

        all_features.append(features)
        all_labels.append(frame_file)  # store frame filename as label

        print(f"[✔] Processed frame: {frame_file}")

print(f"\n✅ Feature extraction done for {len(all_features)} frames.")

# === Convert to NumPy arrays ===
X = np.array(all_features)   # Shape: (num_samples, 132)
y = np.array(all_labels)     # Shape: (num_samples,)

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Labels shape: {y.shape}")
print(f"🔖 Example label: {y[0]}")

# === Train/Test Split (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📂 Training samples: {X_train.shape[0]}")
print(f"📂 Testing samples: {X_test.shape[0]}")

# === Save to .npy files ===
np.save(os.path.join(train_folder, "train.npy"), {"X": X_train, "y": y_train})
np.save(os.path.join(test_folder, "test.npy"), {"X": X_test, "y": y_test})

print(f"\n💾 Saved training data to: {os.path.join(train_folder, 'train.npy')}")
print(f"💾 Saved testing data to: {os.path.join(test_folder, 'test.npy')}")
print("\n🏁 Feature extraction + dataset split complete!")

