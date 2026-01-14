import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG (ABSOLUTE PATHS RECOMMENDED)
# =====================================================
FRAME_DIR = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\frames_30fps_front"
OUTPUT_DIR = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\cj_sorted_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# MEDIAPIPE INIT (CPU SAFE)
# =====================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def is_image(file):
    return file.lower().endswith((".jpg", ".jpeg", ".png"))


# =====================================================
# STEP 1: LOAD FRAMES (HARD CHECK)
# =====================================================
print("\n========== LOADING FRAMES ==========")
print("FRAME DIR:", FRAME_DIR)

if not os.path.exists(FRAME_DIR):
    raise RuntimeError("❌ FRAME DIRECTORY DOES NOT EXIST")

frames = sorted([f for f in os.listdir(FRAME_DIR) if is_image(f)])

print("Total files in directory:", len(os.listdir(FRAME_DIR)))
print("Total image frames found:", len(frames))

if len(frames) == 0:
    raise RuntimeError("❌ NO IMAGE FRAMES FOUND")

# =====================================================
# STEP 2: FEATURE EXTRACTION
# =====================================================
print("\n========== EXTRACTING FEATURES ==========")

data = []
valid = 0
skipped = 0

for idx, frame_name in enumerate(frames):
    frame_path = os.path.join(FRAME_DIR, frame_name)
    image = cv2.imread(frame_path)

    if image is None:
        skipped += 1
        print("❌ Failed to read:", frame_name)
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        skipped += 1
        print("⚠ No pose detected:", frame_name)
        continue

    lm = result.pose_landmarks.landmark

    try:
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)

        data.append({
            "frame": frame_name,
            "index": idx,
            "knee_angle": knee_angle,
            "hip_y": hip[1]
        })

        valid += 1
        print("✔ Pose OK:", frame_name)

    except Exception as e:
        skipped += 1
        print("❌ Error processing:", frame_name, e)

print("\nVALID FRAMES:", valid)
print("SKIPPED FRAMES:", skipped)

if valid < 10:
    raise RuntimeError("❌ NOT ENOUGH VALID FRAMES FOR TRAINING")

df = pd.DataFrame(data)

# =====================================================
# STEP 3: TEMPORAL FEATURES
# =====================================================
print("\n========== TEMPORAL FEATURES ==========")

df["hip_velocity"] = df["hip_y"].diff().fillna(0)
df["knee_velocity"] = df["knee_angle"].diff().fillna(0)

df["knee_angle_smooth"] = df["knee_angle"].rolling(5, min_periods=1).mean()
df["hip_velocity_smooth"] = df["hip_velocity"].rolling(5, min_periods=1).mean()

# =====================================================
# STEP 4: PHASE CLASSIFICATION
# =====================================================
print("\n========== PHASE CLASSIFICATION ==========")

def classify_phase(row):
    ka = row.knee_angle_smooth
    hv = row.hip_velocity_smooth

    if ka > 165:
        return 0  # Setup
    elif 150 < ka <= 165:
        return 1  # First Pull
    elif hv < -0.02 and ka > 140:
        return 2  # Second Pull
    elif ka < 120:
        return 3  # Catch
    elif ka > 150 and hv > 0:
        return 4  # Recovery
    elif 130 < ka < 150:
        return 5  # Dip
    elif hv < -0.03:
        return 6  # Drive
    elif ka > 170:
        return 7  # Lockout
    else:
        return 8  # Finish

df["label"] = df.apply(classify_phase, axis=1)

PHASE_MAP = {
    0: "P1_setup",
    1: "P2_first_pull",
    2: "P3_second_pull",
    3: "P4_catch",
    4: "P5_recovery",
    5: "P6_dip",
    6: "P7_drive",
    7: "P8_lockout",
    8: "P9_finish"
}

df["phase"] = df["label"].map(PHASE_MAP)

print("Phase counts:\n", df["phase"].value_counts())

# =====================================================
# STEP 5: SORT FRAMES INTO PHASE FOLDERS
# =====================================================
print("\n========== SORTING FRAMES ==========")

for _, row in df.iterrows():
    phase_dir = os.path.join(OUTPUT_DIR, row.phase)
    os.makedirs(phase_dir, exist_ok=True)

    src = os.path.join(FRAME_DIR, row.frame)
    dst = os.path.join(phase_dir, row.frame)

    img = cv2.imread(src)
    if img is not None:
        cv2.imwrite(dst, img)

# =====================================================
# STEP 6: CREATE .NPY DATASET
# =====================================================
print("\n========== CREATING DATASET ==========")

FEATURES = df[[
    "knee_angle_smooth",
    "hip_velocity_smooth",
    "knee_velocity"
]].values.astype(np.float32)

LABELS = df["label"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    FEATURES, LABELS, test_size=0.2, random_state=42, shuffle=True
)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# =====================================================
# STEP 7: SAVE CSV
# =====================================================
csv_path = os.path.join(OUTPUT_DIR, "cj_phase_data.csv")
df.to_csv(csv_path, index=False)

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
print("CSV:", csv_path)
print("NPY FILES SAVED IN:", OUTPUT_DIR)
