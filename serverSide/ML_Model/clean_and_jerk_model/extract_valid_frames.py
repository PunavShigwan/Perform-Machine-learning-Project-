import cv2
import os
import sys
import mediapipe as mp

# ===============================
# CONFIG
# ===============================

VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\sample_videos\input1.mp4"
OUTPUT_DIR = r"C:\major_project\serverSide\ML_Model\data\clean_jerk\frames_30fps_front"
TARGET_FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# MEDIAPIPE INIT
# ===============================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.2,   # VERY RELAXED
    min_tracking_confidence=0.2
)

# ===============================
# LOAD VIDEO
# ===============================

print("🔍 Checking video path...")

if not os.path.exists(VIDEO_PATH):
    print("❌ ERROR: Video file does not exist")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ ERROR: OpenCV failed to open video")
    sys.exit(1)

original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, round(original_fps / TARGET_FPS))

print("📂 Video loaded successfully")
print(f"🎥 FPS            : {original_fps}")
print(f"🎯 Target FPS     : {TARGET_FPS}")
print(f"🎯 Frame interval : {frame_interval}\n")

# ===============================
# ULTRA-RELAXED FRONT VIEW CHECK
# (LOWER BODY ONLY)
# ===============================

def is_front_facing(landmarks):
    try:
        l_hip  = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip  = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # Only check that left/right are horizontally separated
        hip_sep  = abs(l_hip.x - r_hip.x)
        knee_sep = abs(l_knee.x - r_knee.x)

        # Extremely relaxed thresholds
        if hip_sep > 0.03 and knee_sep > 0.03:
            return True

        return False

    except:
        return False

# ===============================
# FRAME EXTRACTION
# ===============================

frame_id = 0
checked = 0
saved = 0

print("🚀 Starting frame extraction...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nℹ️ End of video reached")
        break

    if frame_id % frame_interval == 0:
        checked += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            if is_front_facing(results.pose_landmarks.landmark):
                filename = f"frame_{saved:06d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)

                # ✅ SAVE ORIGINAL FRAME (NO ZOOM, NO CROP)
                cv2.imwrite(save_path, frame)
                saved += 1

                print(f"✅ SAVED | Frame #{frame_id} → {filename}")

    frame_id += 1

cap.release()
pose.close()

# ===============================
# FINAL REPORT
# ===============================

print("\n===============================")
print("🎉 EXTRACTION COMPLETE")
print("===============================")
print(f"🔍 Frames checked : {checked}")
print(f"💾 Frames saved   : {saved}")
print(f"📁 Output folder  : {OUTPUT_DIR}")
print("===============================\n")
