import cv2
import os
import sys

# ===============================
# CONFIG (USE ABSOLUTE PATH FIRST)
# ===============================

VIDEO_PATH = r"C:\major_project\serverSide\ML_Model\clean_and_jerk_model\sample_videos\input1.mp4"
OUTPUT_DIR = r"C:\major_project\serverSide\ML_Model\data\clean_jerk\frames_30fps"
TARGET_FPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD VIDEO (HARD CHECK)
# ===============================

cap = cv2.VideoCapture(VIDEO_PATH)

print("🔍 Checking video path...")
if not os.path.exists(VIDEO_PATH):
    print("❌ ERROR: Video file does NOT exist")
    sys.exit(1)

print("📂 Video path exists")

if not cap.isOpened():
    print("❌ ERROR: OpenCV failed to open the video")
    print("👉 Possible causes:")
    print("   - Wrong codec")
    print("   - Corrupt file")
    print("   - Unsupported format")
    sys.exit(1)

print("✅ Video opened successfully")

# ===============================
# VIDEO METADATA CHECK
# ===============================

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n🎥 VIDEO INFO")
print(f"   Resolution : {width} x {height}")
print(f"   FPS        : {original_fps}")
print(f"   Frames     : {total_frames}")

if original_fps <= 0:
    print("❌ ERROR: Invalid FPS detected")
    sys.exit(1)

# ===============================
# FRAME SAMPLING LOGIC
# ===============================

frame_interval = max(1, round(original_fps / TARGET_FPS))

print(f"\n🎯 Target FPS: {TARGET_FPS}")
print(f"🎯 Extracting every {frame_interval} frame(s)\n")

# ===============================
# FRAME EXTRACTION
# ===============================

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("ℹ️ End of video reached")
        break

    if frame_id % frame_interval == 0:
        filename = f"frame_{saved:06d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
        saved += 1

    frame_id += 1

cap.release()

print(f"\n✅ SUCCESS: Extracted {saved} frames at ~{TARGET_FPS} FPS")
print(f"📁 Saved to: {OUTPUT_DIR}")
