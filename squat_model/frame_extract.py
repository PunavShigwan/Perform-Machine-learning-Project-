import cv2
import os

# === Input video folder ===
video_folder = r"C:\major_project\squat_model\dataset_video"
output_folder = r"C:\major_project\squat_model\frames"

# === Create output folder if not exists ===
os.makedirs(output_folder, exist_ok=True)

# === Loop through all videos ===
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

total_frames = 0

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    success = True

    print(f"\n🎬 Processing video: {video_file}")

    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # Save frame with video name prefix
        frame_filename = os.path.join(
            output_folder,
            f"{os.path.splitext(video_file)[0]}_frame_{frame_count:05d}.jpg"
        )
        cv2.imwrite(frame_filename, frame)
        
        print(f"[✔] Saved {frame_filename}")
        
        frame_count += 1
        total_frames += 1

    cap.release()
    print(f"✅ Finished {video_file}, frames saved: {frame_count}")

print(f"\n🏁 All videos processed! Total frames extracted: {total_frames}")
