import cv2
import os

# Paths
dataset_path = "pushup_dataset"
output_path = "pushup_frames"

# Create output directories for both categories
categories = ["correct sequence", "wrong sequence"]
for category in categories:
    os.makedirs(os.path.join(output_path, category), exist_ok=True)

# Desired frame rate
target_fps = 30

# Loop through each category
for category in categories:
    video_folder = os.path.join(dataset_path, category)
    output_folder = os.path.join(output_path, category)

    # Loop through each video in the category
    for video_file in os.listdir(video_folder):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Could not open {video_file}")
            continue

        # Get original FPS
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(orig_fps / target_fps)) if orig_fps > 0 else 1

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"✅ Processed {video_file} → {saved_count} frames saved in '{category}'")

print("\n🎯 Frame extraction complete!")

'''
📊 correct sequence: 4503 frames
📊 wrong sequence: 6425 frames
'''