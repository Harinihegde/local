import cv2
import json
import os
import random

random.seed(42)  # fixed seed so the sample is reproducible, not cherry-picked

VIDEOS_PATH = "psci_data/Videos"
DETECTIONS_PATH = "detection_data/pscs_detections.json"
OUTPUT_DIR = "manual_check_frames"
N_SAMPLES_PER_VIDEO = 20

with open(DETECTIONS_PATH) as f:
    det_data = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_lines = []

for video_id in ["8", "10"]:
    video_path = os.path.join(VIDEOS_PATH, f"{video_id}.avi")
    detections_per_frame = det_data[video_id]["detections"]
    total_frames = len(detections_per_frame)

    # Random sample of frame indices, not hand-picked
    sample_indices = sorted(random.sample(range(total_frames), N_SAMPLES_PER_VIDEO))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    sample_set = set(sample_indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_set:
            our_count = len(detections_per_frame[frame_idx])
            out_name = f"video{video_id}_frame{frame_idx}_ourcount{our_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
            summary_lines.append(f"video{video_id}, frame {frame_idx}: our detector counted {our_count}")
        frame_idx += 1
    cap.release()

    print(f"Video {video_id}: saved {N_SAMPLES_PER_VIDEO} sample frames")

with open(os.path.join(OUTPUT_DIR, "_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll sample frames + summary saved to: {OUTPUT_DIR}/")
print("Open each image, count the real people by eye, and compare to the")
print("'ourcount' number already in the filename.")