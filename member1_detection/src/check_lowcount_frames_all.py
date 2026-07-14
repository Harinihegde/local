import cv2
import json
import os

DETECTIONS_PATH = "detection_data/pscs_detections.json"
VIDEOS_PATH = "psci_data/Videos"
OUTPUT_DIR = "manual_check_lowcount_frames"
N_FRAMES_PER_VIDEO = 5

with open(DETECTIONS_PATH) as f:
    det_data = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_lines = []

for video_id in [str(i) for i in range(1, 13)]:
    detections_per_frame = det_data[video_id]["detections"]

    # Find the N frames with the FEWEST detected people (but not zero,
    # since an empty frame tells us nothing useful to verify).
    frame_counts = [(i, len(dets)) for i, dets in enumerate(detections_per_frame)]
    frame_counts = [fc for fc in frame_counts if fc[1] > 0]
    frame_counts.sort(key=lambda x: x[1])
    lowest_frames = frame_counts[:N_FRAMES_PER_VIDEO]

    if not lowest_frames:
        print(f"Video {video_id}: no non-zero-count frames found at all — flag separately")
        summary_lines.append(f"video{video_id}: NO detections in entire video")
        continue

    target_indices = {idx for idx, _ in lowest_frames}
    counts_by_idx = dict(lowest_frames)

    cap = cv2.VideoCapture(f"{VIDEOS_PATH}/{video_id}.avi")
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            our_count = counts_by_idx[frame_idx]
            out_name = f"video{video_id}_frame{frame_idx}_ourcount{our_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
            summary_lines.append(f"video{video_id}, frame {frame_idx}: our detector counted {our_count}")
            saved += 1
        frame_idx += 1
    cap.release()

    print(f"Video {video_id}: saved {saved} low-count frames "
          f"(counts: {sorted(counts_by_idx.values())})")

with open(os.path.join(OUTPUT_DIR, "_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll frames saved to: {OUTPUT_DIR}/")
print("For each image: open it, count the real people, compare to the")
print("'ourcount' number already in the filename.")