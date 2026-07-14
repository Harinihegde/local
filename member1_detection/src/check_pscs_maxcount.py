import json

with open("psci_data/environment_annot.json") as f:
    env_data = json.load(f)

with open("detection_data/pscs_detections.json") as f:
    det_data = json.load(f)

print(f"{'Video':<8} {'Dataset max_count':>18} {'Our max detected/frame':>24} {'Our avg/frame':>15}")
print("=" * 70)

for video_id in sorted(det_data.keys(), key=int):
    dataset_max = env_data.get(video_id, {}).get("max_count", "?")

    detections_per_frame = det_data[video_id]["detections"]
    per_frame_counts = [len(frame) for frame in detections_per_frame]
    our_max = max(per_frame_counts) if per_frame_counts else 0
    our_avg = sum(per_frame_counts) / len(per_frame_counts) if per_frame_counts else 0

    flag = ""
    if isinstance(dataset_max, (int, float)):
        if our_max > dataset_max * 1.3:
            flag = "  <-- WAY MORE than dataset max (possible false positives)"
        elif our_max < dataset_max * 0.5:
            flag = "  <-- WAY LESS than dataset max (possible missed detections)"

    print(f"{video_id:<8} {dataset_max:>18} {our_max:>24} {our_avg:>15.1f}{flag}")