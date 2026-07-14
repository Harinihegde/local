import json
import numpy as np

with open('detection_data/test_combined_detections.json', 'r') as f:
    data = json.load(f)

total_frames = 0
total_detections = 0
fps_list = []

for seq_name, seq_data in data.items():
    fps_list.append(seq_data['fps'])
    total_frames += seq_data['total_frames']
    total_detections += seq_data['total_detections']

print(f"=== Original (best.pt) ===")
print(f"Avg detections/frame: {total_detections/total_frames:.2f}")
print(f"Avg FPS: {np.mean(fps_list):.2f}")
print(f"Total detections: {total_detections}")