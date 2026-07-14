import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch
from tqdm import tqdm
import json

class PersonDetector:

    def __init__(self, model_path, conf_threshold=0.25):
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"✓ Using device: {self.device}")
        print("✓ Model loaded successfully!")

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        top = (640 - new_h) // 2
        bottom = 640 - new_h - top
        left = (640 - new_w) // 2
        right = 640 - new_w - left
        frame_letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        frame_filtered = cv2.bilateralFilter(frame_letterboxed, d=5, sigmaColor=50, sigmaSpace=50)

        # CLAHE applied to every frame using that frame's own pixels
        # (no stale-frame caching — same fix applied to all our detectors today).
        lab = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame_clahe, scale, top, left

    def detect_people(self, frame):
        processed, scale, pad_top, pad_left = self.preprocess(frame)

        results = self.model.predict(
            source=processed,
            classes=[0],
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )

        bboxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                x1 = int((x1 - pad_left) / scale)
                y1 = int((y1 - pad_top) / scale)
                x2 = int((x2 - pad_left) / scale)
                y2 = int((y2 - pad_top) / scale)
                bboxes.append((x1, y1, x2, y2, confidence))

        return bboxes

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return [], 0

        all_detections = []
        start_time = time.time()

        n_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n_frames += 1
            bboxes = self.detect_people(frame)
            all_detections.append([list(bbox) for bbox in bboxes])
            if n_frames % 200 == 0:
                elapsed = time.time() - start_time
                print(f"    ...{n_frames} frames done ({elapsed:.1f}s elapsed, "
                      f"{n_frames/elapsed:.1f} fps)")

        cap.release()
        elapsed = time.time() - start_time
        fps = n_frames / elapsed if elapsed > 0 else 0
        return all_detections, fps


# ================================
# SETUP
# ================================
PSCS_VIDEOS_PATH = 'psci_data/Videos'   # ← adjust if your path differs

# NOTE: conf_threshold=0.25 was tuned/confirmed for UMN's overhead, fixed
# camera. This is a DIFFERENT camera style (body-worn, wide FOV, indoor
# environments) — 0.25 is a reasonable starting point, carried over from
# UMN/Avenue, but NOT yet verified as correct for this new footage. Worth
# sanity-checking detection counts before trusting downstream results.
detector = PersonDetector(
    model_path='models/best_combined.pt',   # ← adjust to your real model path
    conf_threshold=0.25
)

# ================================
# GET VIDEO FILES — numbered 1.avi through 12.avi, matching the
# annotation JSON's video IDs directly (no train/test subfolders here;
# the split assignment comes from behavior_analysis_annot.json instead).
# ================================
video_files = sorted(
    [f for f in os.listdir(PSCS_VIDEOS_PATH) if f.endswith('.avi')],
    key=lambda f: int(os.path.splitext(f)[0])  # sort numerically: 1,2,...,12 not 1,10,11,...
)
print(f"Found {len(video_files)} PSCS-I videos: {video_files}")

all_results = {}

for vid_file in tqdm(video_files, desc="Processing PSCS-I videos"):
    video_id = os.path.splitext(vid_file)[0]  # "1.avi" -> "1", matches annotation JSON keys
    detections, fps = detector.process_video(os.path.join(PSCS_VIDEOS_PATH, vid_file))
    all_results[video_id] = {
        'detections':       detections,
        'fps':              fps,
        'total_frames':     len(detections),
        'total_detections': sum(len(d) for d in detections)
    }
    print(f"  Video {video_id}: {len(detections)} frames, "
          f"{all_results[video_id]['total_detections']} detections")

# ================================
# SAVE
# ================================
os.makedirs('detection_data', exist_ok=True)
with open('detection_data/pscs_detections.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✓ PSCS-I detections saved: detection_data/pscs_detections.json")