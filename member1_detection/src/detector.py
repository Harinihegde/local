import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch
from tqdm import tqdm
import json

class PersonDetector:

    def __init__(self, model_path, conf_threshold=0.8): 
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.cached_clahe_frame = None

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

        self.frame_count += 1
        if self.frame_count % 5 == 0:
            lab = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            self.cached_clahe_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            if self.cached_clahe_frame is None:
                self.cached_clahe_frame = frame_filtered

        return self.cached_clahe_frame, scale, top, left

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

    def process_sequence(self, sequence_folder):
        frames = sorted([f for f in os.listdir(sequence_folder) if f.endswith('.tif')])

        if len(frames) == 0:
            print(f"No .tif frames found in {sequence_folder}")
            return [], 0

        all_detections = []
        start_time = time.time()
        self.frame_count = 0
        self.cached_clahe_frame = None

        for frame_file in frames:
            frame_path = os.path.join(sequence_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                all_detections.append([])
                continue
            bboxes = self.detect_people(frame)
            all_detections.append([list(bbox) for bbox in bboxes])

        elapsed = time.time() - start_time
        fps = len(frames) / elapsed if elapsed > 0 else 0
        return all_detections, fps

# ================================
# SETUP
# ================================
DATASET_PATH = '/Users/harinihegde/Downloads/UCSD_Anomaly_Dataset/UCSDped2'
TRAIN_PATH = os.path.join(DATASET_PATH, 'Train')
TEST_PATH  = os.path.join(DATASET_PATH, 'Test')

detector = PersonDetector(
    model_path='/Users/harinihegde/Downloads/stampede-detection-system-main/models/best_combined.pt',
    conf_threshold=0.8 # ← CHANGE HERE (1/1)
)

# ================================
# GET SEQUENCES
# ================================
train_sequences = sorted([
    f for f in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, f))
    and not f.endswith('_gt') and not f.startswith('.')
])

test_sequences = sorted([
    f for f in os.listdir(TEST_PATH)
    if os.path.isdir(os.path.join(TEST_PATH, f))
    and not f.endswith('_gt') and not f.startswith('.')
])

print(f"Found {len(train_sequences)} training sequences")
print(f"Found {len(test_sequences)} testing sequences")

# ================================
# PROCESS
# ================================
all_results = {'train': {}, 'test': {}}

print("\nProcessing TRAINING sequences...")
for seq_name in tqdm(train_sequences, desc="Training"):
    seq_path = os.path.join(TRAIN_PATH, seq_name)
    all_detections, processing_fps = detector.process_sequence(seq_path)
    all_results['train'][seq_name] = {
        'detections':       all_detections,
        'fps':              processing_fps,
        'total_frames':     len(all_detections),
        'total_detections': sum(len(d) for d in all_detections)
    }

print("\nProcessing TESTING sequences...")
for seq_name in tqdm(test_sequences, desc="Testing"):
    seq_path = os.path.join(TEST_PATH, seq_name)
    all_detections, processing_fps = detector.process_sequence(seq_path)
    all_results['test'][seq_name] = {
        'detections':       all_detections,
        'fps':              processing_fps,
        'total_frames':     len(all_detections),
        'total_detections': sum(len(d) for d in all_detections)
    }

# ================================
# SAVE
# ================================
os.makedirs('detection_data', exist_ok=True)

with open('detection_data/train_combined_detections.json', 'w') as f:
    json.dump(all_results['train'], f, indent=2)

with open('detection_data/test_combined_detections.json', 'w') as f:
    json.dump(all_results['test'], f, indent=2)

print("\n✓ Train detections saved: detection_data/train_detections.json")
print("✓ Test detections saved:  detection_data/test_detections.json")
