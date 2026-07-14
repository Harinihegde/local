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
 
    # ─── CHANGED: reads an .avi video frame-by-frame instead of a folder of .tif images ───
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return [], 0
 
        all_detections = []
        start_time = time.time()
        self.frame_count = 0
        self.cached_clahe_frame = None
 
        n_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break            # no more frames
            n_frames += 1
            bboxes = self.detect_people(frame)
            all_detections.append([list(bbox) for bbox in bboxes])
 
        cap.release()
        elapsed = time.time() - start_time
        fps = n_frames / elapsed if elapsed > 0 else 0
        return all_detections, fps
 
 
# ================================
# SETUP
# ================================
AVENUE_PATH = '/Users/harinihegde/Downloads/Avenue_Dataset'   # ← adjust if different
TRAIN_PATH  = os.path.join(AVENUE_PATH, 'training_videos')
TEST_PATH   = os.path.join(AVENUE_PATH, 'testing_videos')
 
detector = PersonDetector(
    model_path='/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/models/best_combined.pt',
    conf_threshold=0.25
)
 
# ================================
# GET VIDEO FILES
# ================================
def list_avis(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith('.avi')])
 
# ── TEST-RUN SWITCH ──────────────────────────────────────────────
# Set TEST_ONE_VIDEO = True to process just 01.avi first (fast sanity check).
# Once that looks right, set it back to False to process everything.
TEST_ONE_VIDEO = False
# ─────────────────────────────────────────────────────────────────
 
all_results = {'train': {}, 'test': {}}
 
if TEST_ONE_VIDEO:
    print("\n[TEST RUN] Processing only 01.avi from testing_videos...")
    detections, fps = detector.process_video(os.path.join(TEST_PATH, '01.avi'))
    all_results['test']['01'] = {
        'detections':       detections,
        'fps':              fps,
        'total_frames':     len(detections),
        'total_detections': sum(len(d) for d in detections)
    }
    print(f"  frames: {len(detections)} | total detections: {sum(len(d) for d in detections)} | fps: {fps:.2f}")
    print("  If these numbers look sane, set TEST_ONE_VIDEO = False and re-run for all videos.")
 
else:
    train_videos = list_avis(TRAIN_PATH)
    test_videos  = list_avis(TEST_PATH)
    print(f"Found {len(train_videos)} training videos")
    print(f"Found {len(test_videos)} testing videos")
 
    print("\nProcessing TRAINING videos...")
    for vid in tqdm(train_videos, desc="Training"):
        key = os.path.splitext(vid)[0]          # "01.avi" -> "01"
        detections, fps = detector.process_video(os.path.join(TRAIN_PATH, vid))
        all_results['train'][key] = {
            'detections':       detections,
            'fps':              fps,
            'total_frames':     len(detections),
            'total_detections': sum(len(d) for d in detections)
        }
 
    print("\nProcessing TESTING videos...")
    for vid in tqdm(test_videos, desc="Testing"):
        key = os.path.splitext(vid)[0]
        detections, fps = detector.process_video(os.path.join(TEST_PATH, vid))
        all_results['test'][key] = {
            'detections':       detections,
            'fps':              fps,
            'total_frames':     len(detections),
            'total_detections': sum(len(d) for d in detections)
        }
 
# ================================
# SAVE
# ================================
os.makedirs('detection_data', exist_ok=True)
 
with open('detection_data/avenue_train_detections.json', 'w') as f:
    json.dump(all_results['train'], f, indent=2)
 
with open('detection_data/avenue_test_detections.json', 'w') as f:
    json.dump(all_results['test'], f, indent=2)
 
print("\n✓ Avenue train detections saved: detection_data/avenue_train_detections.json")
print("✓ Avenue test detections saved:  detection_data/avenue_test_detections.json")
 