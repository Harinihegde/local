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
 
        # CLAHE applied to every frame using that frame's own pixels.
        # (Previously cached every 5th frame and reused stale pixels for the
        # other 4/5 frames — fed the wrong image to YOLO most of the time.)
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
 
    # reads a FOLDER of .jpg frames (UMN scenes are frame folders, like the UCSD .tif setup)
    def process_sequence(self, sequence_folder):
        frames = sorted([f for f in os.listdir(sequence_folder) if f.endswith('.jpg')])
 
        if len(frames) == 0:
            print(f"No .jpg frames found in {sequence_folder}")
            return [], 0
 
        all_detections = []
        start_time = time.time()
 
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
UMN_PATH = '/Users/harinihegde/Downloads/umn'   # ← adjust to where your umn folder is
SCENES = ['scene1', 'scene2', 'scene3']
 
CONF = 0.25   # ← the value you're trying. change and re-run to compare.
 
detector = PersonDetector(
    model_path='/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/models/best_combined.pt',
    conf_threshold=CONF
)
 
# ── TEST-ONE-SCENE SWITCH ──────────────────────────────────────
# True  = process only scene1 (fast, for trying a threshold)
# False = process all 3 scenes and save the JSON
TEST_ONE_SCENE = False
# ───────────────────────────────────────────────────────────────
 
all_results = {}
 
if TEST_ONE_SCENE:
    print(f"\n[TEST RUN] scene1 only, conf={CONF}...")
    detections, fps = detector.process_sequence(os.path.join(UMN_PATH, 'scene1'))
    total = sum(len(d) for d in detections)
    print(f"  frames: {len(detections)} | total detections: {total} | "
          f"avg/frame: {total/max(len(detections),1):.1f} | fps: {fps:.2f}")
    print("  If this looks sane, set TEST_ONE_SCENE = False and re-run to save all 3 scenes.")
 
else:
    for scene in SCENES:
        print(f"\nProcessing {scene} (conf={CONF})...")
        detections, fps = detector.process_sequence(os.path.join(UMN_PATH, scene))
        all_results[scene] = {
            'detections':       detections,
            'fps':              fps,
            'total_frames':     len(detections),
            'total_detections': sum(len(d) for d in detections)
        }
        print(f"  {scene}: {len(detections)} frames, "
              f"{all_results[scene]['total_detections']} detections")
 
    os.makedirs('detection_data', exist_ok=True)
    with open('detection_data/umn_detections.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ UMN detections saved: detection_data/umn_detections.json")