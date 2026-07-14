"""
check_umn_frames.py — draw detection boxes on several UMN scene1 frames
(a mix of calm frames and panic-onset frames) so you can eyeball conf=0.25.
Saves images to check_umn_frames/. Standalone.
"""
import cv2
import os
import torch
from ultralytics import YOLO

# ---- settings ----
MODEL_PATH = '/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/models/best_combined.pt'
SCENE_DIR  = '/Users/harinihegde/Downloads/umn/scene1'   # adjust if needed
CONF = 0.25
# panic in scene1 starts ~frame 1132; sample calm + panic frames
FRAMES_TO_SAVE = [100, 500, 1000, 1131, 1200, 1300]

# ---- detector setup (same preprocessing/detection as your detector) ----
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device: {device}")
model = YOLO(MODEL_PATH)

_fc = 0
_cache = None

def preprocess(frame):
    global _fc, _cache
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    top = (640 - new_h) // 2
    bottom = 640 - new_h - top
    left = (640 - new_w) // 2
    right = 640 - new_w - left
    lb = cv2.copyMakeBorder(resized, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    filt = cv2.bilateralFilter(lb, d=5, sigmaColor=50, sigmaSpace=50)
    _fc += 1
    if _fc % 5 == 0:
        lab = cv2.cvtColor(filt, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        _cache = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    elif _cache is None:
        _cache = filt
    return _cache, scale, top, left

def detect(frame):
    proc, scale, pad_top, pad_left = preprocess(frame)
    res = model.predict(source=proc, classes=[0], conf=CONF, verbose=False, device=device)
    boxes = []
    if len(res) > 0 and res[0].boxes is not None:
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0].cpu().numpy())
            x1 = int((x1 - pad_left) / scale); y1 = int((y1 - pad_top) / scale)
            x2 = int((x2 - pad_left) / scale); y2 = int((y2 - pad_top) / scale)
            boxes.append((x1, y1, x2, y2, c))
    return boxes

# ---- frames are named frame_00000.jpg etc ----
os.makedirs('check_umn_frames', exist_ok=True)
for idx in FRAMES_TO_SAVE:
    fname = f"frame_{idx:05d}.jpg"
    path = os.path.join(SCENE_DIR, fname)
    frame = cv2.imread(path)
    if frame is None:
        print(f"⚠️  couldn't read {fname} (does it exist in the folder?)")
        continue
    boxes = detect(frame)
    for (x1, y1, x2, y2, c) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{c:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(f'check_umn_frames/{fname}', frame)
    tag = "PANIC" if idx >= 1132 else "calm"
    print(f"saved {fname} ({tag}) with {len(boxes)} boxes")

print("\nDone — open check_umn_frames/ and look. Are boxes on real people?")
