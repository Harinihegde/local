"""
compare_umn_thresholds.py — render the SAME UMN panic frame at several thresholds
so you can compare box QUALITY (are boxes on real people?), not just count.
Saves one image per threshold to compare_umn/. Standalone.
"""
import cv2
import os
import torch
from ultralytics import YOLO

# ---- settings ----
MODEL_PATH = '/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/models/best_combined.pt'
SCENE_DIR  = '/Users/harinihegde/Downloads/umn/scene1'   # adjust if needed
FRAME_IDX  = 1200                      # a panic frame in scene1
CONF_VALUES = [0.1, 0.15, 0.25, 0.35, 0.45]

# ---- detector setup ----
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device: {device}")
model = YOLO(MODEL_PATH)

def preprocess(frame):
    # note: no CLAHE state here — we render a single frame per threshold, so keep it simple/consistent
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
    return filt, scale, top, left

def detect(frame, conf):
    proc, scale, pad_top, pad_left = preprocess(frame)
    res = model.predict(source=proc, classes=[0], conf=conf, verbose=False, device=device)
    boxes = []
    if len(res) > 0 and res[0].boxes is not None:
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0].cpu().numpy())
            x1 = int((x1 - pad_left) / scale); y1 = int((y1 - pad_top) / scale)
            x2 = int((x2 - pad_left) / scale); y2 = int((y2 - pad_top) / scale)
            boxes.append((x1, y1, x2, y2, c))
    return boxes

# ---- load the one frame, render at each threshold ----
os.makedirs('compare_umn', exist_ok=True)
fname = f"frame_{FRAME_IDX:05d}.jpg"
src = cv2.imread(os.path.join(SCENE_DIR, fname))
if src is None:
    print(f"⚠️  couldn't read {fname} — check SCENE_DIR / FRAME_IDX")
else:
    print(f"\nRendering {fname} at each threshold:\n")
    for conf in CONF_VALUES:
        frame = src.copy()
        boxes = detect(frame, conf)
        for (x1, y1, x2, y2, c) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{c:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        out = f"compare_umn/conf_{conf:.2f}.jpg"
        cv2.imwrite(out, frame)
        print(f"  conf={conf:.2f}: {len(boxes)} boxes  ->  {out}")

    print("\nOpen compare_umn/ and compare. Pick the one where boxes are")
    print("cleanest ON PEOPLE — NOT the one with the most boxes.")
