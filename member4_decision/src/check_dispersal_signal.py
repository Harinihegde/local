"""
Tests the DispersalDetector (sudden crowd-count drop) against real panic
labels, across all 3 UMN scenes, trying a few z_threshold candidates.

Usage:
    python3 check_dispersal_signal.py
"""

import json
import numpy as np

from member4_decision.src.dispersal_detector import DispersalDetector

UMN_GT = {
    'scene1': [(525, 614), (1330, 1439)],
    'scene2': [(353, 532), (1152, 1231), (1766, 1975),
               (2485, 2564), (3354, 3475), (3969, 4142)],
    'scene3': [(598, 637), (1286, 1315), (2103, 2122)],
}
SCENES = ['scene1', 'scene2', 'scene3']

UMN_DETECTIONS_PATH = "/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/detection_data/umn_detections.json"  # ADJUST if needed


def is_panic_frame(scene, frame_idx):
    return any(start <= frame_idx <= end for start, end in UMN_GT[scene])


def run_scene(scene, z_threshold, window=200):
    with open(UMN_DETECTIONS_PATH) as f:
        all_det = json.load(f)
    dets_per_frame = all_det[scene]["detections"]

    detector = DispersalDetector(window=window, z_threshold=z_threshold)
    results = []
    for frame_idx, dets in enumerate(dets_per_frame):
        is_flagged, z = detector.update(len(dets))
        results.append((is_flagged, is_panic_frame(scene, frame_idx)))
    return results


def score(all_results):
    tp = sum(1 for f, l in all_results if f and l)
    fp = sum(1 for f, l in all_results if f and not l)
    fn = sum(1 for f, l in all_results if not f and l)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


if __name__ == "__main__":
    print(f"{'z_threshold':<12} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("=" * 45)
    for z_thr in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        all_results = []
        for scene in SCENES:
            all_results.extend(run_scene(scene, z_threshold=z_thr))
        f1, prec, rec = score(all_results)
        print(f"{z_thr:<12} {f1:>7.4f} {prec:>10.4f} {rec:>8.4f}")