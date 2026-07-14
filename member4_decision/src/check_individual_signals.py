"""
Checks each Member 4 signal INDIVIDUALLY (crawling, overcrowding, zone
imbalance, high density, large crowd) against real panic labels — before
touching any weights. If a signal barely matches real panic, it's likely
noise, and giving it weight would hurt rather than help.

Usage:
    python3 check_individual_signals.py
"""

import json
from pathlib import Path

import numpy as np

from density_estimator import EnhancedDensityEstimator
from anomaly_detector import EnhancedAnomalyDetector

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


def check_signals_for_scene(scene):
    with open(UMN_DETECTIONS_PATH) as f:
        all_det = json.load(f)
    detections_per_frame = all_det[scene]["detections"]

    density_est = EnhancedDensityEstimator()
    anomaly_det = EnhancedAnomalyDetector()

    rows = []
    for frame_idx, dets in enumerate(detections_per_frame):
        grid = density_est.compute_density_grid(dets)
        zone_densities = density_est.get_zone_densities(grid)
        density_level, _ = density_est.classify_density(len(dets))

        crawling = anomaly_det.detect_crawling(dets)
        overcrowding = anomaly_det.detect_overcrowding(grid, zone_densities)
        imbalance = anomaly_det.detect_zone_imbalance(zone_densities, len(dets))

        rows.append({
            "label": is_panic_frame(scene, frame_idx),
            "crawling": len(crawling) > 0,
            "overcrowding": len(overcrowding) > 0,
            "imbalance": imbalance is not None,
            "density_high": density_level == "HIGH",
            "density_medium_or_high": density_level in ("MEDIUM", "HIGH"),
            "large_crowd": len(dets) > 50,
        })
    return rows


def score_signal(rows, signal_name):
    tp = sum(1 for r in rows if r[signal_name] and r["label"])
    fp = sum(1 for r in rows if r[signal_name] and not r["label"])
    fn = sum(1 for r in rows if not r[signal_name] and r["label"])
    tn = sum(1 for r in rows if not r[signal_name] and not r["label"])

    fire_rate = (tp + fp) / len(rows) if rows else 0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"fire_rate": fire_rate, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


if __name__ == "__main__":
    signals = ["crawling", "overcrowding", "imbalance",
               "density_high", "density_medium_or_high", "large_crowd"]

    all_rows = []
    for scene in SCENES:
        print(f"Scanning {scene}...")
        all_rows.extend(check_signals_for_scene(scene))

    total_panic = sum(1 for r in all_rows if r["label"])
    print(f"\nTotal frames: {len(all_rows)}, real panic frames: {total_panic} "
          f"({total_panic/len(all_rows)*100:.1f}%)\n")

    print(f"{'Signal':<25} {'Fires':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("=" * 60)
    for sig in signals:
        s = score_signal(all_rows, sig)
        print(f"{sig:<25} {s['fire_rate']*100:>6.1f}% {s['precision']:>7.4f} "
              f"{s['recall']:>7.4f} {s['f1']:>7.4f}")