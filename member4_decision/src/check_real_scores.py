"""
Looks at the REAL risk scores already saved (from run_member4_on_umn.py),
split by real panic vs normal, to find an achievable Critical cutoff —
instead of guessing one.

Usage:
    python3 check_real_risk_scores.py
"""

import json
import numpy as np

UMN_GT = {
    'scene1': [(525, 614), (1330, 1439)],
    'scene2': [(353, 532), (1152, 1231), (1766, 1975),
               (2485, 2564), (3354, 3475), (3969, 4142)],
    'scene3': [(598, 637), (1286, 1315), (2103, 2122)],
}
SCENES = ['scene1', 'scene2', 'scene3']


def is_panic_frame(scene, frame_idx):
    return any(start <= frame_idx <= end for start, end in UMN_GT[scene])


def describe(name, values):
    values = np.array(values)
    if len(values) == 0:
        print(f"  {name}: (no frames)")
        return
    print(f"  {name}: n={len(values)}, mean={values.mean():.3f}, "
          f"median={np.median(values):.3f}, "
          f"p75={np.percentile(values, 75):.3f}, "
          f"p90={np.percentile(values, 90):.3f}, "
          f"max={values.max():.3f}")


if __name__ == "__main__":
    panic_scores, normal_scores = [], []

    for scene in SCENES:
        with open(f"member4_results/{scene}_results.json") as f:
            data = json.load(f)
        for frame in data["frames"]:
            score = frame["risk_score"]
            if is_panic_frame(scene, frame["frame_num"]):
                panic_scores.append(score)
            else:
                normal_scores.append(score)

    print("=== RISK SCORE DISTRIBUTION ===")
    describe("Normal frames", normal_scores)
    describe("Panic frames ", panic_scores)

    print("\n=== TESTING CANDIDATE 'HIGH' CUTOFFS ===")
    print("(this is the main trigger — since more false alarms is preferred")
    print(" over missing real danger, favor higher RECALL here)")
    print(f"{'Cutoff':<10} {'F1':>7} {'Precision':>10} {'Recall':>8} "
          f"{'Panic caught':>13} {'Normal wrongly flagged':>23}")
    all_scores_labels = [(s, 1) for s in panic_scores] + [(s, 0) for s in normal_scores]
    for cutoff in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        tp = sum(1 for s, l in all_scores_labels if s >= cutoff and l == 1)
        fp = sum(1 for s, l in all_scores_labels if s >= cutoff and l == 0)
        fn = sum(1 for s, l in all_scores_labels if s < cutoff and l == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{cutoff:<10} {f1:>7.4f} {precision:>10.4f} {recall:>8.4f} "
              f"{tp:>13} {fp:>23}")

    print("\n=== TESTING CANDIDATE 'CRITICAL' CUTOFFS ===")
    print(f"{'Cutoff':<10} {'F1':>7} {'Precision':>10} {'Recall':>8} "
          f"{'Panic caught':>13} {'Normal wrongly flagged':>23}")
    all_scores_labels = [(s, 1) for s in panic_scores] + [(s, 0) for s in normal_scores]
    for cutoff in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        tp = sum(1 for s, l in all_scores_labels if s >= cutoff and l == 1)
        fp = sum(1 for s, l in all_scores_labels if s >= cutoff and l == 0)
        fn = sum(1 for s, l in all_scores_labels if s < cutoff and l == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{cutoff:<10} {f1:>7.4f} {precision:>10.4f} {recall:>8.4f} "
              f"{tp:>13} {fp:>23}")