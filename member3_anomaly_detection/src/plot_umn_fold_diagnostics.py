"""
Plot reconstruction MSE over time vs. the true panic segments, for any UMN
LOSO fold. Reads the diagnostics.npz saved by train_evaluate_umn_loso.py —
does NOT retrain anything, so this is fast and can be re-run freely.

Updated for multi-event ground truth: each scene can have several separate
panic segments (scene1=2, scene2=6, scene3=3), not just one.

Usage:
    python3 plot_umn_fold_diagnostics.py scene2
    python3 plot_umn_fold_diagnostics.py scene1 scene2 scene3   # all three
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from train_evaluate_umn_loso import UMN_GT, is_panic_frame  # no TF import triggered


def plot_fold(loso_root: str, scene: str, out_dir: str = "diagnostic_plots"):
    diag_path = Path(loso_root) / f"test_{scene}" / "diagnostics.npz"
    data = np.load(diag_path)

    test_mse = data["test_mse"]
    threshold = float(data["threshold"])
    last_frames = data["last_frames"]

    panic_segments = UMN_GT[scene]  # list of (start, end) tuples

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(last_frames, test_mse, label="Reconstruction MSE", color="steelblue", linewidth=1)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"Threshold (95th pct of train MSE) = {threshold:.4f}")

    for i, (start, end) in enumerate(panic_segments):
        label = f"True panic segments ({len(panic_segments)} total)" if i == 0 else None
        ax.axvspan(start, end, color="red", alpha=0.15, label=label)

    ax.set_xlabel("Frame index (local to scene)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"UMN {scene} — held-out fold: reconstruction error over time "
                 f"({len(panic_segments)} panic segments)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"{scene}_diagnostic.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Quick numeric summary alongside the plot
    panic_mask = np.array([is_panic_frame(scene, int(fi)) for fi in last_frames])
    if panic_mask.any():
        print(f"  Mean MSE during panic:  {test_mse[panic_mask].mean():.4f}")
    else:
        print("  (no panic frames in test sequences?)")
    print(f"  Mean MSE during normal: {test_mse[~panic_mask].mean():.4f}")

    # Per-segment breakdown — useful now that there are multiple events
    for i, (start, end) in enumerate(panic_segments):
        seg_mask = (last_frames >= start) & (last_frames <= end)
        if seg_mask.any():
            print(f"  Segment {i+1} [{start}-{end}]: mean MSE = {test_mse[seg_mask].mean():.4f} "
                  f"({seg_mask.sum()} test sequences)")


if __name__ == "__main__":
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["scene1", "scene2", "scene3"]
    for s in scenes:
        plot_fold(loso_root="/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/member2_tracking/outputs/features_umn/loso_folds", scene=s)