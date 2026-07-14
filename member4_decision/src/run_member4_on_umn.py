"""
Runs Member 4's pipeline on REAL UMN data — real detections (Member 1),
real per-frame anomaly scores (Member 3, from its saved diagnostics).

For a given scene, we use the fold where THAT scene was the held-out test
scene — meaning Member 3's model genuinely never saw this scene during
training. This matches a real deployment scenario (testing on unseen video).

DESIGN NOTE: unlike the original mock/demo code (which only included
frames Member 3 specifically "flagged"), this passes EVERY frame's
translated score to Member 4 — not just ones that crossed the cutoff.
This preserves the full picture (a frame at 0.3 "somewhat unusual" looks
different from 0.05 "very normal"), rather than collapsing everything
down to flagged-or-not before Member 4 even sees it.

Usage:
    python3 run_member4_on_umn.py scene1
"""

import sys
import json
from pathlib import Path

import numpy as np

from pipeline import Member4Pipeline
from member3_to4 import zscore_to_probability

MEMBER3_Z_CUTOFF = 5.5  # same placeholder cutoff used in pipeline.py


def build_member3_anomalies(loso_root: str, scene: str) -> dict:
    """
    Load Member 3's saved diagnostics for the fold where `scene` was the
    held-out test scene, and build the {"anomalies": [...]} structure
    Member4Pipeline expects — one entry per frame, with the RAW z-score
    (pipeline.py's own translator will convert it, so we pass raw here).
    """
    diag_path = Path(loso_root) / f"test_{scene}" / "diagnostics.npz"
    data = np.load(diag_path)

    robust_z = data["robust_z"]       # already causal/rolling, per test sequence
    last_frames = data["last_frames"] # which frame each score belongs to

    anomalies = [
        {"frame": int(fi), "anomaly_score": float(z)}
        for fi, z in zip(last_frames, robust_z)
    ]
    return {scene: {"anomalies": anomalies}}


def build_member1_detections(umn_detections_path: str, scene: str) -> dict:
    """Load real detections for one scene from the real detection JSON."""
    with open(umn_detections_path) as f:
        all_detections = json.load(f)
    scene_data = all_detections[scene]
    return {scene: {"detections": scene_data["detections"], "fps": scene_data.get("fps", 30.0)}}


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene1"

    # ADJUST these two paths to match where the real files actually live
    UMN_DETECTIONS_PATH = "/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/detection_data/umn_detections.json"
    LOSO_ROOT = "/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/member2_tracking/outputs/features_umn/loso_folds"

    print(f"Loading real Member 1 detections for {scene}...")
    member1_data = build_member1_detections(UMN_DETECTIONS_PATH, scene)

    print(f"Loading real Member 3 anomaly scores for {scene} "
          f"(from the fold where {scene} was held out)...")
    member3_data = build_member3_anomalies(LOSO_ROOT, scene)

    pipeline = Member4Pipeline(member1_data, member2_features_folder=None,
                                member3_anomalies=member3_data)
    results = pipeline.process_video(scene)

    summary = pipeline.generate_summary({scene: results})

    print(f"\n{'='*60}")
    print(f"REAL RESULTS — {scene}")
    print(f"{'='*60}")
    print(f"Total frames: {results['total_frames']}")
    print(f"High-risk frames: {results['high_risk_frames']}")
    print(f"Critical-risk frames: {results['critical_risk_frames']}")
    print(f"Total alerts: {results['total_alerts']}")

    out_dir = Path("member4_results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{scene}_results.json", "w") as f:
        json.dump({"video_id": scene, "frames": results["frames"]}, f, indent=2)
    print(f"\nSaved frame-by-frame results to member4_results/{scene}_results.json")