"""
Diagnostic: for a given UMN scene, count how many NEW track IDs appear in
each frame (tracker "losing" a person and starting a fresh ID for them).
A burst of new IDs at the same frames as an unexplained MSE spike would
suggest the spike is a tracking glitch, not real crowd behavior.

Does NOT touch Member 3's model at all — only re-runs the tracker (fast,
no training) on the raw detections, using the same config as the real
pipeline run, so results are directly comparable.

Usage:
    python3 check_track_fragmentation.py scene1
    python3 check_track_fragmentation.py scene1 scene2 scene3
"""

import sys
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from tracker import run_tracker  # same tracker.py the real pipeline uses


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def per_frame_new_id_counts(frame_tracks):
    """
    frame_tracks: output of run_tracker() — list[frame_idx] -> list of track dicts.
    Returns (new_id_counts, active_counts) — both length == len(frame_tracks).
    """
    new_id_counts = []
    active_counts = []
    prev_ids = set()

    for frame in frame_tracks:
        curr_ids = {t["track_id"] for t in frame}
        new_id_counts.append(len(curr_ids - prev_ids))
        active_counts.append(len(curr_ids))
        prev_ids = curr_ids

    return np.array(new_id_counts), np.array(active_counts)


def check_scene(detection_path: str, scene: str, tracker_cfg: dict,
                windows_of_interest=None, out_dir="diagnostic_plots"):
    with open(detection_path) as f:
        data = json.load(f)

    detections_per_frame = data[scene]["detections"]
    frame_tracks, track_lengths = run_tracker(
        detections_per_frame,
        max_age=tracker_cfg["max_age"],
        min_iou=tracker_cfg["min_iou"],
        min_track_len=tracker_cfg["min_track_len"],
        max_dist=tracker_cfg["max_dist"],
    )

    new_ids, active = per_frame_new_id_counts(frame_tracks)
    frames = np.arange(len(new_ids))

    print(f"\n{scene}: {len(frames)} frames total")
    print(f"  Overall mean new-IDs/frame: {new_ids.mean():.3f}")
    print(f"  Overall mean active tracks/frame: {active.mean():.2f}")

    if windows_of_interest:
        for label, (start, end) in windows_of_interest.items():
            mask = (frames >= start) & (frames <= end)
            print(f"  Window '{label}' [{start}-{end}]: "
                  f"mean new-IDs/frame = {new_ids[mask].mean():.3f} "
                  f"(vs overall {new_ids.mean():.3f}), "
                  f"mean active = {active[mask].mean():.2f}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frames, new_ids, color="darkorange", linewidth=1, label="New track IDs / frame")
    if windows_of_interest:
        for label, (start, end) in windows_of_interest.items():
            ax.axvspan(start, end, color="purple", alpha=0.15, label=f"Spike window: {label}")
    ax.set_xlabel("Frame index (local to scene)")
    ax.set_ylabel("New track IDs this frame")
    ax.set_title(f"UMN {scene} — track fragmentation over time")
    ax.legend(loc="upper right")
    fig.tight_layout()

    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"{scene}_fragmentation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    # Flagged unexplained MSE spike windows, read directly off the plots we
    # just looked at (approximate frame ranges).
    SPIKE_WINDOWS = {
        "scene1": {"unexplained_spike": (500, 620)},
        "scene2": {"unexplained_spike": (3150, 3300)},
        "scene3": {"unexplained_spike": (550, 650)},
    }

    detection_path = "../../member1_detection/detection_data/umn_detections.json"  # ADJUST if needed
    config_path = "../../member2_tracking/configs/config_umn.yaml"  # ADJUST if needed
    cfg = load_config(config_path)
    tracker_cfg = cfg["tracker"]

    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["scene1", "scene2", "scene3"]
    for s in scenes:
        check_scene(detection_path, s, tracker_cfg, SPIKE_WINDOWS.get(s, {}))