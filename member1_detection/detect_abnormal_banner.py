"""
Detect the UMN dataset's built-in red "Abnormal Crowd Activity" banner
across all frames of a scene, and group flagged frames into candidate
event segments. This gives us real per-frame ground truth, replacing the
single-split convention (Mehran et al.) which we confirmed only captures
ONE of several real events per scene (literature: scene1 has 2, scene2 has
6, scene3 has 3 events).

IMPORTANT: this only tells you WHERE the dataset flags something as
"abnormal" (which, per your own check, includes things like sparse/lone-
person frames, not only panic/running). The next step after running this
is a quick MANUAL glance at each candidate segment to decide which ones
are genuine panic/running events relevant to stampede risk, vs. other
"abnormal" flags (e.g. unusually sparse) that aren't relevant to your goal.

Usage:
    python3 detect_abnormal_banner.py scene1
    python3 detect_abnormal_banner.py scene1 scene2 scene3
"""

import sys
import os
from pathlib import Path

import numpy as np
import cv2
import pandas as pd


# Region to check for the red banner: top-left corner.
# Frame size is 320x240 (confirmed dataset spec). Banner appears to span
# roughly the top ~30px and left ~220px based on the sample image — but
# THIS IS AN ESTIMATE, not measured directly. If detection looks wrong,
# widen this region first before assuming the color thresholds are off.
BANNER_REGION = (0, 30, 0, 220)  # (y_start, y_end, x_start, x_end)

# Red-text color thresholds (BGR order, since OpenCV loads images as BGR).
# Tuned to catch strong red, reject white/green/neutral pixels.
RED_MIN = 150   # minimum red channel value
GB_MAX = 80     # maximum green/blue channel value (keeps it "pure" red, not orange/white)

# Minimum count of matching red pixels in the region to call a frame "flagged".
# Starts as a guess — the printed pixel-count distribution will help tune this.
RED_PIXEL_THRESHOLD = 50

# Gap tolerance: consecutive flagged frames within this many frames of each
# other are merged into the same segment (avoids fragmenting one real event
# into many due to an occasional missed/borderline frame).
GAP_TOLERANCE = 15


def count_red_banner_pixels(frame: np.ndarray) -> int:
    y0, y1, x0, x1 = BANNER_REGION
    region = frame[y0:y1, x0:x1]
    b, g, r = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    mask = (r >= RED_MIN) & (g <= GB_MAX) & (b <= GB_MAX)
    return int(mask.sum())


def scan_scene(scene_folder: str, scene_name: str, out_dir: str = "banner_scan_results"):
    frames = sorted([f for f in os.listdir(scene_folder) if f.endswith(".jpg")])
    if not frames:
        print(f"No .jpg frames found in {scene_folder}")
        return

    red_counts = []
    for fname in frames:
        img = cv2.imread(os.path.join(scene_folder, fname))
        if img is None:
            red_counts.append(0)
            continue
        red_counts.append(count_red_banner_pixels(img))

    red_counts = np.array(red_counts)
    flagged = red_counts >= RED_PIXEL_THRESHOLD

    # Print distribution to help tune RED_PIXEL_THRESHOLD if needed
    print(f"\n{scene_name}: {len(frames)} frames scanned")
    print(f"  Red-pixel-count distribution: "
          f"min={red_counts.min()}, max={red_counts.max()}, "
          f"median={np.median(red_counts):.0f}")
    print(f"  Frames flagged (>= {RED_PIXEL_THRESHOLD} px): {flagged.sum()} "
          f"({flagged.mean()*100:.1f}%)")

    # Group flagged frames into segments, tolerating small gaps
    segments = []
    seg_start = None
    last_flagged_idx = None
    for i, is_flag in enumerate(flagged):
        if is_flag:
            if seg_start is None:
                seg_start = i
            last_flagged_idx = i
        else:
            if seg_start is not None and (i - last_flagged_idx) > GAP_TOLERANCE:
                segments.append((seg_start, last_flagged_idx))
                seg_start = None
    if seg_start is not None:
        segments.append((seg_start, last_flagged_idx))

    print(f"  Candidate event segments ({len(segments)}):")
    for start, end in segments:
        print(f"    frames {start}-{end}  (length {end-start+1})")

    # Save full per-frame data + segment summary for later use
    Path(out_dir).mkdir(exist_ok=True)
    pd.DataFrame({
        "frame_idx": np.arange(len(frames)),
        "red_pixel_count": red_counts,
        "flagged": flagged,
    }).to_csv(Path(out_dir) / f"{scene_name}_banner_scan.csv", index=False)

    pd.DataFrame(segments, columns=["start_frame", "end_frame"]).to_csv(
        Path(out_dir) / f"{scene_name}_candidate_segments.csv", index=False
    )
    print(f"  Saved: {out_dir}/{scene_name}_banner_scan.csv, "
          f"{out_dir}/{scene_name}_candidate_segments.csv")

    return segments


if __name__ == "__main__":
    UMN_PATH = "/Users/harinihegde/Downloads/umn"  # ADJUST if different
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["scene1", "scene2", "scene3"]
    for scene in scenes:
        scan_scene(os.path.join(UMN_PATH, scene), scene)