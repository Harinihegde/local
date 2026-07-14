"""
Visualization: renders tracker output + zone features onto video frames and exports MP4.
Run: venv/bin/python -m src.visualize --vid Test001
"""

import json
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

from src.tracker import run_tracker
from src.features import extract_features, get_zone_id
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 360, 240
N_COLS, N_ROWS   = 3, 2
ZONE_W = FRAME_W // N_COLS   # 120
ZONE_H = FRAME_H // N_ROWS   # 120
FPS    = 10.0
SCALE  = 3   # upscale factor for readability (360x240 → 1080x720)

# Colours (BGR)
C_BOX       = (0, 80, 255)    # orange-red boxes
C_TRACK_ID  = (0, 255, 255)   # yellow track IDs
C_ZONE_LINE = (200, 200, 200) # light grey zone grid
C_NORMAL    = (255, 255, 255) # white text — normal zone
C_HEADER    = (180, 255, 100) # green header text
C_ANOMALY   = (0, 100, 255)   # orange — high density zone highlight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def scale_pt(x, y):
    return int(x * SCALE), int(y * SCALE)


def draw_zone_grid(canvas):
    for col in range(1, N_COLS):
        x = col * ZONE_W * SCALE
        cv2.line(canvas, (x, 0), (x, FRAME_H * SCALE), C_ZONE_LINE, 1)
    for row in range(1, N_ROWS):
        y = row * ZONE_H * SCALE
        cv2.line(canvas, (0, y), (FRAME_W * SCALE, y), C_ZONE_LINE, 1)
    # Zone ID labels (top-left corner of each zone)
    for zid in range(N_COLS * N_ROWS):
        row, col = divmod(zid, N_COLS)
        x = col * ZONE_W * SCALE + 4
        y = row * ZONE_H * SCALE + 16
        cv2.putText(canvas, f"Z{zid}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ZONE_LINE, 1, cv2.LINE_AA)


def draw_tracks(canvas, frame_tracks_fi, fi):
    for t in frame_tracks_fi:
        x1, y1 = scale_pt(t["x1"], t["y1"])
        x2, y2 = scale_pt(t["x2"], t["y2"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_BOX, 1)
        cv2.putText(canvas, f"{t['track_id']}", (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TRACK_ID, 1, cv2.LINE_AA)


def density_color(density_raw):
    """Tint zone background based on density. Returns BGR overlay colour."""
    # density in persons/px², max reasonable ~0.0006
    intensity = min(density_raw / 0.0005, 1.0)
    b = int(255 * (1 - intensity))
    r = int(200 * intensity)
    return (b, 30, r)


def draw_zone_features(canvas, zone_row, gt_anomaly: bool):
    """Overlay per-zone feature text on right-side panel."""
    panel_x = FRAME_W * SCALE + 10
    line_h   = 14
    for _, row in zone_row.iterrows():
        zid  = int(row["zone_id"])
        zrow, zcol = divmod(zid, N_COLS)
        base_y = zrow * ZONE_H * SCALE + 20

        # Subtle zone tint on canvas
        x1 = zcol * ZONE_W * SCALE
        y1 = zrow * ZONE_H * SCALE
        x2 = x1 + ZONE_W * SCALE
        y2 = y1 + ZONE_H * SCALE
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      density_color(row.get("density", 0)), -1)
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

        # Right-panel text block per zone
        lines = [
            f"Zone {zid}",
            f"  count : {row.get('count', 0):.0f}  Δ{row.get('delta_count', 0):+.0f}",
            f"  speed : {row.get('speed_mean', 0):.1f} px/s",
            f"  flow  : mag={row.get('flow_magnitude', 0):.1f}  div={row.get('flow_divergence', 0):.2f}",
            f"  accel : {row.get('accel_mean', 0):+.1f} px/s²",
            f"  in/out: +{row.get('zone_inflow', 0):.0f} / -{row.get('zone_outflow', 0):.0f}",
            f"  compr : {row.get('compression', 0):.1f} px",
        ]
        col_color = C_ANOMALY if gt_anomaly else C_NORMAL
        for i, line in enumerate(lines):
            y = base_y + i * line_h
            color = C_HEADER if i == 0 else col_color
            cv2.putText(canvas, line, (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def draw_header(canvas, vid_id, fi, n_frames, n_tracks, gt_anomaly):
    label = "ANOMALY" if gt_anomaly else "NORMAL"
    color = (0, 60, 255) if gt_anomaly else (80, 200, 80)
    text  = f"{vid_id}  frame {fi+1:03d}/{n_frames}  tracks={n_tracks}  [{label}]"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 20), (30, 30, 30), -1)
    cv2.putText(canvas, text, (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_video(vid_id: str, ucsd_root: str, out_path: str):
    cfg = load_config()
    tracker_cfg = cfg["tracker"]

    # Load detections
    split = "train" if "Train" in vid_id else "test"
    det_path = f"detection_data/{'train' if split == 'train' else 'test'}_detections_combined.json"
    with open(det_path) as f:
        data = json.load(f)

    if vid_id not in data:
        print(f"ERROR: {vid_id} not found in {det_path}")
        return

    vid_data = data[vid_id]
    detections_per_frame = vid_data["detections"]
    n_frames = len(detections_per_frame)

    # Load ground truth anomaly frame ranges (test only)
    gt_frames = set()
    if split == "test":
        m_path = os.path.join(ucsd_root, "Test", "UCSDped2.m")
        if os.path.exists(m_path):
            test_ids = sorted([k for k in data.keys()])
            vid_idx = test_ids.index(vid_id)
            with open(m_path) as f:
                lines = [l.strip() for l in f if "gt_frame" in l]
            if vid_idx < len(lines):
                import re
                m = re.search(r'\[(\d+):(\d+)\]', lines[vid_idx])
                if m:
                    start, end = int(m.group(1)), int(m.group(2))
                    gt_frames = set(range(start - 1, end))  # 0-indexed

    # Run tracker
    print(f"Running tracker on {vid_id} ({n_frames} frames)...")
    frame_tracks, track_lengths = run_tracker(
        detections_per_frame,
        max_age=tracker_cfg["max_age"],
        min_iou=tracker_cfg["min_iou"],
        min_track_len=tracker_cfg["min_track_len"],
        max_dist=tracker_cfg["max_dist"],
    )

    # Extract features
    print("Extracting features...")
    import pandas as pd
    df = extract_features(frame_tracks, fps=FPS,
                          frame_w=FRAME_W, frame_h=FRAME_H,
                          n_cols=N_COLS, n_rows=N_ROWS)

    # Load .tif frames
    frame_dir = os.path.join(ucsd_root, "Test" if split == "test" else "Train", vid_id)
    tif_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".tif")])

    # Video writer — canvas width = frame*SCALE + right panel
    panel_w  = 200
    canvas_w = FRAME_W * SCALE + panel_w
    canvas_h = FRAME_H * SCALE
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer   = cv2.VideoWriter(out_path, fourcc, FPS, (canvas_w, canvas_h))

    print(f"Rendering {n_frames} frames → {out_path}")
    for fi in range(n_frames):
        # Load and upscale frame
        tif_path = os.path.join(frame_dir, tif_files[fi])
        gray = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (FRAME_W * SCALE, FRAME_H * SCALE), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Right panel background
        panel = np.zeros((canvas_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)
        canvas = np.hstack([canvas, panel])

        # Zone features for this frame
        zone_row = df[df["frame_idx"] == fi] if not df.empty else pd.DataFrame()

        gt_anomaly = fi in gt_frames

        # Draw layers
        draw_zone_grid(canvas)
        if not zone_row.empty:
            draw_zone_features(canvas, zone_row, gt_anomaly)
        draw_tracks(canvas, frame_tracks[fi], fi)
        draw_header(canvas, vid_id, fi, n_frames,
                    len(frame_tracks[fi]), gt_anomaly)

        writer.write(canvas)

    writer.release()
    print(f"Done. Saved to {out_path}")
    print(f"  Tracks: {len(track_lengths)} raw, "
          f"{sum(1 for l in track_lengths.values() if l >= tracker_cfg['min_track_len'])} kept")
    if gt_frames:
        print(f"  Ground truth anomaly frames: {min(gt_frames)+1}–{max(gt_frames)+1} (1-indexed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", default="Test001",
                        help="Video ID e.g. Test001, Test005, Test011")
    parser.add_argument("--ucsd", default=
        "/Users/jeevithasmacbook/Downloads/UCSD_Anomaly_Dataset.v1p2/UCSDped2",
        help="Path to UCSDped2 root")
    parser.add_argument("--out", default=None,
                        help="Output MP4 path (default: outputs/viz/<vid>.mp4)")
    args = parser.parse_args()

    out = args.out or f"outputs/viz/{args.vid}.mp4"
    os.makedirs("outputs/viz", exist_ok=True)
    render_video(args.vid, args.ucsd, out)