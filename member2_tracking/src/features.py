"""
Single, unified feature-extraction pipeline (Section 3.1 — one code path, batch and real-time).
Computes per-frame per-zone features from tracked detections.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def get_zone_id(cx: float, cy: float, frame_w: int, frame_h: int,
                n_cols: int, n_rows: int) -> int:
    """Map centroid (cx,cy) to a zone index 0..(n_cols*n_rows - 1)."""
    col = min(int(cx / frame_w * n_cols), n_cols - 1)
    row = min(int(cy / frame_h * n_rows), n_rows - 1)
    return row * n_cols + col


def zone_bounds(zone_id: int, frame_w: int, frame_h: int,
                n_cols: int, n_rows: int) -> Tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) pixel bounds of a zone."""
    row, col = divmod(zone_id, n_cols)
    zw = frame_w / n_cols
    zh = frame_h / n_rows
    return col * zw, row * zh, (col + 1) * zw, (row + 1) * zh


# ---------------------------------------------------------------------------
# Per-frame feature helpers
# ---------------------------------------------------------------------------

def _angle_variance(vx_arr: np.ndarray, vy_arr: np.ndarray) -> float:
    """Circular variance of movement angles — flow divergence proxy."""
    if len(vx_arr) == 0:
        return 0.0
    angles = np.arctan2(vy_arr, vx_arr)
    mean_vec = np.mean(np.exp(1j * angles))
    return float(1.0 - abs(mean_vec))


def _nearest_neighbor_dist(centroids: np.ndarray) -> float:
    """Mean nearest-neighbor distance — compression / crowding proxy."""
    n = len(centroids)
    if n < 2:
        return 0.0
    diff = centroids[:, None, :] - centroids[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(np.min(dists, axis=1)))


def _spatial_entropy(centroids: np.ndarray, zone_area: float,
                     n_bins: int = 4) -> float:
    """Shannon entropy of spatial distribution within a zone."""
    if len(centroids) == 0:
        return 0.0
    hist, _ = np.histogramdd(np.stack([centroids[:, 0], centroids[:, 1]], axis=1), bins=n_bins)
    flat = hist.flatten()
    flat = flat[flat > 0]
    if flat.sum() == 0:
        return 0.0
    p = flat / flat.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


# ---------------------------------------------------------------------------
# Main feature pipeline
# ---------------------------------------------------------------------------

def extract_features(
    frame_tracks: List[List[Dict]],
    fps: float,
    frame_w: int = 360,
    frame_h: int = 240,
    n_cols: int = 3,
    n_rows: int = 2,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """
    Compute per-frame per-zone features from tracker output.

    frame_tracks: output of run_tracker() — list[frame_idx] -> list of track dicts.
    fps: dataset native fps (10.0 for UCSD Ped2).

    Returns a DataFrame with columns:
      frame_idx, zone_id,
      count, density, delta_count,
      speed_mean, speed_std,
      flow_x, flow_y, flow_magnitude,
      accel_mean,
      flow_divergence,
      compression,
      spatial_entropy,
      zone_inflow, zone_outflow,
      + smoothed (_sm) variants of each numeric feature.
    """
    n_zones = n_cols * n_rows
    zone_area_px = (frame_w / n_cols) * (frame_h / n_rows)

    # Per-track speed history for true per-track acceleration
    prev_track_speed: Dict[int, float] = {}  # track_id -> speed px/s in previous frame

    # Per-zone track ID sets for entry/exit tracking
    prev_zone_ids: Dict[int, set] = {zid: set() for zid in range(n_zones)}

    rows = []

    for fi, frame in enumerate(frame_tracks):
        # Group tracks by zone, annotate with centroid
        zone_tracks: Dict[int, List[Dict]] = defaultdict(list)
        for t in frame:
            cx = (t["x1"] + t["x2"]) / 2
            cy = (t["y1"] + t["y2"]) / 2
            zid = get_zone_id(cx, cy, frame_w, frame_h, n_cols, n_rows)
            zone_tracks[zid].append({**t, "cx": cx, "cy": cy})

        # Kalman-smoothed velocity → px/s per track
        velocities: Dict[int, np.ndarray] = {}
        curr_track_speed: Dict[int, float] = {}
        for t in frame:
            tid = t["track_id"]
            v = np.array([t["vx"] * fps, t["vy"] * fps])
            velocities[tid] = v
            curr_track_speed[tid] = float(np.linalg.norm(v))

        # Current zone membership for entry/exit
        curr_zone_ids: Dict[int, set] = {
            zid: {t["track_id"] for t in zone_tracks.get(zid, [])}
            for zid in range(n_zones)
        }

        # Per-zone features
        for zid in range(n_zones):
            zone_t = zone_tracks.get(zid, [])
            count = len(zone_t)
            density = count / zone_area_px

            # Zone entry/exit: track IDs that moved into / out of this zone
            zone_inflow  = len(curr_zone_ids[zid] - prev_zone_ids[zid])
            zone_outflow = len(prev_zone_ids[zid] - curr_zone_ids[zid])

            if count == 0:
                rows.append({
                    "frame_idx": fi, "zone_id": zid,
                    "count": 0, "density": 0.0, "delta_count": 0,
                    "speed_mean": 0.0, "speed_std": 0.0,
                    "flow_x": 0.0, "flow_y": 0.0, "flow_magnitude": 0.0,
                    "accel_mean": 0.0,
                    "flow_divergence": 0.0,
                    "compression": 0.0,
                    "spatial_entropy": 0.0,
                    "zone_inflow": zone_inflow,
                    "zone_outflow": zone_outflow,
                })
                continue

            centroids = np.array([[t["cx"], t["cy"]] for t in zone_t])
            vels = np.array([velocities[t["track_id"]] for t in zone_t])
            speeds = np.linalg.norm(vels, axis=1)

            speed_mean = float(np.mean(speeds))
            speed_std  = float(np.std(speeds))
            flow_x     = float(np.mean(vels[:, 0]))
            flow_y     = float(np.mean(vels[:, 1]))
            flow_magnitude = float(np.sqrt(flow_x ** 2 + flow_y ** 2))
            flow_divergence = _angle_variance(vels[:, 0], vels[:, 1])
            compression     = _nearest_neighbor_dist(centroids)
            spatial_entropy = _spatial_entropy(centroids, zone_area_px)

            # Per-track acceleration: diff of each track's Kalman speed vs previous frame,
            # then zone mean. Avoids cancellation from zone-level averaging first.
            per_track_accels = []
            for t in zone_t:
                tid = t["track_id"]
                if tid in prev_track_speed:
                    per_track_accels.append(
                        (curr_track_speed[tid] - prev_track_speed[tid]) * fps
                    )
            accel_mean = float(np.mean(per_track_accels)) if per_track_accels else 0.0

            rows.append({
                "frame_idx": fi, "zone_id": zid,
                "count": count, "density": density, "delta_count": 0,  # filled below
                "speed_mean": speed_mean, "speed_std": speed_std,
                "flow_x": flow_x, "flow_y": flow_y, "flow_magnitude": flow_magnitude,
                "accel_mean": accel_mean,
                "flow_divergence": flow_divergence,
                "compression": compression,
                "spatial_entropy": spatial_entropy,
                "zone_inflow": zone_inflow,
                "zone_outflow": zone_outflow,
            })

        prev_track_speed = curr_track_speed
        prev_zone_ids = curr_zone_ids

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # delta_count: frame-to-frame change in headcount per zone
    df = df.sort_values(["zone_id", "frame_idx"]).reset_index(drop=True)
    df["delta_count"] = (
        df.groupby("zone_id")["count"]
        .transform(lambda s: s.diff().fillna(0.0))
    )

    df = df.sort_values(["frame_idx", "zone_id"]).reset_index(drop=True)

    # Smoothed variants (rolling mean per zone)
    raw_cols = [
        "count", "density", "delta_count",
        "speed_mean", "speed_std",
        "flow_x", "flow_y", "flow_magnitude",
        "accel_mean",
        "flow_divergence",
        "compression",
        "spatial_entropy",
        "zone_inflow", "zone_outflow",
    ]
    df = df.sort_values(["zone_id", "frame_idx"]).reset_index(drop=True)
    for col in raw_cols:
        df[f"{col}_sm"] = (
            df.groupby("zone_id")[col]
            .transform(lambda s: s.rolling(smoothing_window, min_periods=1).mean())
        )

    df = df.sort_values(["frame_idx", "zone_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature schema (for Member 3 reference)
# ---------------------------------------------------------------------------

RAW_FEATURE_COLS = [
    "count", "density", "delta_count",
    "speed_mean", "speed_std",
    "flow_x", "flow_y", "flow_magnitude",
    "accel_mean",
    "flow_divergence",
    "compression",
    "spatial_entropy",
    "zone_inflow", "zone_outflow",
]

SMOOTHED_FEATURE_COLS = [f"{c}_sm" for c in RAW_FEATURE_COLS]

ALL_FEATURE_COLS = RAW_FEATURE_COLS + SMOOTHED_FEATURE_COLS