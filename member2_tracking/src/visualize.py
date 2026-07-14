"""
visualize.py
============
Diagnostic visualizations for Member 2's outputs.

Figures produced
----------------
Fig 1  plot_zone_layout        — zone boundaries on camera frame
Fig 2  plot_density_timeseries — per-zone density + speed vs time
Fig 3  plot_centroid_heatmap   — 2-D centroid density map per video
Fig 4  plot_dataset_summary    — mean/peak count for all videos
Fig 5  plot_speed_distribution — motion speed histogram

All functions:
  • Accept optional save_path; save PNG at 150 dpi when given.
  • Call plt.close(fig) before returning to prevent memory accumulation.
  • Return the Figure so notebooks can still display inline.
"""

from __future__ import annotations
from matplotlib.figure import Figure

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless backend — safe for scripts and notebooks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi":        100,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})

ZONE_COLORS = {
    "left":   "#378ADD",
    "center": "#1D9E75",
    "right":  "#D85A30",
}


# ─────────────────────────────────────────────────────────────
# Fig 1 — Zone layout
# ─────────────────────────────────────────────────────────────

def plot_zone_layout(
    frame_width:  int,
    frame_height: int,
    zones:        Dict[str, List[int]],
    save_path:    Optional[str] = None,
) -> Figure:
    """Draw zone boundaries on a camera-frame canvas."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)   # image convention: y=0 at top
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    for name, coords in zones.items():
        x1, y1, x2, y2 = coords
        color = ZONE_COLORS.get(name, "#888")
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.20,
        ))
        ax.axvline(x=x1, color=color, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, name.upper(),
                ha="center", va="center",
                fontsize=16, fontweight="bold", color=color)

    ax.legend(
        handles=[mpatches.Patch(color=c, label=n, alpha=0.7)
                 for n, c in ZONE_COLORS.items()],
        loc="lower right",
    )
    ax.set_title("Zone Layout — Camera Frame View", fontweight="bold")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Fig 2 — Density time-series
# ─────────────────────────────────────────────────────────────

def plot_density_timeseries(
    df:         pd.DataFrame,
    vid_id:     str,
    fps:        float,
    use_smooth: bool = True,
    save_path:  Optional[str] = None,
) -> Figure:
    """Per-zone occupancy and total count vs time (seconds)."""
    sfx  = "_smooth" if use_smooth else ""
    time = df["frame"] / fps

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"hspace": 0.08}
    )

    # Top panel: per-zone counts
    for zone, color in ZONE_COLORS.items():
        col = f"{zone}{sfx}"
        if col in df.columns:
            ax1.plot(time, df[col], label=zone, color=color,
                     linewidth=1.5, alpha=0.9)
    ax1.set_ylabel("Person count per zone")
    ax1.legend(frameon=False)
    ax1.set_title(
        f"Video {vid_id}  —  Crowd density over time"
        + ("  (smoothed)" if use_smooth else ""),
        fontweight="bold",
    )

    # Bottom panel: total count + mean speed (twin y-axis)
    ax2.plot(time, df[f"total_count{sfx}"], color="#333",
             linewidth=1.8, label="total count")
    ax2.set_ylabel("Total person count")
    ax2.set_xlabel("Time (s)")

    ax3 = ax2.twinx()
    ax3.plot(time, df[f"mean_speed{sfx}"], color="#E24B4A",
             linewidth=1.2, linestyle="--", alpha=0.75, label="mean speed")
    ax3.set_ylabel("Mean speed (px / frame)", color="#E24B4A")
    ax3.tick_params(axis="y", labelcolor="#E24B4A")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Fig 3 — Centroid density map
# ─────────────────────────────────────────────────────────────

def plot_centroid_heatmap(
    tracked_frames: List[List[Tuple[int, float, float]]],
    frame_width:    int,
    frame_height:   int,
    vid_id:         str,
    bins:           int = 64,
    save_path:      Optional[str] = None,
) -> Figure:
    """2-D histogram of all tracked centroids across the video."""
    xs = [cx for frame in tracked_frames for _, cx, _  in frame]
    ys = [cy for frame in tracked_frames for _, _,  cy in frame]

    fig, ax = plt.subplots(figsize=(9, 6))
    h = ax.hist2d(
        xs, ys,
        bins=bins,
        range=[[0, frame_width], [0, frame_height]],
        cmap="hot_r",
        density=True,
    )
    plt.colorbar(h[3], ax=ax, label="Relative density")
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)   # image convention
    ax.set_title(f"Video {vid_id}  —  Centroid Density Map", fontweight="bold")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Fig 4 — Dataset statistics bar chart
# ─────────────────────────────────────────────────────────────

def plot_dataset_summary(
    summary_rows: List[Dict],
    save_path:    Optional[str] = None,
) -> Figure:
    """Mean and peak crowd count for every video in the dataset."""
    df = pd.DataFrame(summary_rows).sort_values("mean_count", ascending=False)
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.22, df["mean_count"], width=0.42,
           label="Mean count", color="#378ADD", alpha=0.9)
    ax.bar(x + 0.22, df["max_count"],  width=0.42,
           label="Peak count", color="#D85A30", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["vid_id"] if "vid_id" in df.columns else df.index,
        rotation=40, ha="right", fontsize=9,
    )
    ax.set_ylabel("Person count")
    ax.set_title("Per-Video Crowd Statistics", fontweight="bold")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Fig 5 — Speed distribution
# ─────────────────────────────────────────────────────────────

def plot_speed_distribution(
    all_dfs:   Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
) -> Figure:
    """Histogram of mean_speed pooled across all videos."""
    all_speeds = pd.concat(
        [df["mean_speed"] for df in all_dfs.values()],
        ignore_index=True,
    )
    mean_val = float(all_speeds.mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_speeds, bins=60, color="#7F77DD",
            edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mean_val, color="#D85A30", linewidth=2,
               linestyle="--", label=f"Mean = {mean_val:.1f} px/frame")
    ax.set_xlabel("Mean speed (px / frame)")
    ax.set_ylabel("Frame count")
    ax.set_title("Crowd Motion Speed Distribution — All Videos",
                 fontweight="bold")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig