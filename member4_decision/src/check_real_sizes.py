"""
Looks at REAL crowd counts in UMN (total people per frame, max people in
any single grid cell), separately for panic frames vs normal frames.
This tells us what "high density" or "overcrowded" actually looks like
in THIS dataset, instead of reusing numbers borrowed from elsewhere.

Usage:
    python3 check_real_crowd_sizes.py
"""

import json
import numpy as np

from density_estimator import EnhancedDensityEstimator

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


def describe(name, values):
    values = np.array(values)
    print(f"  {name}: n={len(values)}, mean={values.mean():.1f}, "
          f"median={np.median(values):.1f}, "
          f"p75={np.percentile(values, 75):.1f}, "
          f"p90={np.percentile(values, 90):.1f}, "
          f"p95={np.percentile(values, 95):.1f}, "
          f"max={values.max():.1f}")


if __name__ == "__main__":
    density_est = EnhancedDensityEstimator()

    with open(UMN_DETECTIONS_PATH) as f:
        all_det = json.load(f)

    total_counts_normal, total_counts_panic = [], []
    max_cell_normal, max_cell_panic = [], []

    for scene in SCENES:
        dets_per_frame = all_det[scene]["detections"]
        for frame_idx, dets in enumerate(dets_per_frame):
            grid = density_est.compute_density_grid(dets)
            total_count = len(dets)
            max_cell = int(grid.max()) if grid.size else 0

            if is_panic_frame(scene, frame_idx):
                total_counts_panic.append(total_count)
                max_cell_panic.append(max_cell)
            else:
                total_counts_normal.append(total_count)
                max_cell_normal.append(max_cell)

    print("=== TOTAL PEOPLE PER FRAME ===")
    describe("Normal frames", total_counts_normal)
    describe("Panic frames ", total_counts_panic)

    print("\n=== MAX PEOPLE IN ANY SINGLE GRID CELL ===")
    describe("Normal frames", max_cell_normal)
    describe("Panic frames ", max_cell_panic)