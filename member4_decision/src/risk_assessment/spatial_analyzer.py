import numpy as np
from config import THRESHOLDS, GRID_ROWS, GRID_COLS, FRAME_WIDTH, FRAME_HEIGHT


class SpatialAnalyzer:
    def __init__(self, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
                 grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.grid_rows    = grid_rows
        self.grid_cols    = grid_cols
        self.cell_width   = frame_width  // grid_cols
        self.cell_height  = frame_height // grid_rows

    def detect_crawling(self, bboxes):
        crawling_indices = [i for i, b in enumerate(bboxes) if (b[3] - b[1]) < THRESHOLDS['crawling_height']]
        return len(crawling_indices) > 0, len(crawling_indices), crawling_indices

    def compute_grid_density(self, bboxes):
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for bbox in bboxes:
            x1, y1, x2, y2, _ = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col = max(0, min(int(cx // self.cell_width),  self.grid_cols - 1))
            row = max(0, min(int(cy // self.cell_height), self.grid_rows - 1))
            grid[row, col] += 1
        return grid

    def detect_overcrowding(self, grid_counts):
        overcrowded = [(r, c, grid_counts[r, c])
                       for r in range(self.grid_rows)
                       for c in range(self.grid_cols)
                       if grid_counts[r, c] > THRESHOLDS['overcrowding_count']]
        return len(overcrowded) > 0, overcrowded

    def detect_zone_imbalance(self, features_row):
        left   = features_row.get('left_smooth',   features_row.get('left',   0))
        center = features_row.get('center_smooth', features_row.get('center', 0))
        right  = features_row.get('right_smooth',  features_row.get('right',  0))
        total  = left + center + right
        if total == 0:
            return False, None, 0.0
        pcts = {'LEFT': left/total, 'CENTER': center/total, 'RIGHT': right/total}
        dominant = max(pcts, key=pcts.get)
        return pcts[dominant] > THRESHOLDS['zone_imbalance'], dominant, pcts[dominant]

    def analyze_frame(self, bboxes, features_row):
        is_crawling, crawl_count, crawl_idx = self.detect_crawling(bboxes)
        grid_counts = self.compute_grid_density(bboxes)
        is_overcrowded, overcrowded_cells = self.detect_overcrowding(grid_counts)
        is_imbalanced, dominant_zone, imbalance_pct = self.detect_zone_imbalance(features_row)

        spatial_anomalies = {
            'crawling': is_crawling, 'crawling_count': crawl_count,
            'overcrowding': is_overcrowded, 'overcrowded_cells': len(overcrowded_cells),
            'zone_imbalance': is_imbalanced, 'dominant_zone': dominant_zone,
            'imbalance_percentage': imbalance_pct, 'grid_density': grid_counts
        }

        spatial_score = 0.0
        if is_crawling:    spatial_score += 0.4
        if is_overcrowded: spatial_score += 0.3
        if is_imbalanced:  spatial_score += 0.3
        spatial_score = min(spatial_score, 1.0)

        return spatial_anomalies, spatial_score