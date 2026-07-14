"""
Enhanced Anomaly Detector for Member 4
Spatial anomaly detection: Crawling, Overcrowding, Zone Imbalance
"""

import numpy as np

from config_loader import (
    OVERCROWDING_THRESHOLD,
    ZONE_IMBALANCE_RATIO,
    get_zone_from_x_coordinate
)

# How much shorter than the "typical person in this frame" someone needs to
# be to count as crawling. E.g. 0.6 means: shorter than 60% of the median
# height of everyone detected in that same frame.
CRAWL_RATIO = 0.6

# A standing person (adult OR child) has a box that's taller than it is
# wide. A person who has fallen/is crawling has a box that flattens out —
# becomes about as wide as, or wider than, it is tall. Checking THIS too
# (not just height) is what tells apart "short person standing normally"
# from "someone down on the ground": a child fails this shape check
# (still tall-and-narrow), while a fallen adult passes it (flattened).
ASPECT_RATIO_THRESHOLD = 1.0  # width / height >= this suggests lying down

# Need at least this many people in frame for "typical height" to mean
# anything. Below this, we skip the crawling check rather than guess.
MIN_PEOPLE_FOR_CRAWL_CHECK = 3


class EnhancedAnomalyDetector:
    """
    Spatial anomaly detection

    Detects:
    - Crawling: people noticeably shorter than others in the same frame
    - Overcrowding: grid cells with more than the threshold people per cell
    - Zone Imbalance: more than the threshold % of people in one zone
    """

    def __init__(self):
        """Initialize with configured thresholds"""
        self.overcrowding_threshold = OVERCROWDING_THRESHOLD
        self.zone_imbalance_ratio = ZONE_IMBALANCE_RATIO

    def detect_crawling(self, detections):
        """
        Detect people noticeably shorter than the typical person in this
        SAME frame — a relative check, not a fixed pixel cutoff. Indicates
        fallen, crawling, or crouching individuals regardless of camera
        distance/zoom.

        Args:
            detections: List of [x1, y1, x2, y2, conf]

        Returns:
            list: Crawling event dictionaries
        """
        crawling_events = []

        if len(detections) < MIN_PEOPLE_FOR_CRAWL_CHECK:
            return crawling_events  # not enough people to judge "typical" height

        heights = np.array([d[3] - d[1] for d in detections])
        median_height = np.median(heights)
        crawl_cutoff = median_height * CRAWL_RATIO

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf = det
            height = y2 - y1
            width = x2 - x1
            aspect_ratio = width / height if height > 0 else 0

            is_short = height < crawl_cutoff
            is_flattened = aspect_ratio >= ASPECT_RATIO_THRESHOLD

            # Require BOTH: short compared to peers AND a flattened/lying
            # shape. Short alone could just be a child standing normally.
            if is_short and is_flattened:
                cx = (x1 + x2) / 2
                zone = get_zone_from_x_coordinate(cx)

                crawling_events.append({
                    'type': 'CRAWLING',
                    'person_id': i,
                    'height': float(height),
                    'width': float(width),
                    'aspect_ratio': float(aspect_ratio),
                    'median_height_this_frame': float(median_height),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'zone': zone,
                    'severity': 'HIGH' if height < crawl_cutoff * 0.7 else 'MEDIUM'
                })

        return crawling_events

    def detect_overcrowding(self, density_grid, zone_densities):
        """
        Detect overcrowded grid cells

        Args:
            density_grid: 2D numpy array
            zone_densities: Dict from get_zone_densities()

        Returns:
            list: Overcrowding event dictionaries
        """
        overcrowded_cells = []
        zone_mapping = {0: 'LEFT', 1: 'LEFT', 2: 'CENTER', 3: 'RIGHT'}

        for r in range(density_grid.shape[0]):
            for c in range(density_grid.shape[1]):
                count = density_grid[r, c]

                if count > self.overcrowding_threshold:
                    if count > 10:
                        severity = 'CRITICAL'
                    elif count > 8:
                        severity = 'HIGH'
                    else:
                        severity = 'MEDIUM'

                    overcrowded_cells.append({
                        'type': 'OVERCROWDING',
                        'grid_pos': (r, c),
                        'count': int(count),
                        'zone': zone_mapping[c],
                        'severity': severity
                    })

        return overcrowded_cells

    def detect_zone_imbalance(self, zone_densities, total_count):
        """
        Detect if crowd is heavily concentrated in one zone
        Classic stampede pattern - everyone rushing one direction

        Args:
            zone_densities: Dict with {'LEFT': n, 'CENTER': n, 'RIGHT': n}
            total_count: Total people in frame

        Returns:
            dict or None: Zone imbalance event or None if balanced
        """
        if total_count < 10:  # Need sufficient people for pattern
            return None

        for zone, count in zone_densities.items():
            ratio = count / total_count if total_count > 0 else 0

            if ratio >= self.zone_imbalance_ratio:
                return {
                    'type': 'ZONE_IMBALANCE',
                    'dominant_zone': zone,
                    'zone_count': count,
                    'total_count': total_count,
                    'concentration_ratio': float(ratio),
                    'severity': 'HIGH' if ratio > 0.85 else 'MEDIUM'
                }

        return None


# For standalone testing
if __name__ == "__main__":
    print("Testing EnhancedAnomalyDetector...")

    detector = EnhancedAnomalyDetector()

    # Test crawling detection — needs BOTH short height AND flattened shape
    sample_detections = [
        [100, 150, 150, 250, 0.9],   # height=100, width=50  (tall/narrow -> standing, typical)
        [200, 150, 250, 245, 0.85],  # height=95,  width=50  (tall/narrow -> standing, typical)
        [300, 150, 350, 240, 0.88],  # height=90,  width=50  (tall/narrow -> standing, typical)
        [400, 200, 430, 230, 0.80],  # height=30,  width=30  (short but still taller-than-wide -> a child, NOT flagged)
        [500, 210, 560, 230, 0.92],  # height=20,  width=60  (short AND wide/flat -> fallen, FLAGGED)
    ]

    crawling = detector.detect_crawling(sample_detections)
    print(f"\nCrawling detections: {len(crawling)} (expected: 1, the flattened one)")
    for event in crawling:
        print(f"  {event['zone']}: height={event['height']:.1f}px, "
              f"aspect_ratio={event['aspect_ratio']:.2f}, "
              f"severity={event['severity']}")