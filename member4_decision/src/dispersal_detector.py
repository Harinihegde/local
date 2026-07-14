"""
Dispersal Detector for Member 4

Detects a SUDDEN DROP in crowd count, relative to that video's own
recent history — not a fixed number.

WHY A DROP, NOT A HIGH COUNT:
Real UMN data shows panic frames average only ~2-4 people visible,
while normal frames average ~15. Panic here looks like people fleeing
OUT of the camera's view, not crowding together. So "high density =
danger" was backwards for this kind of footage.

WHY RELATIVE TO RECENT HISTORY, NOT ONE FIXED NUMBER:
A fixed "drop below 5 people" rule would only make sense for a video
whose normal crowd is around 15. A different video with a normal crowd
of 200 people would need a totally different fixed number. Comparing
each frame to ITS OWN video's recent normal count (causal — only past
frames, same idea as Member 3's rolling baseline) works the same way
regardless of the video's actual scale, and works identically in
real-time (one frame at a time) as in batch processing.
"""

from collections import deque
import numpy as np


class DispersalDetector:
    """
    One instance per video (like the tracker) — call update() once per
    frame, in order. Maintains a rolling window of past crowd counts.
    """

    def __init__(self, window=200, z_threshold=0.7):
        self.window = window
        self.z_threshold = z_threshold
        self.history = deque(maxlen=window)

    def update(self, total_count):
        """
        Call once per frame with that frame's total people count.
        Returns (is_dispersal: bool, drop_z_score: float).

        drop_z_score is positive when count is BELOW recent normal
        (a drop), near zero when normal, negative when count is
        unusually HIGH (not what we're flagging here, but visible
        for debugging).
        """
        if len(self.history) < 10:
            # Not enough history yet to know what "normal" looks like
            # for this video — don't guess.
            self.history.append(total_count)
            return False, 0.0

        hist = np.array(self.history)
        median = np.median(hist)
        mad = np.median(np.abs(hist - median))
        drop_z = (median - total_count) / (1.4826 * mad + 1e-8)

        self.history.append(total_count)
        return drop_z > self.z_threshold, float(drop_z)


# For standalone testing
if __name__ == "__main__":
    print("Testing DispersalDetector...")

    detector = DispersalDetector(window=50, z_threshold=3.0)

    # Simulate: normal crowd (~15 people) for a while, then a sudden drop
    np.random.seed(0)
    normal_counts = np.random.poisson(15, size=60)
    panic_counts = [3, 2, 4, 2, 1, 3]  # sudden drop
    sequence = list(normal_counts) + panic_counts

    for i, count in enumerate(sequence):
        is_dispersal, z = detector.update(count)
        marker = " <-- FLAGGED" if is_dispersal else ""
        if i >= 55:  # only print the interesting tail end
            print(f"  frame {i}: count={count}, drop_z={z:.2f}{marker}")