"""
Bridges Member 3's output to Member 4's expected input.

Member 3 now produces a raw "how far from normal" number (a z-score) that
can be any size — 2, 5, 10, etc. Member 4's RiskScorer expects a clean
number between 0 and 1 (it multiplies this by a weight and adds it to
other things, capped at 1.0 total — a raw z-score would blow past that
immediately).

This turns Member 3's z-score into a smooth 0-to-1 "how confident are we
this is anomaly" number, using the SAME cutoff Member 3 already decided
was the line between normal and anomaly:

  - score sits at the cutoff           -> 0.5 (right on the fence)
  - score is far ABOVE the cutoff      -> close to 1.0 (confident: anomaly)
  - score is far BELOW the cutoff      -> close to 0.0 (confident: normal)
"""

import numpy as np


def zscore_to_probability(z: float, cutoff: float, steepness: float = 2.0) -> float:
    """
    z: Member 3's raw anomaly z-score for one frame.
    cutoff: the z-value Member 3 treats as the normal/anomaly boundary
            (this is the 'best_z' chosen during Member 3's tuning).
    steepness: how quickly the dial moves from 0 to 1 around the cutoff.
               Smaller = more gradual, larger = more like a hard switch.
               Started as a reasonable default, not yet carefully tuned.
    """
    return float(1.0 / (1.0 + np.exp(-(z - cutoff) / steepness)))


# --- Example / standalone check ---
if __name__ == "__main__":
    cutoff = 5.0  # example — this should be the real best_z from Member 3

    print("z-score -> probability (cutoff = 5.0):")
    for z in [0, 2, 4, 5, 6, 8, 12]:
        p = zscore_to_probability(z, cutoff)
        print(f"  z={z:>4} -> {p:.3f}")