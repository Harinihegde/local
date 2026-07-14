"""
BoT-SORT-family tracker: Kalman-filter motion prediction + IoU matching.
One implementation only — used identically for batch and real-time input (Section 3.1/3.6).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between box arrays shaped (N,4) and (M,4), returns (N,M)."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class KalmanBox:
    """
    Constant-velocity Kalman filter for a single bounding box.
    State: [cx, cy, w, h, vcx, vcy, vw, vh]
    """
    _next_id = 1

    def __init__(self, box: np.ndarray):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]

        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)

        # State transition
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (observe cx,cy,w,h)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # Covariances
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4, 1e4])
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.01, 0.01])
        self.R = np.diag([1., 1., 10., 10.])

        self.track_id = KalmanBox._next_id
        KalmanBox._next_id += 1
        self.age = 0           # frames since last confirmed match
        self.hit_streak = 1    # consecutive matched frames
        self.history: List[Tuple[float, float, float, float]] = []  # (x1,y1,x2,y2) per frame

        self._record()

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.hit_streak = 0

    def update(self, box: np.ndarray):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        z = np.array([cx, cy, w, h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.age = 0
        self.hit_streak += 1
        self._record()

    def predicted_box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _record(self):
        b = self.predicted_box()
        self.history.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))


class Tracker:
    """
    BoT-SORT-family multi-object tracker.
    One instance per video — call update(detections) once per frame.
    """

    def __init__(self, max_age: int = 10, min_iou: float = 0.2,
                 min_track_len: int = 3, max_dist: float = 50.0):
        self.max_age = max_age
        self.min_iou = min_iou
        self.min_track_len = min_track_len
        self.max_dist = max_dist
        self.tracks: List[KalmanBox] = []
        self.frame_idx = 0
        KalmanBox.reset_id_counter()

    def update(self, detections: List[List[float]]) -> List[Dict]:
        """
        detections: list of [x1, y1, x2, y2, conf] for one frame.
        Returns active tracks as list of dicts with keys:
          track_id, x1, y1, x2, y2, frame_idx
        Only returns tracks that have hit_streak >= 1 (matched this frame).
        """
        det_boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=float) \
            if detections else np.empty((0, 4))

        # Predict all existing tracks forward
        for t in self.tracks:
            t.predict()

        # Match detections to predicted boxes via IoU
        matched_det = set()
        matched_trk = set()

        if len(self.tracks) > 0 and len(det_boxes) > 0:
            pred_boxes = np.array([t.predicted_box() for t in self.tracks])

            # Distance gate: zero out IoU for pairs whose predicted-centroid to
            # detection-centroid distance exceeds max_dist (pixels).
            pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
            det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
            det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2
            dist_mat = np.sqrt(
                (pred_cx[:, None] - det_cx[None, :]) ** 2 +
                (pred_cy[:, None] - det_cy[None, :]) ** 2
            )  # (n_tracks, n_dets)

            iou_mat = _iou(pred_boxes, det_boxes)
            iou_mat[dist_mat > self.max_dist] = 0.0  # gate out implausible pairs

            # Hungarian assignment on the gated IoU matrix
            cost = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, di in zip(row_ind, col_ind):
                if iou_mat[ti, di] < self.min_iou:
                    continue
                self.tracks[ti].update(det_boxes[di])
                matched_trk.add(ti)
                matched_det.add(di)

        # Start new tracks for unmatched detections
        for di in range(len(det_boxes)):
            if di not in matched_det:
                self.tracks.append(KalmanBox(det_boxes[di]))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Return active (matched this frame) tracks
        # vx/vy are Kalman-smoothed velocities in pixels/frame — multiply by fps for px/s
        result = []
        for t in self.tracks:
            if t.hit_streak >= 1:
                b = t.predicted_box()
                result.append({
                    "track_id": t.track_id,
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                    "vx": float(t.x[4]),  # Kalman velocity cx, pixels/frame
                    "vy": float(t.x[5]),  # Kalman velocity cy, pixels/frame
                    "frame_idx": self.frame_idx,
                })

        self.frame_idx += 1
        return result

    def get_confirmed_tracks(self) -> Dict[int, List[Dict]]:
        """
        After processing all frames, returns only tracks with length >= min_track_len.
        Keys are track_ids; values are lists of frame-level dicts.
        """
        # Gather per-track history from KalmanBox.history
        # We need to reconstruct from per-frame updates stored during tracking.
        # This is called after full video processing — see run_tracker().
        raise NotImplementedError("Use run_tracker() which builds this incrementally.")


def run_tracker(detections_per_frame: List[List[List[float]]],
                max_age: int = 10, min_iou: float = 0.2,
                min_track_len: int = 3,
                max_dist: float = 50.0) -> Tuple[List[List[Dict]], Dict[int, int]]:
    """
    Run tracker over all frames of one video.

    Returns:
      frame_tracks: list[frame_idx] -> list of active track dicts for that frame
                    Only includes tracks that pass min_track_len filter.
      track_lengths: dict[track_id -> num_frames_active] for diagnostics.
    """
    tracker = Tracker(max_age=max_age, min_iou=min_iou, min_track_len=min_track_len,
                      max_dist=max_dist)

    # Pass 1: collect per-frame raw results and count track lengths
    raw_frames: List[List[Dict]] = []
    track_len: Dict[int, int] = {}

    for frame_dets in detections_per_frame:
        active = tracker.update(frame_dets)
        raw_frames.append(active)
        for t in active:
            tid = t["track_id"]
            track_len[tid] = track_len.get(tid, 0) + 1

    # Pass 2: filter out short noisy tracks (Section 3.6 — actually drop them)
    valid_ids = {tid for tid, length in track_len.items() if length >= min_track_len}
    frame_tracks = [
        [t for t in frame if t["track_id"] in valid_ids]
        for frame in raw_frames
    ]

    return frame_tracks, track_len