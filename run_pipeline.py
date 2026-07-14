"""
run_pipeline.py
===============
End-to-end real-time crowd anomaly / stampede detection pipeline.

Wires together the four members' modules WITHOUT modifying any of them:

    Member 1  PersonDetector.detect_people(frame)          -> bboxes
    Member 2  IoUTracker.update(bboxes)                     -> tracked centroids
              + per-frame features computed with ZoneManager
    Member 3  LSTMAnomalyDetectorRealtime.process_frame()   -> anomaly score
    Member 4  SpatialAnalyzer.analyze_frame()
              + RiskScorer.compute_risk_score() / classify_risk_level()
              + EnhancedAlertSystem.trigger_alert()         on HIGH / CRITICAL

Usage:
    python run_pipeline.py <path/to/video.mp4>

Press Q to quit the live window.

Notes on integration (the member modules themselves are unchanged):
  * Member 2's tracker class is named `IoUTracker` (not ByteTracker); its
    update() returns the same (track_id, cx, cy) tuples.
  * Member 3's model lives in the populated root `saved_model/` directory
    (the `member3_prediction/saved_model/` directory is empty).
  * Member 4's SpatialAnalyzer / RiskScorer with compute_risk_score() and
    classify_risk_level() live under src/risk_assessment/; EnhancedAlertSystem
    lives under src/alert_system.py.
  * detect_people() does not resize, so frames are resized to 640x360 here to
    match the zone / grid coordinate space (and member4 config.yaml).
  * For the first 30 frames the LSTM buffer is still filling: process_frame()
    returns None, so we report anomaly score = 0.0 and risk level = LOW.
"""

import os
import sys
import importlib.util
from collections import deque

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

MEMBER1_SRC   = os.path.join(ROOT, "member1_detection", "src")
MEMBER2_SRC   = os.path.join(ROOT, "member2_tracking", "src")
MEMBER3_FILE  = os.path.join(ROOT, "member3_prediction", "anomaly_detection_rollbuff.py")
MEMBER3_MODEL = os.path.join(ROOT, "member3_prediction", "models")
MEMBER3_PRED_SRC = os.path.join(ROOT, "member3_prediction", "src")
PREDICTOR_DIR    = os.path.join(ROOT, "member3_prediction", "saved_model", "predictor")
MEMBER4_SRC   = os.path.join(ROOT, "member4_decision", "src")
MEMBER4_RISK  = os.path.join(MEMBER4_SRC, "risk_assessment")  # SpatialAnalyzer/RiskScorer + config.py

YOLO_MODEL = os.path.join(ROOT, "models", "best.pt")

# risk_assessment/spatial_analyzer.py and risk_scorer.py do `from config import ...`
# (bare import), so their directory must be importable.
if MEMBER4_RISK not in sys.path:
    sys.path.insert(0, MEMBER4_RISK)

# Processing resolution — everything downstream (zones, grid) assumes this frame size.
FRAME_W, FRAME_H = 640, 360

# Zone rectangles for the 640x360 frame, matching member4 config.yaml boundaries.
ZONE_CONFIG = {
    "left":   [0,   0, 213, FRAME_H],
    "center": [213, 0, 426, FRAME_H],
    "right":  [426, 0, 640, FRAME_H],
}

# LSTM input feature order (the 11 *_smooth5 features the model was trained on).
FEATURE_ORDER = [
    "left", "center", "right",
    "total_count", "mean_speed",
    "flow_x", "flow_y",
    "density_left", "density_center", "density_right",
    "active_tracks",
]

SMOOTHING_WINDOW = 5
SPEED_CLIP = 80.0


# ─────────────────────────────────────────────────────────────
# Module loading helpers (no existing files are modified)
# ─────────────────────────────────────────────────────────────
def _load_module(name, path):
    """Import a standalone .py file under an explicit module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_person_detector():
    """
    Load PersonDetector from member1's detector.py.

    detector.py ends with a live `# REAL TIME TEST` block that instantiates the
    detector and opens a video window at import time. We compile only the source
    above that marker so importing the class has no side effects.
    """
    path = os.path.join(MEMBER1_SRC, "detector.py")
    with open(path, "r") as f:
        source = f.read()

    marker = "# REAL TIME TEST"
    if marker in source:
        source = source.split(marker)[0]

    namespace = {"__name__": "member1_detector", "__file__": path}
    try:
        exec(compile(source, path, "exec"), namespace)
    except Exception:
        # If the trailing demo somehow still ran, the class is already defined.
        pass
    return namespace["PersonDetector"]


# ─────────────────────────────────────────────────────────────
# Streaming feature extractor (Member 2 logic, frame-by-frame)
# ─────────────────────────────────────────────────────────────
class StreamingFeatureExtractor:
    """
    Replicates member2_tracking/features.extract_features() in a streaming form
    so it can run per-frame in real time. Maintains previous track positions for
    speed/flow and a rolling window for the smoothed (*_smooth5) values that the
    LSTM expects.
    """

    def __init__(self, zone_manager, smoothing_window=SMOOTHING_WINDOW, speed_clip=SPEED_CLIP):
        self.zm = zone_manager
        self.speed_clip = speed_clip
        self._prev_pos = {}
        self._window = deque(maxlen=smoothing_window)

    def _raw_features(self, frame_tracks):
        dx_list, dy_list, spd_list = [], [], []
        for tid, cx, cy in frame_tracks:
            if tid in self._prev_pos:
                px, py = self._prev_pos[tid]
                dx, dy = cx - px, cy - py
                raw_spd = float(np.hypot(dx, dy))
                if raw_spd > self.speed_clip and raw_spd > 0:
                    scale = self.speed_clip / raw_spd
                    dx *= scale
                    dy *= scale
                dx_list.append(dx)
                dy_list.append(dy)
                spd_list.append(min(raw_spd, self.speed_clip))

        counts = self.zm.count_per_zone(frame_tracks)
        density = self.zm.density_per_zone(frame_tracks)

        return {
            "left":           counts.get("left", 0),
            "center":         counts.get("center", 0),
            "right":          counts.get("right", 0),
            "total_count":    sum(counts.values()),
            "mean_speed":     float(np.mean(spd_list)) if spd_list else 0.0,
            "flow_x":         float(np.mean(dx_list)) if dx_list else 0.0,
            "flow_y":         float(np.mean(dy_list)) if dy_list else 0.0,
            "density_left":   density.get("left", 0.0),
            "density_center": density.get("center", 0.0),
            "density_right":  density.get("right", 0.0),
            "active_tracks":  len(frame_tracks),
        }

    def update(self, frame_tracks):
        """Return (smoothed_feature_dict, smoothed_vector) for this frame."""
        raw = self._raw_features(frame_tracks)
        self._prev_pos = {tid: (cx, cy) for tid, cx, cy in frame_tracks}

        self._window.append(raw)
        smoothed = {
            key: float(np.mean([row[key] for row in self._window]))
            for key in FEATURE_ORDER
        }
        vector = np.array([smoothed[key] for key in FEATURE_ORDER], dtype=np.float32)
        return smoothed, vector


# ─────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────
RISK_COLORS = {
    "LOW":      (0, 200, 0),
    "MEDIUM":   (0, 215, 255),
    "HIGH":     (0, 140, 255),
    "CRITICAL": (0, 0, 255),
}


def draw_overlay(frame, bboxes, risk_level, anomaly_score,
                 predicted_risk="N/A", pred_horizon_sec=None):
    # Green boxes around people.
    for (x1, y1, x2, y2, _conf) in bboxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Current risk level — top left.
    color = RISK_COLORS.get(risk_level, (255, 255, 255))
    cv2.putText(frame, f"RISK: {risk_level}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Predicted future risk — second line, top left.
    horizon_txt = f" (+{pred_horizon_sec:.1f}s)" if pred_horizon_sec else ""
    pred_color = RISK_COLORS.get(predicted_risk, (180, 180, 180))
    cv2.putText(frame, f"PRED{horizon_txt}: {predicted_risk}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2, cv2.LINE_AA)

    # Anomaly score — top right.
    score_text = f"Anomaly: {anomaly_score:.3f}"
    (tw, _th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, score_text, (FRAME_W - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def main(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: video not found: {video_path}")
        return

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # ── Load member modules ───────────────────────────────────
    print("Loading modules...")
    PersonDetector = _load_person_detector()

    tracker_mod = _load_module("member2_tracker", os.path.join(MEMBER2_SRC, "tracker.py"))
    zones_mod   = _load_module("member2_zones",   os.path.join(MEMBER2_SRC, "zones.py"))
    IoUTracker  = tracker_mod.IoUTracker
    ZoneManager = zones_mod.ZoneManager

    member3_mod = _load_module("member3_anomaly", MEMBER3_FILE)
    LSTMAnomalyDetectorRealtime = member3_mod.LSTMAnomalyDetectorRealtime

    predictor_mod = _load_module("member3_predictor", os.path.join(MEMBER3_PRED_SRC, "prediction_model.py"))
    RiskPredictor = predictor_mod.RiskPredictor

    spatial_mod = _load_module("member4_spatial", os.path.join(MEMBER4_RISK, "spatial_analyzer.py"))
    risk_mod    = _load_module("member4_risk",    os.path.join(MEMBER4_RISK, "risk_scorer.py"))
    alert_mod   = _load_module("member4_alerts",  os.path.join(MEMBER4_SRC, "alert_system.py"))
    SpatialAnalyzer     = spatial_mod.SpatialAnalyzer
    RiskScorer          = risk_mod.RiskScorer
    EnhancedAlertSystem = alert_mod.EnhancedAlertSystem

    # ── Instantiate components ────────────────────────────────
    detector  = PersonDetector(model_path=YOLO_MODEL, conf_threshold=0.5)   # device='mps' is set inside detect_people
    tracker   = IoUTracker()
    zone_mgr  = ZoneManager(ZONE_CONFIG)
    features  = StreamingFeatureExtractor(zone_mgr)
    lstm      = LSTMAnomalyDetectorRealtime(MEMBER3_MODEL, sequence_length=30)
    spatial   = SpatialAnalyzer(frame_width=FRAME_W, frame_height=FRAME_H)   # match processing resolution
    scorer    = RiskScorer()
    alerts    = EnhancedAlertSystem()

    # Forward-looking risk predictor (predicts risk ~N seconds ahead).
    try:
        predictor = RiskPredictor.load(PREDICTOR_DIR)
        pred_horizon_sec = predictor.effective_horizon / 20.0   # 20 fps
    except Exception as exc:
        print(f"⚠️  Could not load RiskPredictor from {PREDICTOR_DIR} ({exc}); "
              f"predicted risk disabled.")
        predictor = None
        pred_horizon_sec = None

    # ── Open video ────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return
    print(f"Video opened: {video_path}. Press Q to quit.\n")

    # Rolling window of the last 10 *real* anomaly scores feeding the predictor.
    score_history = deque(maxlen=10)
    predicted_risk = "N/A"

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo ended.")
                break

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # ── Member 1: detection ───────────────────────────
            bboxes = detector.detect_people(frame)          # [(x1,y1,x2,y2,conf), ...]
            people_count = len(bboxes)

            # ── Member 2: tracking + features ─────────────────
            tracked = tracker.update([list(b) for b in bboxes])     # [(track_id, cx, cy), ...]
            feat_row, feat_vector = features.update(tracked)

            # ── Member 3: LSTM anomaly score ──────────────────
            raw_score = lstm.process_frame(feat_vector)     # None for first 30 frames

            if raw_score is None:
                # Warm-up window: buffer still filling.
                anomaly_score = 0.0
                risk_level = "LOW"
            else:
                anomaly_score = raw_score

                # ── Forward prediction: future risk from last 10 scores ──
                score_history.append(anomaly_score)
                if predictor is not None and len(score_history) == score_history.maxlen:
                    predicted_risk = predictor.predict_risk_level(list(score_history))

                # ── Member 4: spatial + risk + alerts ─────────
                features_row = {
                    "left_smooth":   feat_row["left"],
                    "center_smooth": feat_row["center"],
                    "right_smooth":  feat_row["right"],
                }
                spatial_anomalies, spatial_score = spatial.analyze_frame(bboxes, features_row)
                risk_score, _components = scorer.compute_risk_score(
                    anomaly_score, spatial_score, people_count
                )
                risk_level = scorer.classify_risk_level(risk_score)

                if risk_level in ("HIGH", "CRITICAL"):
                    risk_factors = []
                    if anomaly_score > lstm.threshold:
                        risk_factors.append(f"LSTM:{anomaly_score:.2f}")
                    if spatial_anomalies.get("crawling"):
                        risk_factors.append("CRAWLING")
                    if spatial_anomalies.get("overcrowding"):
                        risk_factors.append("OVERCROWDING")
                    if spatial_anomalies.get("zone_imbalance"):
                        risk_factors.append("ZONE_IMBALANCE")

                    anomaly = {
                        "type": "CROWD_ANOMALY",
                        "severity": risk_level,
                        "anomaly_score": float(anomaly_score),
                        "spatial_score": float(spatial_score),
                        "dominant_zone": spatial_anomalies.get("dominant_zone") or "UNKNOWN",
                        "crawling": bool(spatial_anomalies.get("crawling")),
                        "overcrowding": bool(spatial_anomalies.get("overcrowding")),
                        "zone_imbalance": bool(spatial_anomalies.get("zone_imbalance")),
                    }
                    risk_info = {
                        "risk_score": float(risk_score),
                        "risk_level": risk_level,
                        "risk_factors": risk_factors,
                    }
                    alerts.trigger_alert(video_id, frame_num, anomaly, "MEMBER4", risk_info)

            # ── Draw + show ───────────────────────────────────
            draw_overlay(frame, bboxes, risk_level, anomaly_score,
                         predicted_risk, pred_horizon_sec)
            cv2.imshow("Stampede Detection", frame)

            # ── Per-frame log ─────────────────────────────────
            pred_label = (f"Pred(+{pred_horizon_sec:.1f}s): {predicted_risk}"
                          if pred_horizon_sec else f"Pred: {predicted_risk}")
            print(f"Frame {frame_num:5d} | People: {people_count:3d} | "
                  f"Anomaly: {anomaly_score:.3f} | Risk: {risk_level} | {pred_label}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuit requested.")
                break

            frame_num += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = alerts.get_summary()
        print(f"\nProcessed {frame_num} frames. "
              f"Total alerts: {summary['total_alerts']} "
              f"(risk distribution: {summary['risk_distribution']}).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <path/to/video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
