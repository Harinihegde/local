"""
Prediction module — forecasts panic ONSET ahead of time (not detecting
panic that's already happening), using UMN's real 11-event ground truth.

Key design choices, each matching lessons learned earlier in this project:

1. ONSET-ONLY labeling (this was Member 4's original fix, now applied
   with the REAL, complete 11-event ground truth instead of the old
   incomplete 3-event version):
     - Frames already INSIDE a panic segment are discarded entirely —
       forecasting "will panic start" doesn't apply once it's already
       happening.
     - A frame is labeled POSITIVE only if a NEW panic segment starts
       within the next PREDICT_AHEAD frames from that (currently normal)
       frame.
     - Otherwise negative.

2. Sequences built PER SCENE, never crossing a scene boundary — same
   lesson as Member 3's anomaly detection.

3. LOSO evaluation (train on 2 scenes, test on the 3rd, rotate) — same
   honesty principle as everywhere else in this project.

4. Reuses Member 2's already-correct, already-normalized zone-wide
   features (no re-normalization, no re-deriving fps/resolution — those
   are already fixed upstream).
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Reuse the exact same wide-feature loader used for anomaly detection —
# same fold structure, same already-normalized features.
from train_evaluate_umn_loso import load_fold_scene_wide

# Real, confirmed multi-event ground truth (found via banner detection +
# manual visual check — see train_evaluate_umn_loso.py for full provenance).
UMN_GT = {
    'scene1': [(525, 614), (1330, 1439)],
    'scene2': [(353, 532), (1152, 1231), (1766, 1975),
               (2485, 2564), (3354, 3475), (3969, 4142)],
    'scene3': [(598, 637), (1286, 1315), (2103, 2122)],
}
SCENES = ['scene1', 'scene2', 'scene3']

SEQ_LEN = 30          # ~1 second of history at 30fps, used as the forecasting input
PREDICT_AHEAD = 45    # ~1.5 seconds ahead — matches the team's original choice


def is_panic_frame(scene: str, frame_idx: int) -> bool:
    return any(start <= frame_idx <= end for start, end in UMN_GT[scene])


def onset_within_horizon(scene: str, current_frame: int, predict_ahead: int) -> bool:
    """True if a NEW panic segment starts within the next `predict_ahead`
    frames from `current_frame` (which must itself be a normal frame)."""
    return any(current_frame < start <= current_frame + predict_ahead
               for start, _ in UMN_GT[scene])


def aggregate_frame_features(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse zone0_X, zone1_X, ..., zone5_X columns down into ONE column X
    per base feature (averaged across zones), instead of keeping all 6
    zones' numbers separately. This is the fix for the previous version's
    real problem: 144 raw numbers per frame (24 features x 6 zones) with
    only ~5000 training examples was too much detail for too little data
    — the model had no real chance to find a pattern. This brings it down
    to roughly one number per feature type (~24 numbers per frame),
    matching the scale of the team's original, working approach.
    """
    base_features: dict = {}
    for col in wide_df.columns:
        zone_tag, base = col.split("_", 1)  # e.g. "zone0_count_sm" -> "count_sm"
        base_features.setdefault(base, []).append(col)

    agg = pd.DataFrame(index=wide_df.index)
    for base, cols in base_features.items():
        agg[base] = wide_df[cols].mean(axis=1)

    # Total headcount (a SUM across zones, not an average) is its own
    # meaningful number — this is exactly the signal that turned out to
    # matter most for detecting panic (sudden drops), so keep it explicit.
    count_cols = [c for c in wide_df.columns if c.endswith("count_sm")]
    if count_cols:
        agg["total_count_sum"] = wide_df[count_cols].sum(axis=1)

    return agg


def build_prediction_samples(
    wide_df, scene: str, seq_len: int = SEQ_LEN, predict_ahead: int = PREDICT_AHEAD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each frame with enough history behind it and enough room ahead:
      - skip it entirely if it's already mid-panic
      - otherwise, flatten its last `seq_len` frames of SUMMARIZED features
        into one row, and label it based on whether panic starts within
        the next `predict_ahead` frames.
    """
    agg_df = aggregate_frame_features(wide_df)
    values = agg_df.values
    frame_idxs = wide_df.index.to_numpy()
    n = len(values)

    X, y = [], []
    for i in range(seq_len - 1, n - predict_ahead):
        current_frame = int(frame_idxs[i])

        if is_panic_frame(scene, current_frame):
            continue  # already panicking — not a forecasting sample

        window = values[i - seq_len + 1: i + 1].flatten()
        label = 1 if onset_within_horizon(scene, current_frame, predict_ahead) else 0
        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)


def run_fold(loso_root: str, held_out_scene: str, train_scenes: List[str],
             predict_ahead: int = PREDICT_AHEAD) -> dict:
    fold_dir = Path(loso_root) / f"test_{held_out_scene}"
    schema_path = fold_dir / "feature_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    kept_cols = schema["kept_feature_cols"]

    print(f"\n{'='*60}\nFOLD: held-out={held_out_scene} | train={train_scenes}\n{'='*60}")

    X_train_all, y_train_all = [], []
    for scene in train_scenes:
        wide = load_fold_scene_wide(str(fold_dir), scene, "train", kept_cols)
        X, y = build_prediction_samples(wide, scene, predict_ahead=predict_ahead)
        print(f"  {scene}: {len(wide)} frames -> {len(X)} samples, "
              f"{y.sum()} positive ({y.mean()*100:.1f}%)" if len(y) else f"  {scene}: 0 samples")
        if len(X) > 0:
            X_train_all.append(X)
            y_train_all.append(y)

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    print(f"  Total training samples: {X_train.shape}, "
          f"{y_train.sum()} positive ({y_train.mean()*100:.1f}%)")

    model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)

    # --- Tune the decision threshold using ONLY training scenes ---
    # By default, models predict "positive" only above 50% confidence.
    # That default isn't necessarily right for a rare-event problem like
    # this. Same honest method as Member 3's z-threshold tuning: try a
    # mini train/validate split WITHIN the 2 training scenes only (never
    # touching the real test scene), pick whichever confidence cutoff
    # works best there, then apply that one fixed choice to the real
    # unseen scene.
    THRESHOLD_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    mini_f1_per_threshold = {t: [] for t in THRESHOLD_CANDIDATES}

    for val_scene in train_scenes:
        fit_scenes = [s for s in train_scenes if s != val_scene]
        X_fit, y_fit = [], []
        for s in fit_scenes:
            wide = load_fold_scene_wide(str(fold_dir), s, "train", kept_cols)
            X_s, y_s = build_prediction_samples(wide, s, predict_ahead=predict_ahead)
            if len(X_s) > 0:
                X_fit.append(X_s)
                y_fit.append(y_s)
        X_fit = np.concatenate(X_fit, axis=0)
        y_fit = np.concatenate(y_fit, axis=0)

        wide_val = load_fold_scene_wide(str(fold_dir), val_scene, "train", kept_cols)
        X_val, y_val = build_prediction_samples(wide_val, val_scene, predict_ahead=predict_ahead)

        mini_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        mini_model.fit(X_fit, y_fit)
        val_probs = mini_model.predict_proba(X_val)[:, 1]

        for t in THRESHOLD_CANDIDATES:
            preds = (val_probs >= t).astype(int)
            mini_f1_per_threshold[t].append(f1_score(y_val, preds, zero_division=0))

    mean_f1_per_threshold = {t: float(np.mean(v)) for t, v in mini_f1_per_threshold.items()}
    best_threshold = max(mean_f1_per_threshold, key=mean_f1_per_threshold.get)
    print(f"  Threshold tuning (on training scenes only):")
    for t, f1v in mean_f1_per_threshold.items():
        marker = " <- chosen" if t == best_threshold else ""
        print(f"    threshold={t}: mean F1 on train scenes = {f1v:.4f}{marker}")

    # Random Forest with class_weight='balanced' — pays extra attention to
    # the rare "panic about to start" cases instead of defaulting to
    # "probably normal" (which GradientBoosting doesn't directly support).
    # The team's own earlier testing found Random Forest outperformed
    # Gradient Boosting on this exact task — worth re-testing now that the
    # underlying data/labels are fixed.
    model_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42)
    model_rf.fit(X_train, y_train)

    test_wide = load_fold_scene_wide(str(fold_dir), held_out_scene, "test", kept_cols)
    X_test, y_test = build_prediction_samples(test_wide, held_out_scene, predict_ahead=predict_ahead)
    print(f"  Test samples: {X_test.shape}, "
          f"{y_test.sum()} positive ({y_test.mean()*100:.1f}%)" if len(y_test) else "  (no test samples)")

    results_by_model = {}

    # GradientBoosting, default 0.5 threshold
    y_pred = model_gb.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  [GradientBoosting, default threshold] Precision={precision:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    results_by_model["GradientBoosting_default"] = {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    # GradientBoosting, TUNED threshold (chosen from training scenes only)
    test_probs = model_gb.predict_proba(X_test)[:, 1]
    y_pred_tuned = (test_probs >= best_threshold).astype(int)
    precision_t = precision_score(y_test, y_pred_tuned, zero_division=0)
    recall_t = recall_score(y_test, y_pred_tuned, zero_division=0)
    f1_t = f1_score(y_test, y_pred_tuned, zero_division=0)
    cm_t = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    print(f"  [GradientBoosting, tuned threshold={best_threshold}] "
          f"Precision={precision_t:.4f}  Recall={recall_t:.4f}  F1={f1_t:.4f}")
    print(f"    TP={tp_t} FP={fp_t} TN={tn_t} FN={fn_t}")
    results_by_model["GradientBoosting_tuned"] = {
        "threshold": best_threshold,
        "precision": float(precision_t), "recall": float(recall_t), "f1": float(f1_t),
        "tp": int(tp_t), "fp": int(fp_t), "tn": int(tn_t), "fn": int(fn_t),
    }

    # RandomForest, default threshold (kept for comparison)
    y_pred_rf = model_rf.predict(X_test)
    precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    print(f"  [RandomForest, default threshold] Precision={precision_rf:.4f}  "
          f"Recall={recall_rf:.4f}  F1={f1_rf:.4f}")
    print(f"    TP={tp_rf} FP={fp_rf} TN={tn_rf} FN={fn_rf}")
    results_by_model["RandomForest_default"] = {
        "precision": float(precision_rf), "recall": float(recall_rf), "f1": float(f1_rf),
        "tp": int(tp_rf), "fp": int(fp_rf), "tn": int(tn_rf), "fn": int(fn_rf),
    }

    return {"held_out_scene": held_out_scene, **results_by_model}


def run_loso(loso_root: str = "/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/member2_tracking/outputs/features_umn/loso_folds",
             predict_ahead: int = PREDICT_AHEAD):
    results = {}
    for held_out in SCENES:
        train_scenes = [s for s in SCENES if s != held_out]
        results[held_out] = run_fold(loso_root, held_out, train_scenes, predict_ahead)

    print(f"\n{'='*60}\nPREDICTION LOSO SUMMARY (predict {predict_ahead} frames / "
          f"{predict_ahead/30:.1f}s ahead)\n{'='*60}")
    for scene, r in results.items():
        print(f"  test={scene}:")
        print(f"    [GB default]  F1={r['GradientBoosting_default']['f1']:.4f}  "
              f"P={r['GradientBoosting_default']['precision']:.4f}  "
              f"R={r['GradientBoosting_default']['recall']:.4f}")
        print(f"    [GB tuned]    F1={r['GradientBoosting_tuned']['f1']:.4f}  "
              f"P={r['GradientBoosting_tuned']['precision']:.4f}  "
              f"R={r['GradientBoosting_tuned']['recall']:.4f}  "
              f"(threshold={r['GradientBoosting_tuned']['threshold']})")
        print(f"    [RF default]  F1={r['RandomForest_default']['f1']:.4f}  "
              f"P={r['RandomForest_default']['precision']:.4f}  "
              f"R={r['RandomForest_default']['recall']:.4f}")

    gb_default_f1s = [r["GradientBoosting_default"]["f1"] for r in results.values()]
    gb_tuned_f1s = [r["GradientBoosting_tuned"]["f1"] for r in results.values()]
    rf_f1s = [r["RandomForest_default"]["f1"] for r in results.values()]
    mean_gb_default = float(np.mean(gb_default_f1s))
    mean_gb_tuned = float(np.mean(gb_tuned_f1s))
    mean_rf = float(np.mean(rf_f1s))
    print(f"\n  Mean F1 (GB, default threshold): {mean_gb_default:.4f}")
    print(f"  Mean F1 (GB, tuned threshold):    {mean_gb_tuned:.4f}")
    print(f"  Mean F1 (RandomForest):           {mean_rf:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"umn_prediction_results_ahead{predict_ahead}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / f'umn_prediction_results_ahead{predict_ahead}.json'}")

    return results, mean_gb_tuned


if __name__ == "__main__":
    # Test a few different warning times, to see which gives the model
    # the best real chance — predicting further ahead is naturally harder,
    # since the warning signs are more distant/subtle.
    HORIZONS_TO_TEST = {
        "2.0s": 60,
    }

    horizon_summary = {}
    for label, frames in HORIZONS_TO_TEST.items():
        print(f"\n\n{'#'*70}\n# TESTING HORIZON: {label} ({frames} frames)\n{'#'*70}")
        _, mean_f1 = run_loso(predict_ahead=frames)
        horizon_summary[label] = mean_f1

    print(f"\n\n{'='*70}\nHORIZON COMPARISON (mean F1, GB tuned threshold)\n{'='*70}")
    for label, f1 in horizon_summary.items():
        print(f"  {label}: {f1:.4f}")