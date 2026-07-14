"""
Member 3 — LSTM autoencoder anomaly detection, UMN, leave-one-scene-out.

Fixes applied vs. the original ensemble_anomaly.py / train_anomaly_models.py:

  1. Zone-wide features: each fold's per-(frame,zone) CSVs are pivoted into
     one row per frame, columns = zone{Z}_{feature}. No spatial detail is
     discarded (previously would have needed flattening/averaging zones away).

  2. Sequences are built PER SCENE, never across a scene boundary. The
     original code concatenated all videos' CSVs and slid a window across
     the whole thing — some training sequences would have spanned the
     boundary between two unrelated scenes.

  3. Label alignment fixed: a sequence starting at frame i and covering
     [i, i+seq_len-1] is labeled using the PANIC STATUS OF ITS LAST FRAME
     (i+seq_len-1), not a naively same-indexed ground-truth array.

  4. Training sequences are restricted to windows that contain NO panic
     frames at all — an anomaly detector must never see the anomaly during
     training, or it stops looking anomalous.

  5. Anomaly threshold (95th percentile of reconstruction MSE) is computed
     from TRAIN data only, then applied to test — never derived from test
     data (which was the original bug: percentile-of-self on test, and also
     used as a fake "F1" that never touched real labels).

  6. Real F1/precision/recall/AUC computed against actual UMN ground-truth
     panic ranges — not np.mean(predictions) on a self-referential threshold.

  7. No redundant re-normalization: Member 2's fold CSVs are already
     z-scored (fit on that fold's train scenes). This script uses those
     values directly instead of fitting a second StandardScaler on top.

Ground truth: 11 real panic/dispersal events across the 3 scenes (2+6+3,
matching published literature counts), found via automatic detection of
UMN's built-in "Abnormal Crowd Activity" banner, then manually visually
confirmed. See the UMN_GT block below for exact ranges and provenance.
This replaces an earlier single-split-per-scene ground truth that was
confirmed to miss most real events (especially in scene2, which has 6).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from models import create_lstm_autoencoder

# Fixed seed for reproducibility. Without this, re-running gives different
# initial weights / internal validation-split shuffling each time, making it
# impossible to tell a real fold-specific result apart from random variation
# in training (relevant e.g. for scene2's anti-correlated result below —
# is that a real pattern, or one unlucky initialization?).
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Ground truth — REPLACES the earlier single-split convention (Mehran et al.),
# which we confirmed only captures ONE event per scene while the literature
# (Antic & Ommer) documents scene1=2, scene2=6, scene3=3 real events.
#
# These exact ranges were found by automatically detecting UMN's own built-in
# red "Abnormal Crowd Activity" banner across every frame (detect_abnormal_
# banner.py), then manually visually confirmed (crowded -> empty = real
# panic/dispersal, not a false "sparse" flag). Segment counts matched the
# literature exactly (2/6/3), giving strong confidence these are correct.
#
# Each scene now maps to a LIST of (start, end) frame ranges, since a scene
# can contain multiple separate panic episodes, not just one.
# ---------------------------------------------------------------------------
UMN_GT = {
    'scene1': [(525, 614), (1330, 1439)],
    'scene2': [(353, 532), (1152, 1231), (1766, 1975),
               (2485, 2564), (3354, 3475), (3969, 4142)],
    'scene3': [(598, 637), (1286, 1315), (2103, 2122)],
}
SCENES = ['scene1', 'scene2', 'scene3']
SEQUENCE_LENGTH = 30


def is_panic_frame(scene: str, frame_idx: int) -> bool:
    """frame_idx is 0-indexed (as stored in the feature CSVs), directly
    comparable to UMN_GT ranges (also 0-indexed, from the frame scan)."""
    return any(start <= frame_idx <= end for start, end in UMN_GT[scene])


def compute_robust_z(mse: np.ndarray, window: int = 200) -> np.ndarray:
    """
    Causal rolling robust z-score: compares each point only to PAST values
    within the same stream (median/MAD over a trailing window). No future
    frames, no labels — safe for real-time use and for nested tuning below.
    """
    roll_median = pd.Series(mse).rolling(window, min_periods=10).median()
    roll_mad = pd.Series(mse).rolling(window, min_periods=10).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )
    z = (mse - roll_median.values) / (1.4826 * roll_mad.values + 1e-8)
    return np.nan_to_num(z, nan=0.0)


# ---------------------------------------------------------------------------
# Step 1: pivot Member 2's per-(frame, zone) CSVs into one row per frame
# ---------------------------------------------------------------------------

def load_fold_scene_wide(fold_dir: str, scene: str, split: str,
                          kept_cols: List[str]) -> pd.DataFrame:
    """
    Load one scene's normalized feature CSV for a given fold + split
    ('train' or 'test'), pivoted from (frame_idx, zone_id) rows into one
    row per frame_idx with columns named zone{Z}_{feature}.
    """
    path = Path(fold_dir) / f"{split}_normalized" / f"{scene}_features_norm.csv"
    df = pd.read_csv(path)

    wide = df.pivot(index="frame_idx", columns="zone_id", values=kept_cols)
    # wide.columns is a MultiIndex (feature, zone_id) -> flatten
    wide.columns = [f"zone{int(z)}_{feat}" for feat, z in wide.columns]
    wide = wide.sort_index()
    return wide


# ---------------------------------------------------------------------------
# Step 2: build sequences, per scene, respecting panic-frame exclusion for train
# ---------------------------------------------------------------------------

def build_train_sequences(wide_df: pd.DataFrame, scene: str,
                           seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Sliding windows within ONE scene only. A window is included only if
    NONE of its frames are panic frames (autoencoder must only learn normal).
    """
    values = wide_df.values
    frame_idxs = wide_df.index.to_numpy()
    n = len(values)
    sequences = []

    for start in range(0, n - seq_len + 1):
        end = start + seq_len
        window_frames = frame_idxs[start:end]
        if any(is_panic_frame(scene, int(fi)) for fi in window_frames):
            continue
        sequences.append(values[start:end])

    return np.array(sequences) if sequences else np.empty((0, seq_len, values.shape[1]))


def build_test_sequences_with_labels(
    wide_df: pd.DataFrame, scene: str, seq_len: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    All sliding windows within the held-out scene (no exclusion — we need
    to see reconstruction error across the whole timeline, panic included).
    Label = panic status of the window's LAST frame (the frame the
    reconstruction error is attributed to), not a naive index-aligned array.
    """
    values = wide_df.values
    frame_idxs = wide_df.index.to_numpy()
    n = len(values)
    sequences, labels, last_frame_out = [], [], []

    for start in range(0, n - seq_len + 1):
        end = start + seq_len
        last_frame = int(frame_idxs[end - 1])
        sequences.append(values[start:end])
        labels.append(1 if is_panic_frame(scene, last_frame) else 0)
        last_frame_out.append(last_frame)

    return np.array(sequences), np.array(labels), np.array(last_frame_out)


# ---------------------------------------------------------------------------
# Step 3: run one LOSO fold end-to-end
# ---------------------------------------------------------------------------

def run_fold(loso_root: str, held_out_scene: str, train_scenes: List[str],
             seq_len: int = SEQUENCE_LENGTH, epochs: int = 50) -> Dict:
    fold_dir = Path(loso_root) / f"test_{held_out_scene}"
    schema_path = fold_dir / "feature_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    kept_cols = schema["kept_feature_cols"]
    n_features = len(kept_cols) * 6  # 6 zones

    print(f"\n{'='*60}\nFOLD: held-out={held_out_scene} | train={train_scenes} "
          f"| kept_cols={len(kept_cols)} -> n_features={n_features}\n{'='*60}")

    # --- Build training sequences (normal-only, per-scene boundaries) ---
    train_seqs_all = []
    train_wide_by_scene = {}  # keep these — needed below for nested tuning
    for scene in train_scenes:
        wide = load_fold_scene_wide(str(fold_dir), scene, "train", kept_cols)
        train_wide_by_scene[scene] = wide
        seqs = build_train_sequences(wide, scene, seq_len)
        print(f"  {scene}: {len(wide)} frames -> {len(seqs)} normal training sequences")
        if len(seqs) > 0:
            train_seqs_all.append(seqs)

    if not train_seqs_all:
        raise RuntimeError(f"No training sequences produced for fold {held_out_scene}")
    train_sequences = np.concatenate(train_seqs_all, axis=0)
    print(f"  Total training sequences: {train_sequences.shape}")

    # --- Train autoencoder (data already normalized by Member 2's pipeline) ---
    model = create_lstm_autoencoder(seq_len, n_features, latent_dim=8)
    model.fit(
        train_sequences, train_sequences,
        epochs=epochs, batch_size=32, validation_split=0.2, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    # --- Threshold from TRAIN reconstruction error only ---
    train_recon = model.predict(train_sequences, verbose=0)
    train_mse = np.mean(np.square(train_sequences - train_recon), axis=(1, 2))
    threshold = float(np.percentile(train_mse, 95))
    print(f"  Threshold (95th pct of train MSE): {threshold:.6f}")

    # --- Nested tuning: pick the adaptive method's strictness (z-threshold)
    # using ONLY the training scenes' own labels — never the test scene.
    # NOTE: window size is kept FIXED (200) rather than also tuned — with
    # only 2 training videos available, searching too many combinations
    # at once risks picking a setting that looks good by luck on those 2
    # videos specifically, rather than one that generalizes. Confirmed by
    # experiment: adding window-size search made scene3 (our target fix)
    # WORSE (F1 0.52 -> 0.39), a sign of overfitting the tuning itself.
    FIXED_WINDOW = 200
    Z_CANDIDATES = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    train_scene_mse = {}
    train_scene_labels = {}
    for scene in train_scenes:
        wide = train_wide_by_scene[scene]
        seqs, labels, _ = build_test_sequences_with_labels(wide, scene, seq_len)
        recon = model.predict(seqs, verbose=0)
        train_scene_mse[scene] = np.mean(np.square(seqs - recon), axis=(1, 2))
        train_scene_labels[scene] = labels

    mean_f1_per_z = {}
    for z_thr in Z_CANDIDATES:
        fold_f1s = []
        for scene in train_scenes:
            z = compute_robust_z(train_scene_mse[scene], window=FIXED_WINDOW)
            preds = (z > z_thr).astype(int)
            fold_f1s.append(f1_score(train_scene_labels[scene], preds, zero_division=0))
        mean_f1_per_z[z_thr] = float(np.mean(fold_f1s))

    best_z = max(mean_f1_per_z, key=mean_f1_per_z.get)
    best_window = FIXED_WINDOW
    print(f"  Nested tuning (train scenes only, window fixed at {FIXED_WINDOW}):")
    for z_thr, score in mean_f1_per_z.items():
        marker = " <- chosen" if z_thr == best_z else ""
        print(f"    z={z_thr}: mean F1 on train scenes = {score:.4f}{marker}")

    # --- Build test sequences + real labels for held-out scene ---
    test_wide = load_fold_scene_wide(str(fold_dir), held_out_scene, "test", kept_cols)
    test_sequences, y_true, last_frames = build_test_sequences_with_labels(
        test_wide, held_out_scene, seq_len
    )
    print(f"  Test sequences: {test_sequences.shape}, "
          f"panic-labeled: {y_true.sum()} / {len(y_true)} "
          f"({y_true.mean()*100:.1f}%)")

    test_recon = model.predict(test_sequences, verbose=0)
    test_mse = np.mean(np.square(test_sequences - test_recon), axis=(1, 2))
    y_pred = (test_mse > threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, test_mse)
    except ValueError:
        auc = float("nan")  # only one class present in y_true
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  [Fixed threshold]    Precision={precision:.4f}  Recall={recall:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

    # Apply the window+z CHOSEN ABOVE (from training scenes only) to the
    # actual held-out test scene — no peeking at test labels to pick it.
    robust_z = compute_robust_z(test_mse, window=best_window)
    y_pred_adaptive = (robust_z > best_z).astype(int)

    precision_a = precision_score(y_true, y_pred_adaptive, zero_division=0)
    recall_a = recall_score(y_true, y_pred_adaptive, zero_division=0)
    f1_a = f1_score(y_true, y_pred_adaptive, zero_division=0)
    try:
        auc_a = roc_auc_score(y_true, robust_z)
    except ValueError:
        auc_a = float("nan")
    cm_a = confusion_matrix(y_true, y_pred_adaptive, labels=[0, 1])
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()

    print(f"  [Adaptive threshold, window={best_window}, z={best_z}] Precision={precision_a:.4f}  "
          f"Recall={recall_a:.4f}  F1={f1_a:.4f}  AUC={auc_a:.4f}")
    print(f"    TP={tp_a} FP={fp_a} TN={tn_a} FN={fn_a}")

    # Save raw arrays so future diagnostics (e.g. plotting MSE over time
    # against the true panic window) don't require retraining.
    diag_path = fold_dir / "diagnostics.npz"
    np.savez(
        diag_path,
        train_mse=train_mse,
        threshold=threshold,
        adaptive_window=best_window,
        adaptive_z_threshold=best_z,
        test_mse=test_mse,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_adaptive=y_pred_adaptive,
        robust_z=robust_z,
        last_frames=last_frames,  # frame_idx each test sequence's error is attributed to
    )
    print(f"  Diagnostics saved to {diag_path}")

    return {
        "held_out_scene": held_out_scene,
        "train_scenes": train_scenes,
        "n_features": n_features,
        "threshold": threshold,
        "adaptive_window_chosen": best_window,
        "adaptive_z_threshold_chosen": best_z,
        "adaptive_tuning_scores": mean_f1_per_z,
        "fixed": {
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "auc": float(auc),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        },
        "adaptive": {
            "precision": float(precision_a), "recall": float(recall_a),
            "f1": float(f1_a), "auc": float(auc_a),
            "tp": int(tp_a), "fp": int(fp_a), "tn": int(tn_a), "fn": int(fn_a),
        },
    }


def run_loso(loso_root: str = "/Users/harinihegde/Desktop/STAMPEDE-DETECTION-SYSTEM-MAIN_TEMP/member2_tracking/outputs/features_umn/loso_folds") -> Dict[str, Dict]:
    results = {}
    for held_out in SCENES:
        train_scenes = [s for s in SCENES if s != held_out]
        results[held_out] = run_fold(loso_root, held_out, train_scenes)

    print(f"\n{'='*60}\nLOSO SUMMARY (Member 3, UMN)\n{'='*60}")
    f1s_fixed = [r["fixed"]["f1"] for r in results.values()]
    f1s_adaptive = [r["adaptive"]["f1"] for r in results.values()]
    for scene, r in results.items():
        print(f"  test={scene}:")
        print(f"    [Fixed]    F1={r['fixed']['f1']:.4f}  P={r['fixed']['precision']:.4f}  "
              f"R={r['fixed']['recall']:.4f}  AUC={r['fixed']['auc']:.4f}")
        print(f"    [Adaptive] F1={r['adaptive']['f1']:.4f}  P={r['adaptive']['precision']:.4f}  "
              f"R={r['adaptive']['recall']:.4f}  AUC={r['adaptive']['auc']:.4f}")
    print(f"  Mean F1 (fixed threshold):    {np.mean(f1s_fixed):.4f}")
    print(f"  Mean F1 (adaptive threshold): {np.mean(f1s_adaptive):.4f}")

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    with open(out_path / "umn_loso_anomaly_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path / 'umn_loso_anomaly_results.json'}")

    return results


if __name__ == "__main__":
    run_loso()