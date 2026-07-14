"""
Main pipeline runner — processes videos, produces per-video feature CSVs,
plus pruned schema + scaler.

Three modes, selected by cfg['split']['mode']:

  1. (no 'split' key)      File-based split (original, e.g. UCSD):
                            separate detection_train / detection_test JSON files.
  2. mode: 'scene'          Fixed scene split (e.g. quick UMN iteration):
                            one detection file, train_scenes / test_scenes lists.
                            NOTE: fit-once scaler on the fixed train scenes.
                            Fast, but only one train/test assignment — not a
                            statistically meaningful generalization test with
                            only 3 independent UMN events.
  3. mode: 'loso'           Leave-one-scene-out (e.g. UMN final evaluation):
                            features extracted once per scene, then for each
                            scene held out in turn: prune + fit scaler on the
                            OTHER scenes only, transform the held-out scene
                            with that fold's own scaler. No cross-fold leakage.
"""

import json as _json
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.tracker import run_tracker
from src.features import extract_features, ALL_FEATURE_COLS
from src.prune import prune_collinear
from src.normalize import StandardScaler


FORECAST_TARGETS = [
    "density_sm", "speed_mean_sm", "flow_magnitude_sm", "compression_sm",
    "spatial_entropy_sm",
    "zone_inflow_sm", "zone_outflow_sm", "delta_count_sm",
]


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shared: tracker + feature extraction for a set of videos (fold-independent)
# ---------------------------------------------------------------------------

def extract_features_for_videos(
    detection_path: str,
    output_dir: str,
    cfg: dict,
    only_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Run tracker + feature extraction for videos in a detection JSON.
    Saves raw (un-normalized) feature CSVs to output_dir.

    only_keys: restrict to these top-level keys in the JSON (e.g. scene names).
               If None, process every key present.

    Returns (results, appearances, new_ids) — results is vid_id -> DataFrame,
    the two ints are for fragmentation reporting.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(detection_path) as f:
        data = _json.load(f)

    if only_keys is not None:
        missing = [k for k in only_keys if k not in data]
        if missing:
            raise KeyError(
                f"Requested keys {missing} not found in {detection_path}. "
                f"Available keys: {sorted(data.keys())}"
            )
        data = {k: data[k] for k in only_keys}

    tracker_cfg = cfg["tracker"]
    frame_cfg = cfg["frame"]
    zone_cfg = cfg["zones"]
    feat_cfg = cfg["features"]
    native_fps = frame_cfg["native_fps"]

    results = {}
    appearances = 0
    new_ids = 0

    for vid_id, vid_data in data.items():
        detections_per_frame = vid_data["detections"]

        frame_tracks, track_lengths = run_tracker(
            detections_per_frame,
            max_age=tracker_cfg["max_age"],
            min_iou=tracker_cfg["min_iou"],
            min_track_len=tracker_cfg["min_track_len"],
            max_dist=tracker_cfg["max_dist"],
        )

        prev_ids: set = set()
        for frame in frame_tracks:
            curr_ids = {t["track_id"] for t in frame}
            new_ids += len(curr_ids - prev_ids)
            appearances += len(curr_ids)
            prev_ids = curr_ids

        df = extract_features(
            frame_tracks,
            fps=native_fps,
            frame_w=frame_cfg["width"],
            frame_h=frame_cfg["height"],
            n_cols=zone_cfg["cols"],
            n_rows=zone_cfg["rows"],
            smoothing_window=feat_cfg["smoothing_window"],
        )

        df["vid_id"] = vid_id
        df["fps"] = native_fps

        out_path = os.path.join(output_dir, f"{vid_id}_features.csv")
        df.to_csv(out_path, index=False)
        results[vid_id] = df

        n_valid = sum(1 for length in track_lengths.values()
                      if length >= tracker_cfg["min_track_len"])
        print(f"  {vid_id}: {len(detections_per_frame)} frames, "
              f"{len(track_lengths)} raw tracks, {n_valid} kept")

    frag_rate = new_ids / appearances if appearances else 0.0
    print(f"  Fragmentation: {new_ids} new IDs / {appearances} appearances "
          f"= {frag_rate:.4f} ({frag_rate*100:.2f}%)")
    return results, appearances, new_ids


# Backward-compatible alias — old name, same behavior, used by file-based mode.
def process_split(detection_path, output_dir, cfg, only_keys=None):
    return extract_features_for_videos(detection_path, output_dir, cfg, only_keys)


# ---------------------------------------------------------------------------
# Prune + normalize for one train/test assignment (one "fold")
# ---------------------------------------------------------------------------

def prune_and_normalize_fold(
    train_results: Dict[str, pd.DataFrame],
    test_results: Dict[str, pd.DataFrame],
    cfg: dict,
    fold_dir: str,
) -> Tuple[List[str], "StandardScaler"]:
    """
    Fit pruning + scaler on train_results ONLY, apply to both train and test.
    Saves normalized CSVs, scaler, and schema under fold_dir.
    This is the one place cross-set leakage could happen — kept deliberately
    isolated so each fold (or the single split) gets its own independent fit.
    """
    os.makedirs(fold_dir, exist_ok=True)
    train_dir = os.path.join(fold_dir, "train_normalized")
    test_dir = os.path.join(fold_dir, "test_normalized")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_combined = pd.concat(train_results.values(), ignore_index=True)

    _, kept_cols, corr_report = prune_collinear(
        train_combined,
        feature_cols=ALL_FEATURE_COLS,
        threshold=cfg["features"]["corr_threshold"],
        anchor_cols=FORECAST_TARGETS,
    )
    if not corr_report.empty:
        corr_report.to_csv(os.path.join(fold_dir, "pruned_pairs.csv"), index=False)
    print(f"    Kept {len(kept_cols)} / {len(ALL_FEATURE_COLS)} features after pruning.")

    scaler = StandardScaler()
    scaler.fit(train_combined, kept_cols)
    scaler.save(os.path.join(fold_dir, "scaler.json"))

    for vid_id, df in train_results.items():
        scaler.transform(df).to_csv(
            os.path.join(train_dir, f"{vid_id}_features_norm.csv"), index=False
        )
    for vid_id, df in test_results.items():
        scaler.transform(df).to_csv(
            os.path.join(test_dir, f"{vid_id}_features_norm.csv"), index=False
        )

    schema = {
        "kept_feature_cols": kept_cols,
        "pruned_cols": [c for c in ALL_FEATURE_COLS if c not in kept_cols],
        "corr_threshold": cfg["features"]["corr_threshold"],
        "normalization": "StandardScaler (z-score), fit on this fold's train videos only",
        "train_videos": sorted(train_results.keys()),
        "test_videos": sorted(test_results.keys()),
    }
    with open(os.path.join(fold_dir, "feature_schema.json"), "w") as f:
        _json.dump(schema, f, indent=2)

    print(f"    Fold outputs saved to {fold_dir}")
    return kept_cols, scaler


# ---------------------------------------------------------------------------
# Mode 1: file-based split (original, e.g. UCSD)
# ---------------------------------------------------------------------------

def run_pipeline_file_based(cfg: dict, project_root: Path):
    print("=== TRAIN SPLIT ===")
    train_results, tr_app, tr_new = extract_features_for_videos(
        detection_path=str(project_root / cfg["data"]["detection_train"]),
        output_dir=str(project_root / cfg["data"]["output_features"] / "train"),
        cfg=cfg,
    )
    print("\n=== TEST SPLIT ===")
    test_results, te_app, te_new = extract_features_for_videos(
        detection_path=str(project_root / cfg["data"]["detection_test"]),
        output_dir=str(project_root / cfg["data"]["output_features"] / "test"),
        cfg=cfg,
    )

    total_app, total_new = tr_app + te_app, tr_new + te_new
    print(f"\n=== FRAGMENTATION ===")
    print(f"  Train : {tr_new}/{tr_app} = {tr_new/tr_app*100:.2f}%" if tr_app else "  Train : n/a")
    print(f"  Test  : {te_new}/{te_app} = {te_new/te_app*100:.2f}%" if te_app else "  Test  : n/a")
    print(f"  Combined: {total_new}/{total_app} = {total_new/total_app*100:.2f}%" if total_app else "  Combined: n/a")

    schema_dir = str(project_root / cfg["data"]["output_features"])
    print("\n=== FEATURE PRUNING + NORMALIZATION ===")
    kept_cols, scaler = prune_and_normalize_fold(train_results, test_results, cfg, schema_dir)
    print("Done.")
    return train_results, test_results, kept_cols, scaler


# ---------------------------------------------------------------------------
# Mode 2: fixed scene split (quick iteration, e.g. UMN scene1+2 -> scene3)
# ---------------------------------------------------------------------------

def run_pipeline_fixed_scene_split(cfg: dict, project_root: Path):
    split_cfg = cfg["split"]
    train_scenes = split_cfg["train_scenes"]
    test_scenes = split_cfg["test_scenes"]
    overlap = set(train_scenes) & set(test_scenes)
    if overlap:
        raise ValueError(f"train_scenes and test_scenes overlap: {overlap}")

    if cfg["data"]["detection_train"] != cfg["data"]["detection_test"]:
        raise ValueError(
            "Fixed scene-split mode expects detection_train == detection_test "
            "(the split happens by scene name within one file, not by separate files)."
        )
    detection_path = str(project_root / cfg["data"]["detection_train"])

    print(f"=== TRAIN SPLIT (scenes: {train_scenes}) ===")
    train_results, _, _ = extract_features_for_videos(
        detection_path, str(project_root / cfg["data"]["output_features"] / "train"),
        cfg, only_keys=train_scenes,
    )
    print(f"\n=== TEST SPLIT (scenes: {test_scenes}) ===")
    test_results, _, _ = extract_features_for_videos(
        detection_path, str(project_root / cfg["data"]["output_features"] / "test"),
        cfg, only_keys=test_scenes,
    )

    schema_dir = str(project_root / cfg["data"]["output_features"])
    print("\n=== FEATURE PRUNING + NORMALIZATION ===")
    print("  NOTE: single fixed split — not a statistically meaningful "
          "generalization test with only 3 independent UMN events. "
          "Use mode: 'loso' for the real evaluation.")
    kept_cols, scaler = prune_and_normalize_fold(train_results, test_results, cfg, schema_dir)
    print("Done.")
    return train_results, test_results, kept_cols, scaler


# ---------------------------------------------------------------------------
# Mode 3: leave-one-scene-out (UMN final evaluation)
# ---------------------------------------------------------------------------

def run_pipeline_loso(cfg: dict, project_root: Path):
    split_cfg = cfg["split"]
    scenes = split_cfg["scenes"]
    if len(scenes) < 2:
        raise ValueError("LOSO requires at least 2 scenes.")

    detection_path = str(project_root / cfg["data"]["detection_train"])
    raw_dir = str(project_root / cfg["data"]["output_features"] / "raw")

    print(f"=== EXTRACTING FEATURES ONCE FOR ALL SCENES: {scenes} ===")
    all_results, appearances, new_ids = extract_features_for_videos(
        detection_path, raw_dir, cfg, only_keys=scenes,
    )
    frag_rate = new_ids / appearances if appearances else 0.0
    print(f"  Fragmentation (all scenes): {new_ids}/{appearances} = {frag_rate*100:.2f}%")

    fold_root = project_root / cfg["data"]["output_features"] / "loso_folds"
    fold_summary = {}

    for held_out in scenes:
        train_scenes = [s for s in scenes if s != held_out]
        print(f"\n=== LOSO FOLD — held-out test: {held_out} | train: {train_scenes} ===")

        train_results = {s: all_results[s] for s in train_scenes}
        test_results = {held_out: all_results[held_out]}

        fold_dir = str(fold_root / f"test_{held_out}")
        kept_cols, scaler = prune_and_normalize_fold(train_results, test_results, cfg, fold_dir)
        fold_summary[held_out] = {
            "train_scenes": train_scenes,
            "kept_cols": kept_cols,
            "fold_dir": fold_dir,
        }

    print(f"\n=== LOSO COMPLETE — {len(scenes)} folds written to {fold_root} ===")
    for held_out, info in fold_summary.items():
        print(f"  test={held_out}: train={info['train_scenes']} -> {info['fold_dir']}")

    return all_results, fold_summary


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    project_root = Path(config_path).parent.parent
    os.makedirs(str(project_root / cfg["data"]["output_features"]), exist_ok=True)

    mode = cfg.get("split", {}).get("mode")

    if mode == "loso":
        return run_pipeline_loso(cfg, project_root)
    elif mode == "scene":
        return run_pipeline_fixed_scene_split(cfg, project_root)
    elif mode is None:
        return run_pipeline_file_based(cfg, project_root)
    else:
        raise ValueError(f"Unknown split mode: {mode!r}. Expected 'loso', 'scene', or omit 'split' entirely.")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    run_pipeline(cfg_path)