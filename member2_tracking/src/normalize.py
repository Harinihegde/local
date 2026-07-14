"""
Feature normalization (Section 3.1 — one unified pipeline for train, eval, real-time).

StandardScaler (z-score) is used:
  - Zero-mean, unit-variance per feature
  - Handles the density_sm / speed_mean magnitude gap (~0.0006 vs ~50)
  - More robust than min-max when accel_mean has outlier spikes

Scaler is fit on the train split only, saved to disk, and reloaded
identically for test evaluation and real-time inference.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import List, Tuple


class StandardScaler:
    """Fit-once, apply-everywhere z-score scaler. No sklearn dependency."""

    def __init__(self):
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None
        self.feature_cols_: List[str] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "StandardScaler":
        self.feature_cols_ = list(feature_cols)
        vals = df[feature_cols].values.astype(float)
        self.mean_ = vals.mean(axis=0)
        self.std_ = vals.std(axis=0)
        # Replace zero std with 1 to avoid division-by-zero on near-constant columns
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        vals = df[self.feature_cols_].values.astype(float)
        df[self.feature_cols_] = (vals - self.mean_) / self.std_
        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        self.fit(df, feature_cols)
        return self.transform(df)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols_,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        with open(path) as f:
            payload = json.load(f)
        scaler = cls()
        scaler.feature_cols_ = payload["feature_cols"]
        scaler.mean_ = np.array(payload["mean"])
        scaler.std_ = np.array(payload["std"])
        return scaler


def apply_normalization(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply a pre-fit scaler to a DataFrame. Safe to call from real-time inference."""
    return scaler.transform(df)