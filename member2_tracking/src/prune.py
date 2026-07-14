"""
Collinear feature pruning (Section 3.7).
Drop one of each highly-correlated pair before handing off to Member 3.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def prune_collinear(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95,
    anchor_cols: List[str] = None,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Given a DataFrame and a list of candidate feature columns, drop one column
    from each pair with |Pearson r| >= threshold.

    anchor_cols: columns that must be retained regardless of correlation
                 (e.g. required forecasting targets). When an anchor is the
                 "other" in a correlated pair, the non-anchor is dropped instead.

    Returns:
      pruned_df: DataFrame with redundant columns removed
      kept_cols: list of retained feature column names
      corr_report: DataFrame of dropped pairs with their correlation values
    """
    anchors = set(anchor_cols or [])

    # Drop near-constant columns first (std ~ 0 means correlation is undefined)
    stds = df[feature_cols].std()
    near_const = [c for c in feature_cols if stds[c] < 1e-8 and c not in anchors]
    candidates = [c for c in feature_cols if c not in near_const]

    corr = df[candidates].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped = set()
    drop_log = []

    for col in upper.columns:
        if col in dropped:
            continue
        correlated = upper[col][upper[col] >= threshold].index.tolist()
        for other in correlated:
            if other in dropped:
                continue
            # Never drop an anchor; if col is anchor, drop other instead
            if other in anchors:
                # other is protected — drop col if it isn't also an anchor
                if col not in anchors:
                    dropped.add(col)
                    drop_log.append({
                        "dropped": col,
                        "kept": other,
                        "abs_corr": float(upper.loc[other, col]),
                        "note": "anchor protected",
                    })
                # if both are anchors, keep both (don't drop either)
            else:
                dropped.add(other)
                drop_log.append({
                    "dropped": other,
                    "kept": col,
                    "abs_corr": float(upper.loc[other, col]),
                    "note": "",
                })

    all_dropped = set(near_const) | dropped
    kept_cols = [c for c in feature_cols if c not in all_dropped]
    corr_report = pd.DataFrame(drop_log)
    pruned_df = df.drop(columns=list(all_dropped), errors="ignore")

    return pruned_df, kept_cols, corr_report