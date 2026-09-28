"""
Leak-free Team WR features (training-side only)
===============================================

The dataset CSV stores FULL-DATA shrunk statistics in
`Team Overall WR` / `Team Map WR`. That is exactly what the backend
needs at serving time (a lookup), but for TRAINING those values contain
each row's own outcome.

This module recomputes the same two features without self-leakage:

    train rows : out-of-fold  (row's own fold is excluded)
    test rows  : computed from TRAIN rows only

Formula is identical to build_team_dataset.py:
    (wins + K * global_wr) / (maps + K)
Column order in the encoded matrix is unchanged (116-feature contract).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.team.config import WR_SHRINKAGE_K, RANDOM_STATE

WR_KEYS = (["Team"], ["Team", "Map"])          # -> [overall, map]
WR_FEATURE_NAMES = ["team_overall_wr", "team_map_wr"]


def _wr_from(fit: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    global_wr = (
        fit["Total Wins By Map"].sum() / fit["Total Maps Played"].sum()
    )
    out = np.zeros((len(apply), len(WR_KEYS)))
    for j, keys in enumerate(WR_KEYS):
        stats = fit.groupby(keys).agg(
            wins=("Total Wins By Map", "sum"),
            maps=("Total Maps Played", "sum"),
        )
        merged = apply[keys].merge(
            stats, left_on=keys, right_index=True, how="left"
        )
        wins = merged["wins"].fillna(0).to_numpy()
        maps = merged["maps"].fillna(0).to_numpy()
        out[:, j] = (
            (wins + WR_SHRINKAGE_K * global_wr) / (maps + WR_SHRINKAGE_K)
        )
    return out


def honest_wr(df, train_idx, test_idx, n_splits=5):
    """Return (wr_train_oof, wr_test) as (n, 2) arrays."""
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    wr_train = np.zeros((len(train_idx), len(WR_KEYS)))
    kf = KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_pos, hold_pos in kf.split(train_idx):
        wr_train[hold_pos] = _wr_from(
            df.loc[train_idx[fit_pos]], df.loc[train_idx[hold_pos]]
        )

    wr_test = _wr_from(df.loc[train_idx], df.loc[test_idx])
    return wr_train, wr_test


def apply_honest_wr(train_data, df, feature_names):
    """Replace the two WR columns of X_train / X_test with honest values."""
    cols = [feature_names.index(n) for n in WR_FEATURE_NAMES]
    wr_train, wr_test = honest_wr(
        df, train_data["y_train"].index, train_data["y_test"].index
    )
    X_train = train_data["X_train"].copy()
    X_test = train_data["X_test"].copy()
    X_train[:, cols] = wr_train
    X_test[:, cols] = wr_test
    return {**train_data, "X_train": X_train, "X_test": X_test}
