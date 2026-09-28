"""
Hierarchical, evidence-aware Composition Strength (Tahap 4)
=============================================================

Composition Strength used to be a pure USAGE score (how often this
composition/pattern/agent combo was picked). That let a composition
played exactly once, in a context with no alternatives, score as high
as one played 14 times (both get usage_ratio = 1.0).

This module recomputes it as historical PERFORMANCE (win rate), backed
off through a chain of progressively broader contexts so that thin
evidence at one level borrows strength from the level above it instead
of being reported as if it were solid:

    L1  Team + Map + exact composition   (pooled across years)
    L2  Team + Map + Role Pattern
    L3  Team + Role Pattern
    L4  Role Pattern, global (all teams)
    L5  Team Overall WR (already computed elsewhere; the ultimate floor)

    value(level) = (wins + K * value(level_above)) / (maps + K)

Year is intentionally NOT part of any key: a team's identity, a map's
identity and a playstyle's identity persist across years, and pooling
across years is the only way most of these groups get enough maps to
say anything. This mirrors how Team Overall WR / Team Map WR were
fixed in Tahap 1.

Two entry points:
    build_full_dataset_strength(df)   -> used by build_team_dataset.py
                                          to (re)compute the CSV column,
                                          and by the backend at serving
                                          time via the same aggregates.
    honest_strength(df, train_idx,    -> used by training only, so the
                     test_idx)           row's own outcome never leaks
                                          into its own feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from training.team.config import CS_SHRINKAGE_K, RANDOM_STATE

WINS = "Total Wins By Map"
MAPS = "Total Maps Played"

# Ordered coarse -> fine is used for the *global_wr* anchor; the chain
# itself is applied fine -> coarse below.
_LEVELS = (
    ("global_pattern", ["Role Pattern"]),
    ("team_pattern", ["Team", "Role Pattern"]),
    ("team_map_pattern", ["Team", "Map", "Role Pattern"]),
    ("exact", ["Team", "Map", "Composition Key"]),
)


def _shrink(fit: pd.DataFrame, apply_keys: pd.DataFrame, keys, prior: np.ndarray, K: float) -> np.ndarray:
    """(wins + K*prior) / (maps + K), aggregated over `keys` from `fit`."""
    stats = fit.groupby(keys).agg(wins=(WINS, "sum"), maps=(MAPS, "sum"))
    merged = apply_keys[keys].merge(stats, left_on=keys, right_index=True, how="left")
    wins = merged["wins"].fillna(0).to_numpy()
    maps = merged["maps"].fillna(0).to_numpy()
    return (wins + K * prior) / (maps + K)


def _chain(fit: pd.DataFrame, apply_df: pd.DataFrame) -> dict:
    """Run the full L4 -> L1 chain. Returns every level (for diagnostics)."""
    apply_df = apply_df.reset_index(drop=True)
    global_wr = fit[WINS].sum() / fit[MAPS].sum()

    out = {"global_wr": np.full(len(apply_df), global_wr)}
    prior = out["global_wr"]
    for name, keys in _LEVELS:
        prior = _shrink(fit, apply_df, keys, prior, CS_SHRINKAGE_K[name])
        out[name] = prior
    out["final"] = prior  # == out["exact"], kept as an explicit alias
    return out


def build_full_dataset_strength(df: pd.DataFrame) -> pd.Series:
    """
    Full-data (leaky-by-design) Composition Strength for the CSV / serving
    lookup table. Training must NOT use this directly (see honest_strength).
    """
    levels = _chain(df, df)
    return pd.Series((levels["final"] * 100).round(2), index=df.index, name="Composition Strength")


def honest_strength(df: pd.DataFrame, train_idx, test_idx, n_splits: int = 5):
    """
    Leak-free version for training: out-of-fold for train rows, train-only
    for test rows. Returns (strength_train, strength_test) as 0-100 arrays,
    aligned to train_idx / test_idx order.
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    strength_train = np.zeros(len(train_idx))
    kf = KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_pos, hold_pos in kf.split(train_idx):
        levels = _chain(df.loc[train_idx[fit_pos]], df.loc[train_idx[hold_pos]])
        strength_train[hold_pos] = levels["final"]

    levels = _chain(df.loc[train_idx], df.loc[test_idx])
    strength_test = levels["final"]

    return np.round(strength_train * 100, 2), np.round(strength_test * 100, 2)


def strength_for_new_row(df: pd.DataFrame, team: str, map_name: str, role_pattern: str, composition_key: str) -> dict:
    """
    Serving-time lookup for ANY composition (seen or brand-new), including
    ones with a Role Pattern nobody in the dataset has ever played.

    Returns every level plus evidence counts, so the caller can tell an
    exact-comp match apart from a full backoff to the team's own average.
    """
    row = pd.DataFrame([{"Team": team, "Map": map_name, "Role Pattern": role_pattern, "Composition Key": composition_key}])
    levels = _chain(df, row)

    def maps_played(keys):
        sub = df
        for k, v in zip(keys, [team, map_name, role_pattern, composition_key][: len(keys)]):
            pass
        return None

    # Evidence counts per level (maps played), independent of the shrink math.
    def count(keys, values):
        mask = np.ones(len(df), dtype=bool)
        for k, v in zip(keys, values):
            mask &= (df[k] == v)
        return int(df.loc[mask, MAPS].sum())

    evidence = {
        "global_pattern_played": count(["Role Pattern"], [role_pattern]),
        "team_pattern_played": count(["Team", "Role Pattern"], [team, role_pattern]),
        "team_map_pattern_played": count(["Team", "Map", "Role Pattern"], [team, map_name, role_pattern]),
        "exact_played": count(["Team", "Map", "Composition Key"], [team, map_name, composition_key]),
    }

    return {
        "composition_strength": float(round(levels["final"][0] * 100, 2)),
        "levels": {k: float(round(v[0], 4)) for k, v in levels.items()},
        "evidence": evidence,
        "role_pattern_never_observed": evidence["global_pattern_played"] == 0,
    }
