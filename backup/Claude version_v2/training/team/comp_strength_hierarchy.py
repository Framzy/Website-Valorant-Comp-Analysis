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

from training.team.config import (
    CS_SHRINKAGE_K,
    CS_RARE_PATTERN_SHARE_THRESHOLD,
    CS_RARE_PATTERN_DISCOUNT_CAP,
    RANDOM_STATE,
)

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


def _rarity_adjusted_anchor(fit: pd.DataFrame) -> float:
    """
    Anchor for Level 4 (Role Pattern, global) -- Tahap 6.

    Plain global_wr treats a role pattern nobody has ever tried the
    same as an average team: a neutral outcome. But at the pro level,
    teams that reach for a genuinely unusual role pattern are almost
    always doing so out of necessity (agent bans, a forced experiment)
    rather than because it is secretly strong, and the small amount of
    data that DOES exist for rare patterns confirms this: patterns
    covering <3% of maps each win at ~44%, vs ~50% for the six patterns
    that make up 97% of all professional play (a real, if modest, gap
    -- not something invented for this feature).

    So Level 4's anchor is the win rate of that "long tail" of rare
    patterns, not the flat dataset mean. This barely affects COMMON
    patterns (their own map count dominates the K-shrinkage regardless
    of the anchor) but means a pattern with ZERO recorded maps
    anywhere -- e.g. 5 Sentinel -- collapses to a modest, data-grounded
    discount instead of a neutral average. A pattern any team HAS
    played, however rarely, is scored from its own results as before;
    this only changes the anchor used when there is truly nothing else
    to go on.

    Recomputed from `fit` every call (same as global_wr), so it is
    automatically leak-free during training (out-of-fold for train
    rows, train-only for test rows) exactly like every other stat here.
    """
    pattern_maps = fit.groupby("Role Pattern")[MAPS].sum()
    share = pattern_maps / pattern_maps.sum()

    tail_patterns = share[share < CS_RARE_PATTERN_SHARE_THRESHOLD].index
    tail = fit[fit["Role Pattern"].isin(tail_patterns)]

    global_wr = fit[WINS].sum() / fit[MAPS].sum()

    if len(tail) == 0 or tail[MAPS].sum() == 0:
        return global_wr

    tail_wr = tail[WINS].sum() / tail[MAPS].sum()

    discount = max(tail_wr - global_wr, -CS_RARE_PATTERN_DISCOUNT_CAP)

    return global_wr + discount


def _chain(fit: pd.DataFrame, apply_df: pd.DataFrame) -> dict:
    """Run the full L4 -> L1 chain. Returns every level (for diagnostics)."""
    apply_df = apply_df.reset_index(drop=True)
    global_wr = fit[WINS].sum() / fit[MAPS].sum()
    anchor = _rarity_adjusted_anchor(fit)

    out = {
        "global_wr": np.full(len(apply_df), global_wr),
        "rarity_adjusted_anchor": np.full(len(apply_df), anchor),
    }
    prior = out["rarity_adjusted_anchor"]
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


_SERVING_CACHE: dict = {}


def _serving_stats(df: pd.DataFrame) -> dict:
    """
    Aggregates needed to score ONE composition at serving time, computed
    once per dataset and reused (the previous version re-ran every groupby
    on every prediction, twice per request, which made sweeps very slow).
    """
    key = (id(df), len(df), int(df[MAPS].sum()))
    cached = _SERVING_CACHE.get("entry")
    if cached is not None and cached["key"] == key:
        return cached

    def agg(keys):
        g = df.groupby(keys).agg(wins=(WINS, "sum"), maps=(MAPS, "sum"))
        return {
            (k if isinstance(k, tuple) else (k,)): (int(w), int(m))
            for k, w, m in zip(g.index, g["wins"], g["maps"])
        }

    pattern_maps = df.groupby("Role Pattern")[MAPS].sum()

    entry = {
        "key": key,
        "global_wr": float(df[WINS].sum() / df[MAPS].sum()),
        "anchor": float(_rarity_adjusted_anchor(df)),
        "levels": {name: agg(keys) for name, keys in _LEVELS},
        "pattern_maps": {p: int(m) for p, m in pattern_maps.items()},
        "total_maps": int(pattern_maps.sum()),
        "team_totals": {t: int(m) for t, m in df.groupby("Team")[MAPS].sum().items()},
    }
    _SERVING_CACHE["entry"] = entry
    return entry


def strength_for_new_row(df: pd.DataFrame, team: str, map_name: str, role_pattern: str, composition_key: str) -> dict:
    """
    Serving-time lookup for ANY composition (seen or brand-new), including
    ones with a Role Pattern nobody in the dataset has ever played.

    Returns every level plus evidence counts, so the caller can tell an
    exact-comp match apart from a full backoff to the team's own average.
    Same math as the vectorised _chain(), evaluated for a single row from
    pre-aggregated lookups (see _serving_stats).
    """
    st = _serving_stats(df)
    values = {"Team": team, "Map": map_name, "Role Pattern": role_pattern, "Composition Key": composition_key}

    prior = st["anchor"]
    levels = {"global_wr": st["global_wr"], "rarity_adjusted_anchor": st["anchor"]}
    played = {}

    for name, keys in _LEVELS:
        wins, maps = st["levels"][name].get(tuple(values[k] for k in keys), (0, 0))
        k_shrink = CS_SHRINKAGE_K[name]
        prior = (wins + k_shrink * prior) / (maps + k_shrink)
        levels[name] = prior
        played[name] = maps

    levels["final"] = prior

    evidence = {
        "global_pattern_played": played["global_pattern"],
        "team_pattern_played": played["team_pattern"],
        "team_map_pattern_played": played["team_map_pattern"],
        "exact_played": played["exact"],
    }

    # How much this Role Pattern is actually chosen: across the whole pro
    # scene (global) and by this team. Pure counts, no model involved.
    global_maps = evidence["global_pattern_played"]
    team_maps = evidence["team_pattern_played"]
    team_total = st["team_totals"].get(team, 0)
    total_maps = st["total_maps"]

    pattern_adoption = {
        "global_maps": global_maps,
        "global_share": (global_maps / total_maps) if total_maps else 0.0,
        "global_rank": (
            sum(1 for m in st["pattern_maps"].values() if m > global_maps) + 1
            if global_maps > 0 else None
        ),
        "n_patterns": sum(1 for m in st["pattern_maps"].values() if m > 0),
        "team_maps": team_maps,
        "team_share": (team_maps / team_total) if team_total else 0.0,
    }

    return {
        "composition_strength": float(round(prior * 100, 2)),
        "levels": {k: float(round(v, 4)) for k, v in levels.items()},
        "evidence": evidence,
        "role_pattern_never_observed": global_maps == 0,
        "pattern_adoption": pattern_adoption,
    }
