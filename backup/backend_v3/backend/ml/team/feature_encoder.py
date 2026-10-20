"""
Shared Team Feature Encoder
===========================

Single source of truth for the Team V2 feature contract.

This module is shared by:
- training_v2
- production/backend

IMPORTANT:
Input column names and model feature names are intentionally kept
as separate constants. They represent different things.

Feature matrix order:
    1. Team          : 57
    2. Map           : 12
    3. Year          : 3
    4. Agent         : 29
    5. Role Count    : 4
    6. Role Pattern  : 8
    7. Numeric       : 3
    --------------------------------
    Total            : 116
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONTRACT
# ============================================================

from backend.constants import (
    EXPECTED_FEATURE_COUNT,
)


# ============================================================
# INPUT DATAFRAME COLUMNS
# ============================================================
# These names MUST match the Team V2 dataset / prepared
# inference DataFrame.

ROLE_COUNT_COLUMNS = [
    "Controller Count",
    "Duelist Count",
    "Initiator Count",
    "Sentinel Count",
]

NUMERIC_COLUMNS = [
    "Team Overall WR",
    "Team Map WR",
    "Composition Strength",
]


# ============================================================
# MODEL FEATURE NAMES
# ============================================================
# These names MUST match feature_names.joblib.
# They are NOT DataFrame input column names.

ROLE_COUNT_FEATURE_NAMES = [
    "role_controller",
    "role_duelist",
    "role_initiator",
    "role_sentinel",
]

ROLE_PATTERN_FEATURE_NAMES = [
    "has_controller",
    "has_duelist",
    "has_initiator",
    "has_sentinel",
    "double_controller",
    "double_initiator",
    "double_duelist",
    "double_sentinel",
]

NUMERIC_FEATURE_NAMES = [
    "team_overall_wr",
    "team_map_wr",
    "composition_strength",
]


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_encoder_contract(encoders: dict) -> None:
    """
    Validate that the required categorical encoders exist.
    """

    required = {
        "team",
        "map",
        "year",
        "agent",
    }

    missing = required - set(encoders.keys())

    if missing:
        raise ValueError(
            f"Missing encoders: {sorted(missing)}"
        )


def validate_input_columns(X: pd.DataFrame) -> None:
    """
    Validate columns required by the encoder.

    These are INPUT column names, not model feature names.
    """

    required = [
        "Team",
        "Map",
        "Year",
        "Agent",
        *ROLE_COUNT_COLUMNS,
        *NUMERIC_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing input columns: {missing}"
        )


# ============================================================
# ROLE FEATURES
# ============================================================

def build_role_features(
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Build four role-count features.

    INPUT columns:
        Controller Count
        Duelist Count
        Initiator Count
        Sentinel Count

    OUTPUT order:
        role_controller
        role_duelist
        role_initiator
        role_sentinel
    """

    role_feature = X[
        ROLE_COUNT_COLUMNS
    ].to_numpy(dtype=int)

    if role_feature.ndim != 2:
        raise ValueError(
            "Role feature must be a 2-dimensional array."
        )

    if role_feature.shape[1] != 4:
        raise ValueError(
            f"Invalid role feature shape: "
            f"{role_feature.shape}. "
            f"Expected 4 columns."
        )

    return role_feature


# ============================================================
# ROLE PATTERN FEATURES
# ============================================================

def build_role_pattern_features(
    role_feature: np.ndarray,
) -> np.ndarray:
    """
    Build eight binary role-pattern features.

    INPUT role order:
        Controller
        Duelist
        Initiator
        Sentinel

    OUTPUT order:
        has_controller
        has_duelist
        has_initiator
        has_sentinel
        double_controller
        double_initiator
        double_duelist
        double_sentinel
    """

    if role_feature.ndim != 2:
        raise ValueError(
            "Role feature must be 2-dimensional."
        )

    if role_feature.shape[1] != 4:
        raise ValueError(
            f"Invalid role feature shape: "
            f"{role_feature.shape}. "
            f"Expected (n, 4)."
        )

    controller = role_feature[:, 0]
    duelist = role_feature[:, 1]
    initiator = role_feature[:, 2]
    sentinel = role_feature[:, 3]

    role_pattern_feature = np.column_stack([
        (controller >= 1).astype(int),
        (duelist >= 1).astype(int),
        (initiator >= 1).astype(int),
        (sentinel >= 1).astype(int),

        (controller >= 2).astype(int),
        (initiator >= 2).astype(int),
        (duelist >= 2).astype(int),
        (sentinel >= 2).astype(int),
    ])

    if role_pattern_feature.shape[1] != 8:
        raise ValueError(
            "Role pattern feature count must be 8."
        )

    return role_pattern_feature


# ============================================================
# MAIN ENCODER
# ============================================================

def encode_features(
    X: pd.DataFrame,
    encoders: dict,
) -> np.ndarray:
    """
    Encode Team V2 data into the fixed 116-feature matrix.

    Feature matrix order:
        Team
        Map
        Year
        Agent
        Role Count
        Role Pattern
        Numeric

    This order MUST NOT change without retraining the model.
    """

    validate_input_columns(X)
    validate_encoder_contract(encoders)

    # --------------------------------------------------------
    # 1. Team
    # --------------------------------------------------------

    team_feature = encoders["team"].transform(
        X[["Team"]]
    )

    # --------------------------------------------------------
    # 2. Map
    # --------------------------------------------------------

    map_feature = encoders["map"].transform(
        X[["Map"]]
    )

    # --------------------------------------------------------
    # 3. Year
    # --------------------------------------------------------

    year_feature = encoders["year"].transform(
        X[["Year"]]
    )

    # --------------------------------------------------------
    # 4. Agent
    # --------------------------------------------------------

    agent_feature = encoders["agent"].transform(
        X["Agent"]
    )

    # --------------------------------------------------------
    # 5. Role Count
    # --------------------------------------------------------

    role_feature = build_role_features(X)

    # --------------------------------------------------------
    # 6. Role Pattern
    # --------------------------------------------------------

    role_pattern_feature = (
        build_role_pattern_features(
            role_feature
        )
    )

    # --------------------------------------------------------
    # 7. Numeric
    # --------------------------------------------------------

    numeric_feature = X[
        NUMERIC_COLUMNS
    ].to_numpy(dtype=float)

    if np.isnan(numeric_feature).any():
        raise ValueError(
            "Numeric feature contains NaN values."
        )

    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    X_encoded = np.concatenate(
        [
            team_feature,
            map_feature,
            year_feature,
            agent_feature,
            role_feature,
            role_pattern_feature,
            numeric_feature,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # FINAL CONTRACT VALIDATION
    # --------------------------------------------------------

    if X_encoded.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Invalid encoded feature count: "
            f"{X_encoded.shape[1]}. "
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )

    return X_encoded


# ============================================================
# FEATURE NAME BUILDER
# ============================================================

def build_feature_names(
    encoders: dict,
) -> list[str]:
    """
    Build model feature names in exactly the same order
    as encode_features().
    """

    validate_encoder_contract(encoders)

    feature_names = []

    # --------------------------------------------------------
    # 1. Team
    # --------------------------------------------------------

    feature_names.extend(
        f"team_{value}"
        for value in encoders[
            "team"
        ].categories_[0]
    )

    # --------------------------------------------------------
    # 2. Map
    # --------------------------------------------------------

    feature_names.extend(
        f"map_{value}"
        for value in encoders[
            "map"
        ].categories_[0]
    )

    # --------------------------------------------------------
    # 3. Year
    # --------------------------------------------------------

    feature_names.extend(
        f"meta_{value}"
        for value in encoders[
            "year"
        ].categories_[0]
    )

    # --------------------------------------------------------
    # 4. Agent
    # --------------------------------------------------------

    feature_names.extend(
        f"agent_{value}"
        for value in encoders[
            "agent"
        ].classes_
    )

    # --------------------------------------------------------
    # 5. Role Count
    # --------------------------------------------------------

    feature_names.extend(
        ROLE_COUNT_FEATURE_NAMES
    )

    # --------------------------------------------------------
    # 6. Role Pattern
    # --------------------------------------------------------

    feature_names.extend(
        ROLE_PATTERN_FEATURE_NAMES
    )

    # --------------------------------------------------------
    # 7. Numeric
    # --------------------------------------------------------

    feature_names.extend(
        NUMERIC_FEATURE_NAMES
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Invalid feature name count: "
            f"{len(feature_names)}. "
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )

    return feature_names
