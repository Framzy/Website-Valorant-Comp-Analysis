"""
Shared Team Feature Encoder Contract Test
=========================================

This test verifies that the shared encoder matches the
existing Team V2 artifacts.

Run from project root:

    python -m training.team.analysis.test.test_feature_encoder

The test imports the shared encoder through the project's
shared package, so copy feature_encoder.py into:

    shared/team/feature_encoder.py
"""

import ast

import joblib
import pandas as pd

from training.team.config import (
    MODEL_TEAM_DIR,
    TEAM_DATASET_PATH,
)

from training.team.feature_encoder import (
    EXPECTED_FEATURE_COUNT,
    ROLE_COUNT_FEATURE_NAMES,
    ROLE_PATTERN_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    build_feature_names,
    encode_features,
)


def main():
    print("=" * 60)
    print("SHARED TEAM FEATURE ENCODER TEST")
    print("=" * 60)

    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    encoders = joblib.load(
        MODEL_TEAM_DIR / "encoders.joblib"
    )

    stored_feature_names = list(
        joblib.load(
            MODEL_TEAM_DIR / "feature_names.joblib"
        )
    )

    # ========================================================
    # FEATURE NAME CONTRACT
    # ========================================================

    generated_feature_names = (
        build_feature_names(encoders)
    )

    assert len(generated_feature_names) == (
        EXPECTED_FEATURE_COUNT
    )

    print(
        f"[PASS] Feature names: "
        f"{len(generated_feature_names)}"
    )

    assert generated_feature_names == (
        stored_feature_names
    )

    print(
        "[PASS] Feature names match "
        "stored artifact"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv(
        TEAM_DATASET_PATH
    )

    df["Agent"] = df["Agent"].apply(
        lambda value:
            ast.literal_eval(value)
            if isinstance(value, str)
            else value
    )

    # ========================================================
    # ENCODE ONE REAL DATASET ROW
    # ========================================================

    X = df.iloc[[0]].copy()

    encoded = encode_features(
        X,
        encoders,
    )

    assert encoded.shape == (
        1,
        EXPECTED_FEATURE_COUNT,
    )

    print(
        "[PASS] Encoded shape: "
        "(1, 116)"
    )

    # ========================================================
    # CATEGORICAL CONTRACT
    # ========================================================

    team_count = len(
        encoders["team"].categories_[0]
    )

    map_count = len(
        encoders["map"].categories_[0]
    )

    year_count = len(
        encoders["year"].categories_[0]
    )

    agent_count = len(
        encoders["agent"].classes_
    )

    assert team_count == 57
    assert map_count == 12
    assert year_count == 3
    assert agent_count == 29

    print("[PASS] Team : 57")
    print("[PASS] Map  : 12")
    print("[PASS] Year : 3")
    print("[PASS] Agent: 29")

    # ========================================================
    # ROLE COUNT CONTRACT
    # ========================================================

    role_count_start = (
        team_count
        + map_count
        + year_count
        + agent_count
    )

    role_count_end = (
        role_count_start + 4
    )

    actual_role_count_names = (
        generated_feature_names[
            role_count_start:role_count_end
        ]
    )

    assert actual_role_count_names == (
        ROLE_COUNT_FEATURE_NAMES
    )

    print(
        "[PASS] Role count feature order"
    )

    # ========================================================
    # ROLE PATTERN CONTRACT
    # ========================================================

    role_pattern_start = role_count_end
    role_pattern_end = (
        role_pattern_start + 8
    )

    actual_role_pattern_names = (
        generated_feature_names[
            role_pattern_start:role_pattern_end
        ]
    )

    assert actual_role_pattern_names == (
        ROLE_PATTERN_FEATURE_NAMES
    )

    print(
        "[PASS] Role pattern feature order"
    )

    # ========================================================
    # NUMERIC CONTRACT
    # ========================================================

    actual_numeric_names = (
        generated_feature_names[-3:]
    )

    assert actual_numeric_names == (
        NUMERIC_FEATURE_NAMES
    )

    print(
        "[PASS] Numeric feature order"
    )

    # ========================================================
    # FINAL TOTAL
    # ========================================================

    total = (
        team_count
        + map_count
        + year_count
        + agent_count
        + len(ROLE_COUNT_FEATURE_NAMES)
        + len(ROLE_PATTERN_FEATURE_NAMES)
        + len(NUMERIC_FEATURE_NAMES)
    )

    assert total == EXPECTED_FEATURE_COUNT

    print(
        f"[PASS] Feature total: {total}"
    )

    print()
    print(
        "ALL SHARED ENCODER TESTS PASSED"
    )


if __name__ == "__main__":
    main()
