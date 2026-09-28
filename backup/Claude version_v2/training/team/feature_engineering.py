"""
Feature Engineering V2
======================

Training-side feature engineering.

The actual feature encoding contract is centralized in:
    shared.team.feature_encoder

This module keeps training-specific responsibilities:
- dataset loading
- dataset validation
- feature/target split
- fitting encoders
- training pipeline orchestration
- artifact debugging/saving
"""

import ast
import joblib

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from training.team.config import (
    MODEL_DIR,
    TEAM_DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Shared feature contract
# ---------------------------------------------------------------------------

from training.team.feature_encoder import (
    EXPECTED_FEATURE_COUNT,
    build_role_features,
    build_role_pattern_features,
    encode_features,
    build_feature_names,
)


def load_dataset() -> pd.DataFrame:
    """
    Load normalized training dataset.
    """

    print("=" * 60)
    print("LOAD TRAINING DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)
    df["Agent"] = df["Agent"].apply(ast.literal_eval)

    print(f"[INFO] Shape : {df.shape}")

    return df


def validate_dataset(df: pd.DataFrame):
    """
    Validate normalized dataset.
    """

    print("\n" + "=" * 60)
    print("VALIDATE DATASET")
    print("=" * 60)

    invalid = df[df["Agent"].apply(len) != 5]

    print(f"[INFO] Invalid Composition : {len(invalid)}")

    if len(invalid):
        raise ValueError(
            "Dataset still contains composition != 5 agents."
        )

    print("[INFO] Validation Passed")


def split_feature_target(df: pd.DataFrame):
    """
    Split features and target.
    """

    print("\n" + "=" * 60)
    print("SPLIT FEATURE & TARGET")
    print("=" * 60)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"[INFO] Feature Shape : {X.shape}")
    print(f"[INFO] Target Shape  : {y.shape}")

    return X, y


def build_encoders(X: pd.DataFrame):
    """
    Fit categorical encoders from the training dataset.

    This remains training-specific because production must load the
    persisted encoders rather than fit new ones.
    """

    print("\n" + "=" * 60)
    print("BUILD ENCODERS")
    print("=" * 60)

    team_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    map_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    year_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    agent_encoder = MultiLabelBinarizer()

    team_encoder.fit(X[["Team"]])
    map_encoder.fit(X[["Map"]])
    year_encoder.fit(X[["Year"]])
    agent_encoder.fit(X["Agent"])

    print(f"[INFO] Team Category : {len(team_encoder.categories_[0])}")
    print(f"[INFO] Map Category  : {len(map_encoder.categories_[0])}")
    print(f"[INFO] Meta Category : {len(year_encoder.categories_[0])}")
    print(f"[INFO] Agent Count   : {len(agent_encoder.classes_)}")

    return {
        "team": team_encoder,
        "map": map_encoder,
        "year": year_encoder,
        "agent": agent_encoder,
    }


def validate_feature_names(
    X_encoded,
    feature_names,
):
    """
    Validate encoded feature count against feature names.
    """

    print("\n" + "=" * 60)
    print("VALIDATE FEATURE NAMES")
    print("=" * 60)

    if X_encoded.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch.\n"
            f"Encoded : {X_encoded.shape[1]}\n"
            f"Names   : {len(feature_names)}"
        )

    if X_encoded.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature contract.\n"
            f"Encoded : {X_encoded.shape[1]}\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}"
        )

    print("[INFO] Validation Passed")


def build_feature_pipeline(df: pd.DataFrame) -> dict:
    """
    Complete feature engineering pipeline.
    """

    validate_dataset(df)

    X, y = split_feature_target(df)

    encoders = build_encoders(X)

    X_encoded = encode_features(
        X,
        encoders,
    )

    feature_names = build_feature_names(
        encoders
    )

    validate_feature_names(
        X_encoded,
        feature_names,
    )

    return {
        "X": X_encoded,
        "y": y,
        "encoders": encoders,
        "feature_names": feature_names,
    }


def debugging():
    """
    Debugging helper for inspecting encoder artifacts and feature output.
    """

    df = load_dataset()

    validate_dataset(df)

    print()

    print("=" * 60)
    print("AGENT TYPE")
    print("=" * 60)

    print(type(df.iloc[0]["Agent"]))
    print(df.iloc[0]["Agent"])

    X, y = split_feature_target(df)

    encoders = build_encoders(X)

    print("\n" + "=" * 60)
    print("SAVE ENCODERS")
    print("=" * 60)

    joblib.dump(
        encoders,
        MODEL_DIR / "encoders.joblib",
    )

    print(f"[INFO] Saved : {MODEL_DIR / 'encoders.joblib'}")

    X_encoded = encode_features(
        X,
        encoders,
    )

    print()

    print("=" * 60)
    print("ENCODED SAMPLE")
    print("=" * 60)

    print(X_encoded[:5])

    feature_names = build_feature_names(
        encoders
    )

    validate_feature_names(
        X_encoded,
        feature_names,
    )

    print()

    print("=" * 60)
    print("FEATURE NAME SAMPLE")
    print("=" * 60)

    for feature in feature_names[:10]:
        print(feature)

    print()

    print("=" * 60)
    print("FEATURE SAMPLE")
    print("=" * 60)

    print(X.head())

    print()

    print("=" * 60)
    print("TARGET SAMPLE")
    print("=" * 60)

    print(y.head())


def main():
    df = load_dataset()

    pipeline = build_feature_pipeline(df)

    print()

    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(f"X Shape : {pipeline['X'].shape}")
    print(f"y Shape : {pipeline['y'].shape}")
    print(f"Feature : {len(pipeline['feature_names'])}")

    print()

    print("=" * 60)
    print("TARGET SUMMARY")
    print("=" * 60)

    print(pipeline["y"].describe())


if __name__ == "__main__":
    main()
