"""
Validate consistency between direct dataset encoding and inference encoding.

Purpose:
    Use an existing historical composition as a golden case.

    Path A:
        Dataset row
        -> feature encoding

    Path B:
        Application-style input
        -> historical lookup
        -> inference row
        -> feature encoding

    Both paths must produce the same 116-feature vector and the
    same model prediction.

This is a validation script, not the production service.
"""

import ast
import numpy as np
import joblib
import pandas as pd

from training.team.config import (
    TEAM_DATASET_PATH,
    MODEL_TEAM_DIR,
)
from training.team.feature_engineering import (
    encode_features,
)


# ============================================================
# GOLDEN TEST INPUT
# ============================================================

TEAM = "100 Thieves"
MAP_NAME = "Abyss"
YEAR = 2024

AGENTS = [
    "cypher",
    "gekko",
    "jett",
    "omen",
    "sova",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_agents(agents):
    return sorted(
        agent.strip().lower()
        for agent in agents
    )


def load_dataset():
    print("=" * 60)
    print("LOAD TEAM DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)
    df["Agent"] = df["Agent"].apply(ast.literal_eval)

    print(f"[INFO] Dataset shape : {df.shape}")

    return df


def load_artifacts():
    print("\n" + "=" * 60)
    print("LOAD INFERENCE ARTIFACTS")
    print("=" * 60)

    model = joblib.load(
        MODEL_TEAM_DIR / "team_model_v2.joblib"
    )

    encoders = joblib.load(
        MODEL_TEAM_DIR / "encoders.joblib"
    )

    feature_names = joblib.load(
        MODEL_TEAM_DIR / "feature_names.joblib"
    )

    print(f"[INFO] Model features : {model.n_features_in_}")
    print(f"[INFO] Feature names  : {len(feature_names)}")

    return model, encoders, feature_names


def find_historical_row(
    df,
    team,
    map_name,
    year,
    agents,
):
    target_agents = normalize_agents(agents)

    composition_match = df["Agent"].apply(
        lambda composition:
            normalize_agents(composition) == target_agents
    )

    matches = df[
        (df["Team"] == team)
        & (df["Map"] == map_name)
        & (df["Year"] == year)
        & composition_match
    ]

    if matches.empty:
        return None

    if len(matches) > 1:
        raise ValueError(
            "Multiple historical rows found for "
            "the same Team + Map + Year + Composition."
        )

    return matches.iloc[0]


def build_inference_row(
    historical_row,
    team,
    map_name,
    year,
    agents,
):
    row = historical_row.copy()

    row["Team"] = team
    row["Map"] = map_name
    row["Year"] = year
    row["Agent"] = agents

    return pd.DataFrame([row])


def compare_feature_vectors(
    direct_features,
    inference_features,
    feature_names,
):
    print("\n" + "=" * 60)
    print("COMPARE FEATURE VECTORS")
    print("=" * 60)

    if direct_features.shape != inference_features.shape:
        raise ValueError(
            "Feature shape mismatch.\n"
            f"Direct    : {direct_features.shape}\n"
            f"Inference : {inference_features.shape}"
        )

    differences = np.abs(
        direct_features - inference_features
    )

    max_difference = float(differences.max())

    different_indices = np.where(
        differences[0] > 1e-12
    )[0]

    print(f"Direct shape    : {direct_features.shape}")
    print(f"Inference shape : {inference_features.shape}")
    print(f"Max difference  : {max_difference:.12f}")
    print(f"Different count : {len(different_indices)}")

    if len(different_indices) > 0:
        print("\n[FAIL] Feature vectors are different")

        print("\nDIFFERENCES")
        print("-" * 60)

        for index in different_indices[:20]:
            print(
                f"{index:3d} | "
                f"{feature_names[index]:30s} | "
                f"direct={direct_features[0, index]} | "
                f"inference={inference_features[0, index]}"
            )

        raise AssertionError(
            "Direct and inference feature vectors "
            "are not identical."
        )

    print("[PASS] Feature vectors are identical")


def compare_predictions(
    direct_prediction,
    inference_prediction,
):
    print("\n" + "=" * 60)
    print("COMPARE MODEL PREDICTIONS")
    print("=" * 60)

    direct = float(direct_prediction[0])
    inference = float(inference_prediction[0])

    difference = abs(direct - inference)

    print(f"Direct prediction    : {direct:.10f}")
    print(f"Inference prediction : {inference:.10f}")
    print(f"Difference           : {difference:.10f}")

    if not np.isclose(
        direct,
        inference,
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Direct and inference predictions differ."
        )

    print("[PASS] Predictions are identical")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TEST TEAM INFERENCE CONSISTENCY")
    print("=" * 60)

    print("\nGOLDEN INPUT")
    print("-" * 60)
    print(f"Team : {TEAM}")
    print(f"Map  : {MAP_NAME}")
    print(f"Year : {YEAR}")
    print(f"Agents : {', '.join(AGENTS)}")

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_dataset()

    model, encoders, feature_names = load_artifacts()

    # --------------------------------------------------------
    # 2. Find exact historical row
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HISTORICAL LOOKUP")
    print("=" * 60)

    historical_row = find_historical_row(
        df,
        TEAM,
        MAP_NAME,
        YEAR,
        AGENTS,
    )

    if historical_row is None:
        raise ValueError(
            "Golden test composition was not found."
        )

    print("[PASS] Exact historical row found")
    print(
        f"Composition : "
        f"{historical_row['Composition Key']}"
    )

    # --------------------------------------------------------
    # 3. PATH A — Direct dataset encoding
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PATH A - DIRECT DATASET ENCODING")
    print("=" * 60)

    direct_row = pd.DataFrame([historical_row])

    direct_features = encode_features(
        direct_row,
        encoders,
    )

    print(
        f"[INFO] Direct feature shape : "
        f"{direct_features.shape}"
    )

    # --------------------------------------------------------
    # 4. PATH B — Application inference encoding
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PATH B - INFERENCE ENCODING")
    print("=" * 60)

    inference_row = build_inference_row(
        historical_row,
        TEAM,
        MAP_NAME,
        YEAR,
        AGENTS,
    )

    inference_features = encode_features(
        inference_row,
        encoders,
    )

    print(
        f"[INFO] Inference feature shape : "
        f"{inference_features.shape}"
    )

    # --------------------------------------------------------
    # 5. Feature consistency
    # --------------------------------------------------------

    compare_feature_vectors(
        direct_features,
        inference_features,
        feature_names,
    )

    # --------------------------------------------------------
    # 6. Prediction consistency
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PATH A - MODEL PREDICTION")
    print("=" * 60)

    direct_prediction = model.predict(
        direct_features
    )

    print(
        f"Direct predicted winrate : "
        f"{float(direct_prediction[0]):.10f}"
    )

    print("\n" + "=" * 60)
    print("PATH B - MODEL PREDICTION")
    print("=" * 60)

    inference_prediction = model.predict(
        inference_features
    )

    print(
        f"Inference predicted winrate : "
        f"{float(inference_prediction[0]):.10f}"
    )

    compare_predictions(
        direct_prediction,
        inference_prediction,
    )

    # --------------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    if direct_features.shape[1] != 116:
        raise AssertionError(
            "Expected exactly 116 features."
        )

    if len(feature_names) != 116:
        raise AssertionError(
            "Expected exactly 116 feature names."
        )

    if model.n_features_in_ != 116:
        raise AssertionError(
            "Expected model to accept exactly 116 features."
        )

    print("[PASS] 116-feature contract")
    print("[PASS] Direct vs inference features")
    print("[PASS] Direct vs inference prediction")
    print("[PASS] Inference consistency validation")


if __name__ == "__main__":
    main()
