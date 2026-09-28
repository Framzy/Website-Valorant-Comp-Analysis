"""
Test Team Prediction V2 inference pipeline.

Purpose:
    Validate that a real application-style input:
        Team + Map + Year + 5 Agents
    can be converted into the exact 116-feature contract expected
    by the Team Prediction V2 model, then produce a prediction.

This is a TEST script, not the production service.
"""

import ast
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
# TEST INPUT
# ============================================================

TEAM = "Paper Rex"
MAP_NAME = "Ascent"
YEAR = 2024

AGENTS = [
    "fade",
    "gekko",
    "viper",
    "omen",
    "raze",
]

# ============================================================
# HELPERS
# ============================================================

def load_artifacts():
    print("=" * 60)
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

    print(f"[INFO] Model features  : {model.n_features_in_}")
    print(f"[INFO] Feature names   : {len(feature_names)}")
    print("[INFO] Artifacts loaded")

    return model, encoders, feature_names


def load_dataset():
    print("\n" + "=" * 60)
    print("LOAD TEAM DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)

    # Same representation used by feature_engineering.py
    df["Agent"] = df["Agent"].apply(ast.literal_eval)

    print(f"[INFO] Dataset shape : {df.shape}")

    return df


def normalize_agents(agents):
    return sorted(agent.strip().lower() for agent in agents)


def find_historical_row(
    df,
    team,
    map_name,
    year,
    agents,
):
    """
    Find the exact historical Team + Map + Year + Composition row.

    Agent order is normalized because composition order should not
    affect the historical lookup.
    """

    target_agents = normalize_agents(agents)

    composition_match = df["Agent"].apply(
        lambda composition: normalize_agents(composition) == target_agents
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
            "Historical lookup returned multiple rows "
            "for the same Team + Map + Year + Composition."
        )

    return matches.iloc[0]


def build_inference_row(
    historical_row,
    team,
    map_name,
    year,
    agents,
):
    """
    Build a one-row DataFrame using the historical row's feature
    values. This intentionally reuses the already-built Team V2
    dataset for the three numeric historical features.
    """

    row = historical_row.copy()

    # Keep the application input as the source for categorical fields.
    row["Team"] = team
    row["Map"] = map_name
    row["Year"] = year
    row["Agent"] = agents

    return pd.DataFrame([row])


def print_input(
    team,
    map_name,
    year,
    agents,
):
    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)

    print(f"Team : {team}")
    print(f"Map  : {map_name}")
    print(f"Year : {year}")
    print("Agents:")

    for agent in agents:
        print(f"  - {agent}")


def print_historical_features(row):
    print("\n" + "=" * 60)
    print("HISTORICAL FEATURES")
    print("=" * 60)

    print("[FOUND]")
    print(f"Composition Key      : {row['Composition Key']}")
    print(f"Team Overall WR      : {row['Team Overall WR']:.4f}")
    print(f"Team Map WR          : {row['Team Map WR']:.4f}")
    print(f"Composition Strength : {row['Composition Strength']:.4f}")


def validate_feature_vector(
    X_encoded,
    feature_names,
    model,
):
    print("\n" + "=" * 60)
    print("FEATURE CONTRACT")
    print("=" * 60)

    feature_count = X_encoded.shape[1]

    print(f"Feature Vector : {feature_count}")
    print(f"Feature Names  : {len(feature_names)}")
    print(f"Model Features : {model.n_features_in_}")

    if feature_count != len(feature_names):
        raise ValueError(
            f"Feature vector/name mismatch: "
            f"{feature_count} != {len(feature_names)}"
        )

    if feature_count != model.n_features_in_:
        raise ValueError(
            f"Feature vector/model mismatch: "
            f"{feature_count} != {model.n_features_in_}"
        )

    print("[PASS] Feature vector compatible with model")


def print_role_features(row):
    print("\n" + "=" * 60)
    print("ROLE FEATURES")
    print("=" * 60)

    print(f"Controller : {int(row['Controller Count'])}")
    print(f"Duelist    : {int(row['Duelist Count'])}")
    print(f"Initiator  : {int(row['Initiator Count'])}")
    print(f"Sentinel   : {int(row['Sentinel Count'])}")

    print("\nRole Pattern:")
    print(f"  {row['Role Pattern']}")


def main():
    print("=" * 60)
    print("TEST TEAM INFERENCE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Input
    # --------------------------------------------------------

    print_input(
        TEAM,
        MAP_NAME,
        YEAR,
        AGENTS,
    )

    # Basic input validation.
    if len(AGENTS) != 5:
        raise ValueError(
            "Team inference requires exactly 5 agents."
        )

    if len(set(AGENTS)) != 5:
        raise ValueError(
            "Team composition cannot contain duplicate agents."
        )

    # --------------------------------------------------------
    # 2. Load dataset + artifacts
    # --------------------------------------------------------

    df = load_dataset()

    model, encoders, feature_names = load_artifacts()

    # --------------------------------------------------------
    # 3. Historical lookup
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
            "Exact historical composition was not found.\n"
            "This first test intentionally requires an exact "
            "historical row so we can validate the feature contract "
            "without introducing fallback logic."
        )

    print("[PASS] Exact historical composition found")

    print_historical_features(historical_row)

    # --------------------------------------------------------
    # 4. Role information
    # --------------------------------------------------------

    print_role_features(historical_row)

    # --------------------------------------------------------
    # 5. Build inference row
    # --------------------------------------------------------

    inference_row = build_inference_row(
        historical_row,
        TEAM,
        MAP_NAME,
        YEAR,
        AGENTS,
    )

    # --------------------------------------------------------
    # 6. Encode using saved training encoders
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILD INFERENCE FEATURES")
    print("=" * 60)

    X_encoded = encode_features(
        inference_row,
        encoders,
    )

    print(f"[INFO] Encoded shape : {X_encoded.shape}")

    # --------------------------------------------------------
    # 7. Validate exact 116-feature contract
    # --------------------------------------------------------

    validate_feature_vector(
        X_encoded,
        feature_names,
        model,
    )

    # --------------------------------------------------------
    # 8. Prediction
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL PREDICTION")
    print("=" * 60)

    prediction = model.predict(X_encoded)

    predicted_winrate = float(prediction[0])

    print(f"Predicted Winrate : {predicted_winrate:.4f}")
    print(f"Predicted Percent : {predicted_winrate * 100:.2f}%")

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print("[PASS] Historical lookup")
    print("[PASS] Feature preparation")
    print("[PASS] 116-feature contract")
    print("[PASS] Model prediction")
    print("[PASS] Team inference pipeline")


if __name__ == "__main__":
    main()
