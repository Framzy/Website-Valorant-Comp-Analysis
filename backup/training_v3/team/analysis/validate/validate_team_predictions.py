"""
Validate Team Prediction V2 model behavior.

Purpose:
    1. Reproduce the Team V2 holdout split (80/20, random_state=42).
    2. Measure MAE, RMSE, and R2 on the holdout set.
    3. Inspect prediction distribution.
    4. Compare historical Winrate vs predicted Winrate for selected
       historical compositions.
    5. Highlight large prediction errors.

This script is a model sanity check, not the production service.
"""

import ast
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.team.config import (
    TEAM_DATASET_PATH,
    MODEL_TEAM_DIR,
)
from training.team.feature_engineering import (
    build_feature_pipeline,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLE_SIZE = 20
LARGE_ERROR_THRESHOLD = 0.30


def load_dataset():
    print("=" * 60)
    print("LOAD TEAM DATASET")
    print("=" * 60)

    df = pd.read_csv(TEAM_DATASET_PATH)
    df["Agent"] = df["Agent"].apply(ast.literal_eval)

    print(f"[INFO] Dataset shape : {df.shape}")

    return df


def load_model():
    print("\n" + "=" * 60)
    print("LOAD MODEL")
    print("=" * 60)

    model = joblib.load(
        MODEL_TEAM_DIR / "team_model_v2.joblib"
    )

    print(f"[INFO] Model features : {model.n_features_in_}")

    return model


def evaluate_holdout(df, model):
    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION")
    print("=" * 60)

    pipeline = build_feature_pipeline(df)

    X = pipeline["X"]
    y = pipeline["y"].to_numpy()

    indices = np.arange(len(df))

    (
        X_train,
        X_test,
        y_train,
        y_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction,
        )
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    print(f"Train rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")
    print()
    print(f"MAE  : {mae:.4f} ({mae * 100:.2f} percentage points)")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return pipeline, prediction, idx_test


def print_prediction_distribution(
    y_test,
    prediction,
):
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)

    print(
        f"Actual mean      : {np.mean(y_test):.4f}"
    )
    print(
        f"Prediction mean  : {np.mean(prediction):.4f}"
    )
    print(
        f"Actual min       : {np.min(y_test):.4f}"
    )
    print(
        f"Actual max       : {np.max(y_test):.4f}"
    )
    print(
        f"Prediction min   : {np.min(prediction):.4f}"
    )
    print(
        f"Prediction max   : {np.max(prediction):.4f}"
    )

    negative_count = int(
        np.sum(prediction < 0)
    )

    over_one_count = int(
        np.sum(prediction > 1)
    )

    print(
        f"Prediction < 0   : {negative_count}"
    )
    print(
        f"Prediction > 1   : {over_one_count}"
    )


def print_holdout_samples(
    df,
    y_test,
    prediction,
    idx_test,
):
    print("\n" + "=" * 60)
    print("HOLDOUT SAMPLE PREDICTIONS")
    print("=" * 60)

    result = df.iloc[idx_test].copy()

    result["Actual WR"] = y_test
    result["Predicted WR"] = prediction
    result["Absolute Error"] = np.abs(
        y_test - prediction
    )

    result = result.sort_values(
        "Absolute Error",
        ascending=False,
    )

    display_columns = [
        "Team",
        "Map",
        "Year",
        "Composition Key",
        "Actual WR",
        "Predicted WR",
        "Absolute Error",
    ]

    sample = result[
        display_columns
    ].head(SAMPLE_SIZE)

    print(
        sample.to_string(
            index=False,
            formatters={
                "Actual WR": "{:.4f}".format,
                "Predicted WR": "{:.4f}".format,
                "Absolute Error": "{:.4f}".format,
            },
        )
    )


def inspect_golden_composition(
    df,
    model,
    pipeline,
):
    print("\n" + "=" * 60)
    print("GOLDEN COMPOSITION CHECK")
    print("=" * 60)

    target_team = "100 Thieves"
    target_map = "Abyss"
    target_year = 2024
    target_agents = [
        "cypher",
        "gekko",
        "jett",
        "omen",
        "sova",
    ]

    def normalize(agents):
        return sorted(
            str(agent).strip().lower()
            for agent in agents
        )

    target = normalize(target_agents)

    composition_match = df["Agent"].apply(
        lambda composition: normalize(composition) == target
    )

    mask = (
        (df["Team"] == target_team)
        & (df["Map"] == target_map)
        & (df["Year"] == target_year)
        & composition_match
    )

    matches = df[mask]

    if matches.empty:
        print("[INFO] Golden composition not found")
        return

    row_index = matches.index[0]

    X_row = pipeline["X"][row_index:row_index + 1]

    prediction = float(
        model.predict(X_row)[0]
    )

    actual = float(
        df.loc[row_index, "Winrate"]
    )

    print(f"Team       : {target_team}")
    print(f"Map        : {target_map}")
    print(f"Year       : {target_year}")
    print(
        f"Composition: "
        f"{df.loc[row_index, 'Composition Key']}"
    )
    print(f"Historical WR : {actual:.4f} ({actual * 100:.2f}%)")
    print(f"Predicted WR  : {prediction:.4f} ({prediction * 100:.2f}%)")
    print(
        f"Absolute Error: "
        f"{abs(actual - prediction):.4f}"
    )


def main():
    print("=" * 60)
    print("VALIDATE TEAM PREDICTIONS")
    print("=" * 60)

    df = load_dataset()
    model = load_model()

    pipeline, prediction, idx_test = evaluate_holdout(
        df,
        model,
    )

    X = pipeline["X"]
    y = pipeline["y"].to_numpy()

    # Recreate the same holdout split to obtain y_test.
    indices = np.arange(len(df))

    (
        _,
        _,
        y_train,
        y_test,
        _,
        _,
    ) = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print_prediction_distribution(
        y_test,
        prediction,
    )

    print_holdout_samples(
        df,
        y_test,
        prediction,
        idx_test,
    )

    inspect_golden_composition(
        df,
        model,
        pipeline,
    )

    large_errors = np.abs(
        y_test - prediction
    )

    count_large_errors = int(
        np.sum(
            large_errors >= LARGE_ERROR_THRESHOLD
        )
    )

    print("\n" + "=" * 60)
    print("ERROR SUMMARY")
    print("=" * 60)

    print(
        f"Errors >= {LARGE_ERROR_THRESHOLD:.2f} "
        f"({LARGE_ERROR_THRESHOLD * 100:.0f} pp): "
        f"{count_large_errors}/{len(y_test)}"
    )

    print()
    print("[INFO] This test evaluates model behavior.")
    print("[INFO] It does not modify the model or artifacts.")


if __name__ == "__main__":
    main()
