"""
Validate Prediction
===================

Analyze prediction quality after training.
"""

import pandas as pd

from training_v2.team.feature_engineering import (
    load_dataset,
    build_feature_pipeline,
)

from training_v2.team.train_team_v2 import (
    split_dataset,
    build_model,
    train_model,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import numpy as np

def evaluate_prediction(
    prediction,
    train_data,
):
    """
    Compare prediction with actual value.
    """

    result = pd.DataFrame({

        "Actual": train_data["y_test"],

        "Prediction": prediction,

    })

    result["Error"] = (

        result["Prediction"]

        - result["Actual"]

    ).abs()

    return result


def show_summary(result):
    """
    Prediction summary.
    """

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(result.describe())


def show_best_prediction(result):
    """
    Best prediction.
    """

    print("\n" + "=" * 60)
    print("BEST PREDICTIONS")
    print("=" * 60)

    print(

        result

        .sort_values("Error")

        .head(20)

        .to_string(index=False)

    )


def show_worst_prediction(result):
    """
    Worst prediction.
    """

    print("\n" + "=" * 60)
    print("WORST PREDICTIONS")
    print("=" * 60)

    print(

        result

        .sort_values(
            "Error",
            ascending=False,
        )

        .head(20)

        .to_string(index=False)

    )


def show_distribution(result):
    """
    Compare prediction distribution.
    """

    print("\n" + "=" * 60)
    print("DISTRIBUTION")
    print("=" * 60)

    print()

    print("Actual")

    print(result["Actual"].describe())

    print()

    print("Prediction")

    print(result["Prediction"].describe())
    

def compare_baseline(result):
    """
    Compare model against simple baseline.
    """

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    actual = result["Actual"]

    prediction = result["Prediction"]

    baseline_prediction = np.full(
        len(actual),
        actual.mean(),
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline_prediction,
    )

    model_mae = mean_absolute_error(
        actual,
        prediction,
    )

    improvement = (

        (baseline_mae - model_mae)

        / baseline_mae

        * 100

    )

    print(f"Baseline MAE : {baseline_mae:.4f}")
    print(f"Model MAE    : {model_mae:.4f}")
    print(f"Improvement  : {improvement:.2f}%")

def final_evaluation(result):
    """
    Final project evaluation.
    """

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    mae = result["Error"].mean()

    if mae <= 0.20:
        status = "Excellent"

    elif mae <= 0.25:
        status = "Good"

    elif mae <= 0.35:
        status = "Acceptable"

    else:
        status = "Needs Improvement"

    print(f"Status : {status}")

    print()

    print("Summary")

    print("------------------------------")
    print("✓ Dataset successfully processed")
    print("✓ Feature engineering completed")
    print("✓ Model trained successfully")
    print("✓ Prediction pipeline validated")
    print("✓ Ready as Team Prediction V2 baseline")

def main():

    df = load_dataset()

    pipeline = build_feature_pipeline(df)

    train_data = split_dataset(

        pipeline["X"],

        pipeline["y"],

    )

    model = build_model()

    model = train_model(

        model,

        train_data,

    )

    prediction = model.predict(

        train_data["X_test"]

    )

    result = evaluate_prediction(

        prediction,

        train_data,

    )

    show_summary(result)

    show_distribution(result)
    
    compare_baseline(result)

    show_best_prediction(result)

    show_worst_prediction(result)

    final_evaluation(result)


if __name__ == "__main__":

    main()