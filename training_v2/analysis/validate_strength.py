from pathlib import Path

import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from training_v2.config import TEAM_DATASET_PATH


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_dataset():

    print_section("LOAD DATASET")

    dataset = pd.read_csv(TEAM_DATASET_PATH)

    print(f"[INFO] Shape : {dataset.shape}")

    return dataset


def validate_correlation(dataset: pd.DataFrame):

    print_section("VALIDATION 1 : CORRELATION")

    pearson, _ = pearsonr(
        dataset["Composition Strength"],
        dataset["Winrate"],
    )

    spearman, _ = spearmanr(
        dataset["Composition Strength"],
        dataset["Winrate"],
    )

    print(f"Pearson  : {pearson:.4f}")
    print(f"Spearman : {spearman:.4f}")

    print()

    if spearman >= 0.80:
        print("[INFO] Very Strong Relationship")

    elif spearman >= 0.60:
        print("[INFO] Strong Relationship")

    elif spearman >= 0.40:
        print("[INFO] Moderate Relationship")

    elif spearman >= 0.20:
        print("[INFO] Weak Relationship")

    else:
        print("[INFO] Very Weak Relationship")
        
        
def validate_filtered_correlation(dataset: pd.DataFrame):

    print_section("VALIDATION 1.5 : FILTERED CORRELATION")

    thresholds = [1, 2, 3, 5, 8, 10]

    results = []

    for minimum in thresholds:

        filtered = dataset[
            dataset["Total Maps Played"] >= minimum
        ]

        if len(filtered) < 5:
            continue

        pearson, _ = pearsonr(
            filtered["Composition Strength"],
            filtered["Winrate"],
        )

        spearman, _ = spearmanr(
            filtered["Composition Strength"],
            filtered["Winrate"],
        )

        results.append({
            "Min Maps": minimum,
            "Samples": len(filtered),
            "Pearson": round(pearson, 4),
            "Spearman": round(spearman, 4),
        })

    results = pd.DataFrame(results)

    print(results)

def validate_sample_distribution(dataset):

    print_section("SAMPLE DISTRIBUTION")

    summary = (

        dataset["Total Maps Played"]

        .describe()

        .round(2)

    )

    print(summary)

    print()

    print(

        dataset

        ["Total Maps Played"]

        .value_counts()

        .sort_index()

    )


def validate_bucket(dataset: pd.DataFrame):

    print_section("VALIDATION 2 : STRENGTH BUCKET")

    bucket = pd.cut(

        dataset["Composition Strength"],

        bins=[0, 20, 40, 60, 80, 100],

        include_lowest=True,

    )

    summary = (

        dataset

        .groupby(bucket)

        .agg(

            Sample=("Composition Strength", "count"),

            Avg_Strength=("Composition Strength", "mean"),

            Avg_Winrate=("Winrate", "mean"),

            Avg_Reliability=("Reliability", "mean"),

        )

        .round(4)

    )

    print(summary)


def validate_top(dataset: pd.DataFrame):

    print_section("VALIDATION 3 : TOP 20")

    columns = [

        "Team",
        "Map",
        "Year",
        "Agent",
        "Composition Strength",
        "Winrate",

    ]

    print(

        dataset

        .nlargest(
            20,
            "Composition Strength",
        )[columns]

    )


def validate_bottom(dataset: pd.DataFrame):

    print_section("VALIDATION 4 : BOTTOM 20")

    columns = [

        "Team",
        "Map",
        "Year",
        "Agent",
        "Composition Strength",
        "Winrate",

    ]

    print(

        dataset

        .nsmallest(
            20,
            "Composition Strength",
        )[columns]

    )


def summary(dataset):

    print_section("COMPOSITION STRENGTH SUMMARY")

    print(

        dataset["Composition Strength"]

        .describe()

        .round(4)

    )


def main():

    dataset = load_dataset()

    summary(dataset)
    
    validate_correlation(dataset)

    validate_filtered_correlation(dataset)
    
    validate_sample_distribution(dataset)

    validate_bucket(dataset)

    validate_top(dataset)

    validate_bottom(dataset)


if __name__ == "__main__":
    main()