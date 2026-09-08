from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "valorant_dataset_general_v2.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    print("\n" + "=" * 60)
    print("LOAD GENERAL DATASET")
    print("=" * 60)

    dataset = pd.read_csv(DATASET_PATH)

    print(f"[INFO] Shape : {dataset.shape}")

    return dataset


# ============================================================
# CALCULATE BASIC STATISTICS
# ============================================================

def calculate_basic_statistics(dataset):
    print("\n" + "=" * 60)
    print("CALCULATE BASIC STATISTICS")
    print("=" * 60)

    dataset = dataset.copy()

    # --------------------------------------------------------
    # Historical Winrate
    # --------------------------------------------------------

    dataset["Winrate"] = (
        dataset["Total Wins By Map"]
        / dataset["Total Maps Played"]
    )

    # --------------------------------------------------------
    # Total Maps per Map + Year context
    # --------------------------------------------------------

    context_maps = (
        dataset
        .groupby(
            ["Map", "Year"]
        )["Total Maps Played"]
        .transform("sum")
    )

    dataset["Context Maps"] = context_maps

    # --------------------------------------------------------
    # Pick Rate
    # --------------------------------------------------------

    dataset["Pick Rate"] = (
        dataset["Total Maps Played"]
        / dataset["Context Maps"]
    )

    return dataset


# ============================================================
# VALIDATE BASIC STATISTICS
# ============================================================

def validate_statistics(dataset):
    print("\n" + "=" * 60)
    print("VALIDATE BASIC STATISTICS")
    print("=" * 60)

    invalid_winrate = (
        (dataset["Winrate"] < 0)
        | (dataset["Winrate"] > 1)
        | dataset["Winrate"].isna()
    ).sum()

    invalid_pick_rate = (
        (dataset["Pick Rate"] < 0)
        | (dataset["Pick Rate"] > 1)
        | dataset["Pick Rate"].isna()
    ).sum()

    print(
        f"[INFO] Invalid Winrate   : "
        f"{invalid_winrate}"
    )

    print(
        f"[INFO] Invalid Pick Rate : "
        f"{invalid_pick_rate}"
    )

    if invalid_winrate > 0:
        raise ValueError(
            "Invalid Winrate detected."
        )

    if invalid_pick_rate > 0:
        raise ValueError(
            "Invalid Pick Rate detected."
        )

    print("[INFO] Validation Passed")


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def display_sample(dataset):
    print("\n" + "=" * 60)
    print("STATISTICS SAMPLE")
    print("=" * 60)

    columns = [
        "Map",
        "Year",
        "Agent",
        "Total Wins By Map",
        "Total Loss By Map",
        "Total Maps Played",
        "Winrate",
        "Context Maps",
        "Pick Rate",
    ]

    print(
        dataset[columns]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    dataset = calculate_basic_statistics(
        dataset
    )

    validate_statistics(
        dataset
    )

    display_sample(
        dataset
    )


if __name__ == "__main__":
    main()