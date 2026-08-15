import ast
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
# CALCULATE POPULARITY
# ============================================================

def calculate_popularity(dataset):
    """
    Calculate composition popularity within
    each Map + Year context.
    """

    print("\n" + "=" * 60)
    print("CALCULATE POPULARITY")
    print("=" * 60)

    dataset = dataset.copy()

    # --------------------------------------------------------
    # Context total
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
    # Pick rate
    # --------------------------------------------------------

    dataset["Pick Rate"] = (
        dataset["Total Maps Played"]
        / dataset["Context Maps"]
    )

    # --------------------------------------------------------
    # Popularity rank
    # --------------------------------------------------------

    dataset["Popularity Rank"] = (
        dataset
        .groupby(["Map", "Year"])["Pick Rate"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Context composition count
    # --------------------------------------------------------

    dataset["Context Composition Count"] = (
        dataset
        .groupby(["Map", "Year"])["Agent"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Popularity percentile
    # --------------------------------------------------------

    dataset["Popularity Percentile"] = (
        1
        - (
            dataset["Popularity Rank"] - 1
        )
        / (
            dataset["Context Composition Count"] - 1
        ).replace(0, 1)
    )

    return dataset


# ============================================================
# DISPLAY TOP COMPOSITIONS
# ============================================================

def display_top_compositions(
    dataset,
    map_name,
    year,
    limit=10,
):
    """
    Display the most frequently played compositions
    for a specific Map + Year context.
    """

    context = dataset[
        (dataset["Map"] == map_name)
        & (dataset["Year"] == year)
    ].copy()

    if context.empty:
        print(
            f"[INFO] No historical data for "
            f"{map_name} {year}"
        )
        return

    context = context.sort_values(
        "Popularity Rank"
    )

    columns = [
        "Map",
        "Year",
        "Agent",
        "Total Maps Played",
        "Pick Rate",
        "Popularity Rank",
    ]

    print("\n" + "=" * 60)
    print(
        f"TOP COMPOSITIONS : "
        f"{map_name} {year}"
    )
    print("=" * 60)

    print(
        context[columns]
        .head(limit)
        .to_string(index=False)
    )


# ============================================================
# FIND EXACT COMPOSITION
# ============================================================

def find_exact_composition(
    dataset,
    map_name,
    year,
    agents,
):
    """
    Find an exact historical composition
    within a specific Map + Year context.

    Returns None when no exact composition exists.
    """

    context = dataset[
        (dataset["Map"] == map_name)
        & (dataset["Year"] == year)
    ].copy()

    if context.empty:
        return None

    target_agents = sorted(
        agent.lower().strip()
        for agent in agents
    )

    for _, row in context.iterrows():

        historical_agents = row["Agent"]

        if isinstance(
            historical_agents,
            str,
        ):
            try:
                historical_agents = ast.literal_eval(
                    historical_agents
                )
            except (ValueError, SyntaxError):
                continue

        historical_agents = sorted(
            agent.lower().strip()
            for agent in historical_agents
        )

        if historical_agents == target_agents:
            return row

    return None


# ============================================================
# DISPLAY EXACT COMPOSITION
# ============================================================

def display_exact_composition(
    dataset,
    map_name,
    year,
    agents,
):
    """
    Display historical information for
    an exact composition.
    """

    result = find_exact_composition(
        dataset,
        map_name,
        year,
        agents,
    )

    print("\n" + "=" * 60)
    print("EXACT COMPOSITION LOOKUP")
    print("=" * 60)

    if result is None:

        print("[INFO] Exact Composition : Not Found")
        print(f"[INFO] Map               : {map_name}")
        print(f"[INFO] Year              : {year}")
        print("[INFO] Historical Status : No Exact Historical Data")

        return

    print("[INFO] Exact Composition : Found")
    print(f"[INFO] Map               : {result['Map']}")
    print(f"[INFO] Year              : {result['Year']}")
    print(f"[INFO] Played            : {result['Total Maps Played']}")
    print(f"[INFO] Winrate           : {result['Winrate']:.4f}")
    print(f"[INFO] Pick Rate         : {result['Pick Rate']:.4f}")
    print(
        f"[INFO] Popularity Rank   : "
        f"#{result['Popularity Rank']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    # Basic statistics are required because
    # popularity uses Winrate and Pick Rate.
    dataset["Winrate"] = (
        dataset["Total Wins By Map"]
        / dataset["Total Maps Played"]
    )

    dataset = calculate_popularity(
        dataset
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATE POPULARITY")
    print("=" * 60)

    invalid_rank = (
        dataset["Popularity Rank"] < 1
    ).sum()

    invalid_percentile = (
        (dataset["Popularity Percentile"] < 0)
        | (dataset["Popularity Percentile"] > 1)
    ).sum()

    print(
        f"[INFO] Invalid Rank       : "
        f"{invalid_rank}"
    )

    print(
        f"[INFO] Invalid Percentile : "
        f"{invalid_percentile}"
    )

    if invalid_rank > 0:
        raise ValueError(
            "Invalid popularity rank detected."
        )

    if invalid_percentile > 0:
        raise ValueError(
            "Invalid popularity percentile detected."
        )

    print("[INFO] Validation Passed")

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    display_top_compositions(
        dataset,
        map_name="Abyss",
        year=2024,
        limit=10,
    )

    # --------------------------------------------------------
    # Exact composition test
    # --------------------------------------------------------

    display_exact_composition(
        dataset,
        map_name="Abyss",
        year=2024,
        agents=[
            "cypher",
            "jett",
            "kayo",
            "omen",
            "sova",
        ],
    )

    # --------------------------------------------------------
    # Unknown composition test
    # --------------------------------------------------------

    display_exact_composition(
        dataset,
        map_name="Abyss",
        year=2024,
        agents=[
            "jett",
            "reyna",
            "yoru",
            "neon",
            "phoenix",
        ],
    )


if __name__ == "__main__":
    main()