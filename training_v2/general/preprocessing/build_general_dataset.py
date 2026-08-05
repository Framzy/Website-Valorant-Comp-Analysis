"""
Build General Dataset V2
=====================
"""

from pathlib import Path

import pandas as pd

from training_v2.general.config import (
    DATASET_PATH,
    DATASET_DIR,
    AGENT_ROLE_MAP,
)
from training_v2.general.preprocessing.composition_normalizer import (
    analyze_roles,
    calculate_role_distribution,
    allocate_role_slots,
    select_agents,
)
from training_v2.general.constants import TEAM_NAME_MAPPING

from pprint import pprint

DEBUG = False

def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    print(f"[WARNING] {message}")


def normalize_team_name(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize inconsistent
    team names.
    """

    df = df.copy()

    before = df["Team"].nunique()

    df["Team"] = (
        df["Team"]
        .replace(TEAM_NAME_MAPPING)
    )

    after = df["Team"].nunique()

    print()

    print("=" * 60)
    print("NORMALIZE TEAM NAME")
    print("=" * 60)

    print(f"[INFO] Before : {before}")

    print(f"[INFO] After  : {after}")

    return df

def load_dataset() -> pd.DataFrame:
    """
    Load raw dataset.
    """

    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)
    df = normalize_team_name(df)

    print(f"[INFO] Shape : {df.shape}")

    return df

def aggregate_matches(
    df: pd.DataFrame,
) -> pd.DataFrame:

    log_section("AGGREGATE MATCHES")

    before = len(df)

    group_columns = [

        "Tournament",

        "Stage",

        "Match Type",

        "Map",

        "Team",

    ]
    
    df = df[
    (df["Stage"] == "All Stages")
    &
    (df["Match Type"] == "All Match Types")
    ].copy()

    after = len(df)
    log_info(f"Raw Rows        : {before}")
    log_info(f"All Stages Rows : {after}")
    
    grouped = (

        df.groupby(group_columns)

        .agg(

            {

                "Agent": list,

                "Total Wins By Map": "max",

                "Total Loss By Map": "max",

                "Total Maps Played": "max",

                "Year": "first",

            }

        )

        .reset_index()

    )

    return grouped


def aggregate_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate raw dataset into compositions.
    """

    print("\n" + "=" * 60)
    print("AGGREGATE DATASET")
    print("=" * 60)

    grouped = aggregate_matches(df)

    print(f"[INFO] Total Composition : {len(grouped)}")

    return grouped


def split_dataset(
    grouped: pd.DataFrame
):
    """
    Split composition into

    - exact 5 agents
    - more than 5 agents
    """

    exact_five = grouped[
        grouped["Agent"].apply(len) == 5
    ].copy()

    greater_than_five = grouped[
        grouped["Agent"].apply(len) > 5
    ].copy()

    print("\n" + "=" * 60)
    print("SPLIT DATASET")
    print("=" * 60)

    print(f"[INFO] Exactly 5 Agent : {len(exact_five)}")

    print(f"[INFO] >5 Agent        : {len(greater_than_five)}")

    return exact_five, greater_than_five

def build_agent_played_frequency(
    raw_df: pd.DataFrame,
    tournament: str,
    team: str,
    map_name: str,
) -> dict:
    """
    Build agent played frequency.

    Agent Played =
        Total Wins By Map +
        Total Loss By Map
    """

    mask = (
        ( raw_df["Tournament"] == tournament)
        &
        (raw_df["Team"] == team)
        &
        (raw_df["Map"] == map_name)
        &
        (raw_df["Stage"] == "All Stages")
        &
        (raw_df["Match Type"] == "All Match Types")
    )

    subset = raw_df.loc[mask]

    subset = subset.copy()

    subset["Agent Played"] = (
        subset["Total Wins By Map"]
        +
        subset["Total Loss By Map"]
    )
    
    summary = (
        subset
        .groupby("Agent")
        .agg(
            Played=("Agent Played", "sum"),
            Wins=("Total Wins By Map", "sum"),
            Losses=("Total Loss By Map", "sum"),
        )
    )

    frequency = {}

    for agent, row in summary.iterrows():

        frequency[agent.lower()] = {

            "played": int(row["Played"]),

            "wins": int(row["Wins"]),

            "losses": int(row["Losses"]),

        }
        
    if DEBUG:
        pprint(frequency)

    return frequency
    
def reconstruct_composition(
    frequency: dict[str, int]
) -> list[str]:
    """
    Reconstruct composition from
    historical agent frequency.
    """

    analysis = analyze_roles(
        frequency
    )

    distribution = calculate_role_distribution(
        analysis
    )

    slots = allocate_role_slots(
        distribution
    )

    result = select_agents(
        distribution,
        slots
    )

    return sorted(
        result["composition"]
    )

def reconstruct_dataset(
    raw_df: pd.DataFrame,
    greater_than_five: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct every composition
    containing more than five agents.
    """

    reconstructed = greater_than_five.copy()

    reconstructed_agents = []

    print("\n" + "=" * 60)
    print("RECONSTRUCT DATASET")
    print("=" * 60)

    total = len(reconstructed)

    for index, row in reconstructed.iterrows():

        frequency = build_agent_played_frequency(

            raw_df = raw_df,

            tournament=row["Tournament"],

            team=row["Team"],

            map_name=row["Map"],
        )
        
        composition = reconstruct_composition(
            frequency
        )

        reconstructed_agents.append(
            composition
        )

        if (len(reconstructed_agents) % 100) == 0:

            print(
                f"[INFO] Processed "
                f"{len(reconstructed_agents)}/{total}"
            )

    reconstructed["Agent"] = reconstructed_agents
    
    return reconstructed

def merge_dataset(
    exact_five: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge original and reconstructed compositions.
    """

    print("\n" + "=" * 60)
    print("MERGE DATASET")
    print("=" * 60)

    dataset = pd.concat(

        [

            exact_five,

            reconstructed

        ],

        ignore_index=True

    )

    print(

        f"[INFO] Total Composition : {len(dataset)}"

    )

    return dataset

def aggregate_compositions(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge identical compositions played in different tournaments.

    One row represents one unique:
     Map + Year + Composition
    """

    print("\n" + "=" * 60)
    print("AGGREGATE COMPOSITIONS")
    print("=" * 60)

    dataset = dataset.copy()

    # ----------------------------------------
    # Temporary composition key
    # ----------------------------------------

    dataset["Composition Key"] = (
        dataset["Agent"]
        .apply(lambda agents: "|".join(sorted(agents)))
    )

    grouped = (

        dataset

        .groupby(
            [
                "Map",
                "Year",
                "Composition Key",
            ],
            as_index=False,
        )

        .agg(
            {
                "Agent": lambda x: list(sorted(x.iloc[0])),

                "Total Wins By Map": "sum",

                "Total Loss By Map": "sum",

                "Total Maps Played": "sum",
            }
        )

    )

    print(f"[INFO] Before : {len(dataset)}")
    print(f"[INFO] After  : {len(grouped)}")

    return grouped

def validate_general_dataset(
    dataset: pd.DataFrame,
):
    """
    Validate final general dataset.
    """

    print("\n" + "=" * 60)
    print("VALIDATE general DATASET")
    print("=" * 60)

    temp = dataset.copy()

    temp["Composition Key"] = (
        temp["Agent"]
        .apply(lambda x: "|".join(sorted(x)))
    )
    
    temp["Context Key"] = (
        temp["Map"]
        + "|"
        + temp["Year"].astype(str)
        + "|"
        + temp["Composition Key"]
    )
    
    print(f"[INFO] Sample Context Key : {temp['Context Key'].iloc[0]}")

    duplicates = temp.duplicated(
        subset=[
            "Context Key"
        ]
    )
    
    print(
        f"[INFO] Duplicate Composition : {duplicates.sum()}"
    )

    print(
        f"[INFO] Total Composition     : {len(temp)}"
    )

    print(
        f"[INFO] Unique Composition    : "
        f"{temp['Context Key'].nunique()}"
    )

    if duplicates.sum() == 0:

        print("[INFO] Validation Passed")

    if duplicates.sum() > 0:
        raise ValueError(
            f"Found {duplicates.sum()} duplicate compositions."
        )
        
        
def report_dataset(
    dataset: pd.DataFrame,
):
    """
    Print dataset summary.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print()

    print(
        dataset["Total Maps Played"]
        .describe()
    )

    print()

    print(
        dataset[
            [
                "Map",
                "Year",
                "Agent",
                "Total Maps Played",
            ]
        ]
        .head(20)
    )

OUTPUT_DATASET = (

    DATASET_DIR

    /

    "valorant_dataset_general_v2.csv"

)

def save_dataset(
    dataset: pd.DataFrame
):
    """
    Save General Dataset V2.
    """

    print("\n" + "=" * 60)
    print("SAVE DATASET")
    print("=" * 60)

    dataset.to_csv(

        OUTPUT_DATASET,

        index=False

    )

    print(

        f"[INFO] Saved : "

        f"{OUTPUT_DATASET.name}"

    )
def main():

    df = load_dataset()

    grouped = aggregate_dataset(df)

    exact_five, greater_than_five = split_dataset(grouped)

    reconstructed = reconstruct_dataset(raw_df=df, greater_than_five=greater_than_five,)

    dataset = merge_dataset(exact_five, reconstructed,)
    
    dataset = aggregate_compositions(dataset)

    validate_general_dataset(dataset)
    
    report_dataset(dataset)

    
    print()

    dataset = dataset.sort_values(
        by=[
            "Year",
            "Map",
        ]
    )
    
    save_dataset(
        dataset
    )
    


if __name__ == "__main__":
    main()