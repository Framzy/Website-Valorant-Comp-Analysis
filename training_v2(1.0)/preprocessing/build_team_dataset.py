"""
Build Team Dataset V2
=====================
"""

from pathlib import Path

import pandas as pd
import numpy as np

from training_v2.config import (
    DATASET_PATH,
    DATASET_DIR,
    AGENT_ROLE_MAP,
)
from training_v2.training_utils import aggregate_matches
from training_v2.preprocessing.composition_normalizer import (
    analyze_roles,
    calculate_role_distribution,
    allocate_role_slots,
    select_agents,
)
from training_v2.training_utils import normalize_team_name
from pprint import pprint

DEBUG = False


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
    Team + Map + Year + Composition
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
                "Team",
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

def validate_training_dataset(
    dataset: pd.DataFrame,
):
    """
    Validate final training dataset.
    """

    print("\n" + "=" * 60)
    print("VALIDATE TRAINING DATASET")
    print("=" * 60)

    temp = dataset.copy()

    temp["Composition Key"] = (
        temp["Agent"]
        .apply(lambda x: "|".join(sorted(x)))
    )
    
    temp["Context Key"] = (
        temp["Team"]
        + "|"
        + temp["Map"]
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
                "Team",
                "Map",
                "Year",
                "Agent",
                "Total Maps Played",
            ]
        ]
        .head(20)
    )

def calculate_winrate(
    dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate composition winrate.
    """

    dataset = dataset.copy()

    dataset["Winrate"] = (

        dataset["Total Wins By Map"]

        /

        (

            dataset["Total Wins By Map"]

            +

            dataset["Total Loss By Map"]

        )

    )

    print("\n" + "=" * 60)
    print("CALCULATE WINRATE")
    print("=" * 60)

    print(

        f"[INFO] Mean Winrate : "

        f"{dataset['Winrate'].mean():.4f}"

    )
    

    return dataset

def calculate_context_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate context statistics.

    Context
    -------
    Team
    Map
    Year
    """

    print("\n" + "=" * 60)
    print("CALCULATE CONTEXT STATISTICS")
    print("=" * 60)

    df = df.copy()

    df["Context Played"] = (

        df

        .groupby(
            [
                "Team",
                "Map",
                "Year",
            ]
        )[
            "Total Maps Played"
        ]

        .transform("sum")

    )

    print("[INFO] Context Statistics Calculated")

    print()

    print(

        df[
            [
                "Team",
                "Map",
                "Year",
                "Total Maps Played",
                "Context Played",
            ]
        ].head(10)

    )

    return df

def calculate_usage_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate composition usage.

    Usage Ratio =
    Composition Played
    /
    Context Played
    """

    print("\n" + "=" * 60)
    print("CALCULATE USAGE STATISTICS")
    print("=" * 60)

    df = df.copy()

    df["Usage Ratio"] = (

        df["Total Maps Played"]

        /

        df["Context Played"]

    ).round(4)

    print("[INFO] Usage Statistics Calculated")

    print()

    print(

        df[
            [
                "Total Maps Played",
                "Context Played",
                "Usage Ratio",
            ]
        ].head(10)

    )

    return df

def calculate_reliability(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical reliability.

    Reliability =
    Usage Ratio
    ×
    log(Played)
    """

    print("\n" + "=" * 60)
    print("CALCULATE RELIABILITY")
    print("=" * 60)

    df = df.copy()

    max_played = df["Total Maps Played"].max()

    df["Reliability"] = (
        np.log1p(df["Total Maps Played"])
        /
        np.log1p(max_played)
    ).round(4)

    print("[INFO] Reliability Calculated")

    print()

    print(

        df[
            [
                "Total Maps Played",
                "Context Played",
                "Usage Ratio",
                "Reliability",
            ]
        ].head(10)

    )

    return df

def calculate_role_statistics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate role statistics for every composition.
    """

    print("\n" + "=" * 60)
    print("CALCULATE ROLE STATISTICS")
    print("=" * 60)

    dataset = dataset.copy()

    role_columns = [
        "Duelist Count",
        "Initiator Count",
        "Controller Count",
        "Sentinel Count",
    ]

    for column in role_columns:
        dataset[column] = 0

    role_pattern = []

    for index, row in dataset.iterrows():

        counts = {
            "duelist": 0,
            "initiator": 0,
            "controller": 0,
            "sentinel": 0,
        }

        for agent in row["Agent"]:

            role = AGENT_ROLE_MAP.get(agent)

            if role is not None:
                counts[role] += 1

        dataset.at[index, "Duelist Count"] = counts["duelist"]
        dataset.at[index, "Initiator Count"] = counts["initiator"]
        dataset.at[index, "Controller Count"] = counts["controller"]
        dataset.at[index, "Sentinel Count"] = counts["sentinel"]

        role_pattern.append(
            f"{counts['duelist']}D-"
            f"{counts['initiator']}I-"
            f"{counts['controller']}C-"
            f"{counts['sentinel']}S"
        )

    dataset["Role Pattern"] = role_pattern

    print("[INFO] Role Statistics Calculated")

    print()

    print(

        dataset[
            [
                "Agent",
                "Duelist Count",
                "Initiator Count",
                "Controller Count",
                "Sentinel Count",
                "Role Pattern",
            ]
        ].head(10)

    )

    return dataset

def calculate_role_pattern_statistics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate role pattern statistics.
    """

    print("\n" + "=" * 60)
    print("CALCULATE ROLE PATTERN STATISTICS")
    print("=" * 60)

    dataset = dataset.copy()

    pattern_stats = (

        dataset

        .groupby(
            [
                "Team",
                "Map",
                "Year",
                "Role Pattern",
            ],
            as_index=False,
        )["Total Maps Played"]

        .sum()

        .rename(
            columns={
                "Total Maps Played": "Pattern Played",
            }
        )

    )

    pattern_stats["Pattern Usage"] = (

        pattern_stats["Pattern Played"]

        /

        pattern_stats.groupby(
            [
                "Team",
                "Map",
                "Year",
            ]
        )["Pattern Played"]

        .transform("sum")

    ).round(4)

    dataset = dataset.merge(

        pattern_stats,

        on=[
            "Team",
            "Map",
            "Year",
            "Role Pattern",
        ],

        how="left",

    )

    print("[INFO] Role Pattern Statistics Calculated")

    print()

    print(

        dataset[
            [
                "Team",
                "Map",
                "Year",
                "Role Pattern",
                "Pattern Played",
                "Pattern Usage",
            ]
        ].head(10)

    )

    return dataset

def calculate_historical_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical team statistics.
    """

    print("\n" + "=" * 60)
    print("CALCULATE HISTORICAL STATISTICS")
    print("=" * 60)

    # -------------------------
    # Team Overall WR
    # -------------------------

    df["Team Overall WR"] = (
        df.groupby("Team")["Winrate"]
        .transform("mean")
        .round(4)
    )

    # -------------------------
    # Team Map WR
    # -------------------------

    df["Team Map WR"] = (
        df.groupby(
            [
                "Team",
                "Map",
            ]
        )["Winrate"]
        .transform("mean")
        .round(4)
    )

    print("[INFO] Historical Statistics Calculated")

    return df

def calculate_agent_familiarity(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate agent familiarity for every
    Team + Map + Year.
    """

    print("\n" + "=" * 60)
    print("CALCULATE AGENT FAMILIARITY")
    print("=" * 60)

    records = []

    for _, row in dataset.iterrows():

        for agent in row["Agent"]:

            records.append({

                "Team": row["Team"],

                "Map": row["Map"],

                "Year": row["Year"],

                "Agent": agent,

                "Played": row["Total Maps Played"],

            })

    familiarity = pd.DataFrame(records)

    familiarity = (

        familiarity

        .groupby(
            [
                "Team",
                "Map",
                "Year",
                "Agent",
            ],
            as_index=False,
        )

        .agg(
                **{
                    "Agent Played": ("Played", "sum")
                }
            )

    )

    print(f"[INFO] Total Records : {len(familiarity)}")

    print()

    print(familiarity.head(15))

    return familiarity

def calculate_composition_familiarity(
    dataset: pd.DataFrame,
    familiarity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add agent familiarity statistics into every composition.
    """

    print("\n" + "=" * 60)
    print("CALCULATE COMPOSITION FAMILIARITY")
    print("=" * 60)

    dataset = dataset.copy()

    familiarity_lookup = (

        familiarity

        .set_index(
            [
                "Team",
                "Map",
                "Year",
                "Agent",
            ]
        )["Agent Played"]

        .to_dict()

    )

    usage_mean = []

    usage_min = []

    low_familiarity = []

    for _, row in dataset.iterrows():

        usage = []

        for agent in row["Agent"]:

            agent_played = familiarity_lookup.get(

            (
                row["Team"],
                row["Map"],
                row["Year"],
                agent,
            ),

            0,

            )

            context_played = max(
                row["Context Played"],
                1,
            )

            usage.append(
                agent_played / context_played
            )
            

        usage_mean.append(
            round(np.mean(usage), 4)
        )

        usage_min.append(
            round(np.min(usage), 4)
        )

        low_familiarity.append(
            sum(
                u < 0.20
                for u in usage
            )
        )

    dataset["Agent Usage Mean"] = usage_mean

    dataset["Agent Usage Min"] = usage_min

    dataset["Low Familiarity Count"] = low_familiarity

    print("[INFO] Composition Familiarity Calculated")

    print()

    print(
        dataset[
            [
                "Agent",
                "Agent Usage Mean",
                "Agent Usage Min",
                "Low Familiarity Count",
            ]
        ].head(10)
    )

    return dataset

def calculate_agent_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical statistics
    for every agent.
    """

    print("\n" + "=" * 60)
    print("CALCULATE AGENT STATISTICS")
    print("=" * 60)

    records = []

    # -------------------------
    # Build agent table
    # -------------------------

    for _, row in df.iterrows():

        for agent in row["Agent"]:

            records.append({

                "Agent": agent,

                "Winrate": row["Winrate"]

            })

    agent_df = pd.DataFrame(records)

    statistics = (

        agent_df

        .groupby("Agent")

        .agg(

            Played=("Agent", "count"),

            Agent_WR=("Winrate", "mean"),

        )

        .round(4)

        .reset_index()

    )

    print(statistics.head())

    print()

    print(

        f"[INFO] Total Agent : {len(statistics)}"

    )

    return statistics

def calculate_composition_statistics(
    df: pd.DataFrame,
    agent_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical agent statistics
    into each composition.

    Features
    --------
    Agent WR Mean

    Agent WR Min

    Agent WR Max

    Agent Played Mean
    """

    print("\n" + "=" * 60)
    print("CALCULATE COMPOSITION STATISTICS")
    print("=" * 60)

    agent_lookup = (

        agent_statistics

        .set_index("Agent")

        .to_dict("index")

    )

    agent_wr_mean = []

    agent_wr_min = []

    agent_wr_max = []

    agent_played_mean = []

    for composition in df["Agent"]:

        wr = []

        played = []

        for agent in composition:

            stats = agent_lookup[agent]

            wr.append(
                stats["Agent_WR"]
            )

            played.append(
                stats["Played"]
            )

        agent_wr_mean.append(
            round(np.mean(wr), 4)
        )

        agent_wr_min.append(
            round(np.min(wr), 4)
        )

        agent_wr_max.append(
            round(np.max(wr), 4)
        )

        agent_played_mean.append(
            round(np.mean(played), 2)
        )

    df["Agent WR Mean"] = agent_wr_mean

    df["Agent WR Min"] = agent_wr_min

    df["Agent WR Max"] = agent_wr_max

    df["Agent Played Mean"] = agent_played_mean

    print("[INFO] Composition Statistics Added")

    print()

    print(
        df[
            [
                "Agent",
                "Agent WR Mean",
                "Agent WR Min",
                "Agent WR Max",
                "Agent Played Mean",
            ]
        ].head(10)
    )

    return df

OUTPUT_DATASET = (

    DATASET_DIR

    /

    "valorant_dataset_team_v2.csv"

)

def save_dataset(
    dataset: pd.DataFrame
):
    """
    Save Team Dataset V2.
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

    validate_training_dataset(dataset)
    
    report_dataset(dataset)

    dataset = calculate_winrate(dataset)
    
    dataset = calculate_context_statistics(dataset)

    dataset = calculate_usage_statistics(dataset)

    dataset = calculate_reliability(dataset)
    
    dataset = calculate_role_statistics(dataset)

    dataset = calculate_role_pattern_statistics(dataset)
    
    dataset = calculate_historical_statistics(dataset)
    
    agent_statistics = calculate_agent_statistics(dataset)
    
    agent_familiarity = calculate_agent_familiarity(dataset)
    
    dataset = calculate_composition_statistics(dataset, agent_statistics)
    
    dataset = calculate_composition_familiarity(dataset, agent_familiarity)
    
    print()

    print("=" * 60)
    print("MAP WR SAMPLE")
    print("=" * 60)

    sample = dataset[
        dataset["Team"] == "Paper Rex"
    ][
        [
            "Map",
            "Winrate",
            "Team Overall WR",
            "Team Map WR",
        ]
    ]

    print(
        sample
        .sort_values("Map")
        .head(20)
    )
    
    print()

    print(
        dataset[
            [
                "Team",
                "Map",
                "Winrate",
                "Team Overall WR",
                "Team Map WR",
            ]
        ].head(15)
    )

    dataset = dataset.sort_values(
        by=[
            "Year",
            "Team",
            "Map",
        ]
    )
    
    save_dataset(
        dataset
    )
    


if __name__ == "__main__":
    main()