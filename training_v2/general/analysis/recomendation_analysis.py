import ast
import pandas as pd

from training_v1.config import BASE_DIR
from training_v2.general.constants import PLAYSTYLE_MAP
from training_v2.general.constants import ROLE_MAP


DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "valorant_dataset_general_v2.csv"
)

TOP_N = 3

PRIOR_WINRATE = 0.50
PRIOR_MAPS = 5


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 60)
    print("LOAD GENERAL DATASET")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)

    print(f"[INFO] Shape : {df.shape}")

    return df


# ============================================================
# PARSE AGENT
# ============================================================

def parse_agents(value):

    if isinstance(value, list):
        return sorted(value)

    return sorted(
        ast.literal_eval(value)
    )


def get_role_pattern(agents):

    counts = {
        "D": 0,
        "I": 0,
        "C": 0,
        "S": 0,
    }

    for agent in agents:

        for role, agent_pool in ROLE_MAP.items():

            if agent in agent_pool:

                role_code = role[0].upper()

                counts[role_code] += 1

                break

    return (
        f"{counts['D']}D-"
        f"{counts['I']}I-"
        f"{counts['C']}C-"
        f"{counts['S']}S"
    )


# ============================================================
# CALCULATE PLAYSTYLE
# ============================================================

def get_playstyle(agents):

    pattern = get_role_pattern(agents)

    return PLAYSTYLE_MAP.get(
        pattern,
        "UNCLASSIFIED",
    )


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(df):

    df = df.copy()

    df["Agent"] = df["Agent"].apply(
        parse_agents
    )

    df["Role Pattern"] = df["Agent"].apply(
        get_role_pattern
    )

    df["Playstyle"] = df["Role Pattern"].map(
        PLAYSTYLE_MAP
    ).fillna("UNCLASSIFIED")

    df["Winrate"] = (
        df["Total Wins By Map"]
        / df["Total Maps Played"]
    )

    return df


# ============================================================
# CALCULATE PICK RATE
# ============================================================

def calculate_pick_rate(
    df,
    group_columns,
):

    totals = (
        df.groupby(group_columns)[
            "Total Maps Played"
        ]
        .transform("sum")
    )

    df["Pick Rate"] = (
        df["Total Maps Played"]
        / totals
    )

    return df


# ============================================================
# SMOOTHED WINRATE
# ============================================================

def calculate_smoothed_winrate(df):

    df["Smoothed Winrate"] = (
        df["Total Wins By Map"]
        + (
            PRIOR_WINRATE
            * PRIOR_MAPS
        )
    ) / (
        df["Total Maps Played"]
        + PRIOR_MAPS
    )

    return df


# ============================================================
# PREPARE STATISTICS
# ============================================================

def prepare_statistics(df):

    df = calculate_pick_rate(
        df,
        ["Map", "Year"],
    )

    df = calculate_smoothed_winrate(
        df
    )

    return df


# ============================================================
# EXACT COMPOSITION LOOKUP
# ============================================================

def find_exact_composition(
    df,
    map_name,
    year,
    agents,
):

    target_agents = sorted(agents)

    result = df[
        (df["Map"].str.lower() == map_name.lower())
        &
        (df["Year"] == year)
        &
        (
            df["Agent"].apply(
                lambda x: sorted(x)
                == target_agents
            )
        )
    ]

    return result


# ============================================================
# RANK CANDIDATES
# ============================================================

def rank_candidates(
    candidates,
    top_n=TOP_N,
):

    if candidates.empty:

        return candidates

    candidates = candidates.copy()

    candidates = candidates.sort_values(
        by=[
            "Pick Rate",
            "Total Maps Played",
            "Smoothed Winrate",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return candidates.head(top_n)


# ============================================================
# FIND RECOMMENDATIONS
# ============================================================

def find_recommendations(
    df,
    map_name,
    year,
    playstyle,
    top_n=TOP_N,
):

    # --------------------------------------------------------
    # LEVEL 1
    # Map + Year + Playstyle
    # --------------------------------------------------------

    candidates = df[
        (df["Map"].str.lower() == map_name.lower())
        &
        (df["Year"] == year)
        &
        (df["Playstyle"] == playstyle)
    ]

    if len(candidates) >= top_n:

        return (
            rank_candidates(
                candidates,
                top_n,
            ),
            "MAP_YEAR_PLAYSTYLE",
        )

    # --------------------------------------------------------
    # LEVEL 2
    # Map + Year
    # --------------------------------------------------------

    candidates = df[
        (df["Map"].str.lower() == map_name.lower())
        &
        (df["Year"] == year)
    ]

    if len(candidates) >= top_n:

        return (
            rank_candidates(
                candidates,
                top_n,
            ),
            "MAP_YEAR",
        )

    # --------------------------------------------------------
    # LEVEL 3
    # Map - all years
    # --------------------------------------------------------

    candidates = df[
        df["Map"].str.lower()
        == map_name.lower()
    ]

    if not candidates.empty:

        return (
            rank_candidates(
                candidates,
                top_n,
            ),
            "MAP_HISTORICAL",
        )

    # --------------------------------------------------------
    # LEVEL 4
    # No historical data
    # --------------------------------------------------------

    return (
        pd.DataFrame(),
        "NO_HISTORICAL_DATA",
    )


# ============================================================
# DISPLAY RECOMMENDATION
# ============================================================

def display_recommendations(
    recommendations,
    source,
):

    print("\n" + "=" * 60)
    print("HISTORICAL RECOMMENDATION")
    print("=" * 60)

    print(
        f"[INFO] Recommendation Source : {source}"
    )

    if recommendations.empty:

        print(
            "[INFO] No historical recommendation available"
        )

        return

    display = recommendations.copy()

    display["Winrate"] = (
        display["Winrate"] * 100
    ).round(2)

    display["Smoothed Winrate"] = (
        display["Smoothed Winrate"] * 100
    ).round(2)

    display["Pick Rate"] = (
        display["Pick Rate"] * 100
    ).round(2)

    display.insert(
        0,
        "Rank",
        range(
            1,
            len(display) + 1,
        ),
    )

    print(
        display[
            [
                "Rank",
                "Map",
                "Year",
                "Agent",
                "Playstyle",
                "Total Maps Played",
                "Winrate",
                "Smoothed Winrate",
                "Pick Rate",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    df = prepare_dataset(
        df
    )

    df = prepare_statistics(
        df
    )

    # --------------------------------------------------------
    # TEST CASE
    # --------------------------------------------------------

    map_name = "Abyss"
    year = 2024

    agents = [
        "cypher",
        "jett",
        "kayo",
        "omen",
        "sova",
    ]

    playstyle = get_playstyle(
        agents
    )

    print("\n" + "=" * 60)
    print("INPUT")
    print("=" * 60)

    print(f"[INFO] Map       : {map_name}")
    print(f"[INFO] Year      : {year}")
    print(f"[INFO] Agents    : {agents}")
    print(f"[INFO] Playstyle : {playstyle}")

    # --------------------------------------------------------
    # EXACT LOOKUP
    # --------------------------------------------------------

    exact = find_exact_composition(
        df,
        map_name,
        year,
        agents,
    )

    print("\n" + "=" * 60)
    print("EXACT COMPOSITION")
    print("=" * 60)

    if exact.empty:

        print(
            "[INFO] Historical Data : NOT FOUND"
        )

    else:

        row = exact.iloc[0]

        print(
            "[INFO] Historical Data : FOUND"
        )

        print(
            f"[INFO] Played          : "
            f"{row['Total Maps Played']}"
        )

        print(
            f"[INFO] Winrate         : "
            f"{row['Winrate']:.4f}"
        )

        print(
            f"[INFO] Pick Rate       : "
            f"{row['Pick Rate']:.4f}"
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    recommendations, source = (
        find_recommendations(
            df,
            map_name,
            year,
            playstyle,
        )
    )

    display_recommendations(
        recommendations,
        source,
    )


if __name__ == "__main__":
    main()