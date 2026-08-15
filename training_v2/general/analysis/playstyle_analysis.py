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
# ROLE CONFIGURATION
# ============================================================

AGENT_ROLE_MAP = {
    # Duelist
    "jett": "duelist",
    "neon": "duelist",
    "phoenix": "duelist",
    "raze": "duelist",
    "reyna": "duelist",
    "yoru": "duelist",
    "iso": "duelist",

    # Initiator
    "breach": "initiator",
    "fade": "initiator",
    "gekko": "initiator",
    "kayo": "initiator",
    "skye": "initiator",
    "sova": "initiator",
    "tejo": "initiator",

    # Controller
    "astra": "controller",
    "brimstone": "controller",
    "clove": "controller",
    "harbor": "controller",
    "miks": "controller",
    "omen": "controller",
    "viper": "controller",

    # Sentinel
    "chamber": "sentinel",
    "cypher": "sentinel",
    "deadlock": "sentinel",
    "killjoy": "sentinel",
    "sage": "sentinel",
    "veto": "sentinel",
    "vyse": "sentinel",
    "waylay": "sentinel",
}


ROLE_ORDER = [
    "duelist",
    "initiator",
    "controller",
    "sentinel",
]


# ============================================================
# PLAYSTYLE CONFIGURATION
# ============================================================

PLAYSTYLE_MAP = {
    "1D-2I-1C-1S": "STANDARD",
        
    "1D-1I-2C-1S": "CONTROL",
    "1D-2I-2C-0S": "CONTROL",
    "2D-1I-2C-0S": "CONTROL",
        
    "2D-1I-1C-1S": "AGGRESSIVE",
    "2D-2I-1C-0S": "AGGRESSIVE",
        
    "1D-1I-1C-2S": "UTILITY_HEAVY",
    "0D-2I-2C-1S": "UTILITY_HEAVY",
    "0D-2I-1C-2S": "UTILITY_HEAVY"
}

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 60)
    print("LOAD GENERAL DATASET")
    print("=" * 60)

    dataset = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"[INFO] Shape : {dataset.shape}"
    )

    return dataset


# ============================================================
# PARSE AGENTS
# ============================================================

def parse_agents(value):

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (
            ValueError,
            SyntaxError,
        ):
            return []

    return []


# ============================================================
# CALCULATE ROLE COUNTS
# ============================================================

def calculate_role_counts(agents):

    role_counts = {
        role: 0
        for role in ROLE_ORDER
    }

    for agent in agents:

        agent = (
            str(agent)
            .lower()
            .strip()
        )

        role = AGENT_ROLE_MAP.get(
            agent
        )

        if role is None:
            continue

        role_counts[role] += 1

    return role_counts


# ============================================================
# BUILD ROLE PATTERN
# ============================================================

def build_role_pattern(
    role_counts
):

    return (
        f"{role_counts['duelist']}D-"
        f"{role_counts['initiator']}I-"
        f"{role_counts['controller']}C-"
        f"{role_counts['sentinel']}S"
    )


# ============================================================
# CALCULATE PLAYSTYLE
# ============================================================

def calculate_playstyle(
    dataset
):

    print("\n" + "=" * 60)
    print("CALCULATE PLAYSTYLE")
    print("=" * 60)

    dataset = dataset.copy()

    role_results = []
    patterns = []
    playstyles = []

    for agents in dataset["Agent"]:

        agents = parse_agents(
            agents
        )

        role_counts = calculate_role_counts(
            agents
        )

        pattern = build_role_pattern(
            role_counts
        )

        playstyle = PLAYSTYLE_MAP.get(
            pattern,
            "UNCLASSIFIED"
        )

        role_results.append(
            role_counts
        )

        patterns.append(
            pattern
        )

        playstyles.append(
            playstyle
        )

    dataset["Role Pattern"] = patterns

    dataset["Playstyle"] = playstyles

    for role in ROLE_ORDER:

        dataset[
            f"{role.title()} Count"
        ] = [
            result[role]
            for result in role_results
        ]

    return dataset


# ============================================================
# VALIDATE PLAYSTYLE
# ============================================================

def validate_playstyle(
    dataset
):

    print("\n" + "=" * 60)
    print("VALIDATE PLAYSTYLE")
    print("=" * 60)

    role_columns = [
        "Duelist Count",
        "Initiator Count",
        "Controller Count",
        "Sentinel Count",
    ]

    role_total = (
        dataset[role_columns]
        .sum(axis=1)
    )

    invalid_role_total = (
        role_total != 5
    ).sum()

    invalid_pattern = (
        dataset["Role Pattern"]
        .isna()
    ).sum()

    print(
        f"[INFO] Invalid Role Total : "
        f"{invalid_role_total}"
    )

    print(
        f"[INFO] Invalid Pattern    : "
        f"{invalid_pattern}"
    )

    if invalid_role_total > 0:
        raise ValueError(
            "Role count does not equal 5."
        )

    if invalid_pattern > 0:
        raise ValueError(
            "Invalid role pattern detected."
        )

    print(
        "[INFO] Validation Passed"
    )


# ============================================================
# DISPLAY PATTERN DISTRIBUTION
# ============================================================

def display_pattern_distribution(
    dataset
):

    print("\n" + "=" * 60)
    print("ROLE PATTERN DISTRIBUTION")
    print("=" * 60)

    distribution = (
        dataset[
            [
                "Role Pattern",
                "Playstyle",
            ]
        ]
        .value_counts()
        .reset_index(
            name="Composition Count"
        )
    )

    print(
        distribution
        .to_string(index=False)
    )
    
    
# ============================================================
# PLAYSTYLE SUMMARY
# ============================================================

def display_playstyle_summary(dataset):

    print("\n" + "=" * 60)
    print("PLAYSTYLE SUMMARY")
    print("=" * 60)

    summary = (
        dataset
        .groupby("Playstyle")
        .agg(
            Composition_Count=(
                "Agent",
                "count",
            ),
            Total_Maps=(
                "Total Maps Played",
                "sum",
            ),
            Total_Wins=(
                "Total Wins By Map",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Aggregate Winrate
    # --------------------------------------------------------

    summary["Winrate"] = (
        summary["Total_Wins"]
        / summary["Total_Maps"]
    )

    # --------------------------------------------------------
    # Dataset Share
    # --------------------------------------------------------

    total_maps = summary["Total_Maps"].sum()

    summary["Map_Share"] = (
        summary["Total_Maps"]
        / total_maps
    )

    # --------------------------------------------------------
    # Sort by map usage
    # --------------------------------------------------------

    summary = summary.sort_values(
        "Total_Maps",
        ascending=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display = summary.copy()

    display["Winrate"] = (
        display["Winrate"] * 100
    ).round(2)

    display["Map_Share"] = (
        display["Map_Share"] * 100
    ).round(2)

    print(
        display[
            [
                "Playstyle",
                "Composition_Count",
                "Total_Maps",
                "Total_Wins",
                "Winrate",
                "Map_Share",
            ]
        ].to_string(index=False)
    )

    return summary


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def display_sample(
    dataset
):

    print("\n" + "=" * 60)
    print("PLAYSTYLE SAMPLE")
    print("=" * 60)

    columns = [
        "Map",
        "Year",
        "Agent",
        "Role Pattern",
        "Playstyle",
        "Duelist Count",
        "Initiator Count",
        "Controller Count",
        "Sentinel Count",
        "Total Maps Played",
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

    dataset = calculate_playstyle(
        dataset
    )

    validate_playstyle(
        dataset
    )

    display_pattern_distribution(
        dataset
    )

    display_playstyle_summary(
        dataset
    )

    display_sample(
        dataset
    )

if __name__ == "__main__":
    main()