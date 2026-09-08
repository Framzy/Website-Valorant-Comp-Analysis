from pathlib import Path
import ast
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

from training.general.config import GENERAL_DATASET_PATH

from training.general.constants import (
    AGENT_ROLES,
    PLAYSTYLE_MAP,
)


# ============================================================
# DATASET
# ============================================================

def load_dataset(path=GENERAL_DATASET_PATH):
    """
    Load General dataset.
    """

    df = pd.read_csv(path)

    df["Agent"] = df["Agent"].apply(parse_agents)

    return df


# ============================================================
# HELPERS
# ============================================================

def parse_agents(value):
    """
    Convert stored Agent representation into sorted list.
    """

    if isinstance(value, list):
        agents = value

    elif isinstance(value, str):
        try:
            agents = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            agents = value.split("|")

    else:
        agents = []

    return sorted(
        str(agent).strip().lower()
        for agent in agents
    )


def build_composition_key(agents):
    """
    Build normalized composition key.
    """

    return "|".join(
        sorted(
            agent.strip().lower()
            for agent in agents
        )
    )


def calculate_winrate(row):
    """
    Calculate historical winrate.
    """

    played = row["Total Maps Played"]

    if played <= 0:
        return 0.0

    return (
        row["Total Wins By Map"]
        / played
    )


def calculate_pick_rate(
    row,
    context_maps,
):
    """
    Calculate composition pick rate
    within Map + Year context.
    """

    if context_maps <= 0:
        return 0.0

    return (
        row["Total Maps Played"]
        / context_maps
    )


# ============================================================
# ROLE / PLAYSTYLE
# ============================================================

def get_role_pattern(agents):
    """
    Determine composition role pattern.
    """

    counts = {
        "D": 0,
        "I": 0,
        "C": 0,
        "S": 0,
    }

    unknown_agents = []

    for agent in agents:

        role = AGENT_ROLES.get(agent)

        if role is None:
            unknown_agents.append(agent)
            continue

        counts[role] += 1

    if unknown_agents:
        raise ValueError(
            f"Unknown agent role: {unknown_agents}"
        )

    total = sum(counts.values())

    if total != 5:
        raise ValueError(
            f"Composition must contain 5 valid agents. "
            f"Found: {total}"
        )

    return (
        f"{counts['D']}D-"
        f"{counts['I']}I-"
        f"{counts['C']}C-"
        f"{counts['S']}S"
    )


def get_playstyle(agents):
    """
    Determine playstyle from role pattern.
    """

    pattern = get_role_pattern(agents)

    return {
        "pattern": pattern,
        "name": PLAYSTYLE_MAP.get(
            pattern,
            "UNCLASSIFIED",
        ),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(df):
    """
    Add reusable analysis columns.
    """

    df = df.copy()

    df["Winrate"] = (
        df["Total Wins By Map"]
        / df["Total Maps Played"]
    )

    context_maps = (
        df
        .groupby(["Map", "Year"])["Total Maps Played"]
        .transform("sum")
    )

    df["Pick Rate"] = (
        df["Total Maps Played"]
        / context_maps
    )

    playstyles = df["Agent"].apply(get_playstyle)

    df["Role Pattern"] = playstyles.apply(
        lambda x: x["pattern"]
    )

    df["Playstyle"] = playstyles.apply(
        lambda x: x["name"]
    )

    return df


# ============================================================
# EXACT LOOKUP
# ============================================================

def find_exact_composition(
    df,
    map_name,
    year,
    agents,
):
    """
    Find exact historical composition.
    """

    composition_key = build_composition_key(
        agents
    )

    result = df[
        (df["Map"].str.lower() == map_name.lower())
        & (df["Year"] == year)
        & (
            df["Composition Key"]
            == composition_key
        )
    ]

    if result.empty:
        return None

    row = result.iloc[0]

    return {
        "found": True,
        "map": row["Map"],
        "year": int(row["Year"]),
        "agents": row["Agent"],
        "role_pattern": row["Role Pattern"],
        "playstyle": row["Playstyle"],
        "total_maps": int(
            row["Total Maps Played"]
        ),
        "winrate": float(
            row["Winrate"]
        ),
        "pick_rate": float(
            row["Pick Rate"]
        ),
    }


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendations(
    df,
    limit=3,
    map_name=None,
    year=None,
    playstyle=None,
):
    """
    Recommend historical compositions.

    Ranking priority:
        1. Pick Rate DESC
        2. Total Maps Played DESC

    Filtering scope is determined by the
    provided parameters.
    """

    candidates = df.copy()

    # --------------------------------------------------------
    # MAP FILTER
    # --------------------------------------------------------

    if map_name is not None:

        candidates = candidates[
            candidates["Map"].str.lower()
            == map_name.lower()
        ]

    # --------------------------------------------------------
    # YEAR FILTER
    # --------------------------------------------------------

    if year is not None:

        candidates = candidates[
            candidates["Year"] == year
        ]

    # --------------------------------------------------------
    # PLAYSTYLE FILTER
    # --------------------------------------------------------

    if playstyle is not None:

        candidates = candidates[
            candidates["Playstyle"] == playstyle
        ]

    if candidates.empty:
        return []

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        by=[
            "Pick Rate",
            "Total Maps Played",
        ],
        ascending=[
            False,
            False,
        ],
    )

    recommendations = []

    for rank, (_, row) in enumerate(
        candidates.head(limit).iterrows(),
        start=1,
    ):

        recommendations.append({

            "rank": rank,

            "map": row["Map"],

            "year": int(
                row["Year"]
            ),

            "agents": row["Agent"],

            "role_pattern": row[
                "Role Pattern"
            ],

            "playstyle": row[
                "Playstyle"
            ],

            "total_maps": int(
                row["Total Maps Played"]
            ),

            "pick_rate": float(
                row["Pick Rate"]
            ),

            "winrate": float(
                row["Winrate"]
            ),

        })

    return recommendations


# ============================================================
# FALLBACK RECOMMENDATIONS
# ============================================================

def get_fallback_recommendations(
    df,
    map_name,
    year,
    playstyle,
    limit=3,
):
    """
    Find recommendations using fallback hierarchy.

    Priority:

        MAP + YEAR + PLAYSTYLE
        MAP + YEAR
        MAP
        GLOBAL

    UNCLASSIFIED skips the playstyle level.
    """

    # ========================================================
    # 1. MAP + YEAR + PLAYSTYLE
    # ========================================================

    if playstyle != "UNCLASSIFIED":

        recommendations = get_recommendations(
            df=df,
            map_name=map_name,
            year=year,
            playstyle=playstyle,
            limit=limit,
        )

        if recommendations:

            return (
                recommendations,
                "MAP_YEAR_PLAYSTYLE",
            )

    # ========================================================
    # 2. MAP + YEAR
    # ========================================================

    recommendations = get_recommendations(
        df=df,
        map_name=map_name,
        year=year,
        limit=limit,
    )

    if recommendations:

        return (
            recommendations,
            "MAP_YEAR",
        )

    # ========================================================
    # 3. MAP
    # ========================================================

    recommendations = get_recommendations(
        df=df,
        map_name=map_name,
        limit=limit,
    )

    if recommendations:

        return (
            recommendations,
            "MAP",
        )

    # ========================================================
    # 4. GLOBAL
    # ========================================================

    recommendations = get_recommendations(
        df=df,
        limit=limit,
    )

    if recommendations:

        return (
            recommendations,
            "GLOBAL",
        )

    # ========================================================
    # NO DATA
    # ========================================================

    return [], None

# ============================================================
# MAIN SERVICE
# ============================================================

def analyze_composition(
    map_name,
    year,
    agents,
    dataset=None,
    recommendation_limit=3,
):
    """
    Analyze a General composition.

    Flow:

        Input
          ↓
        Playstyle
          ↓
        Exact Historical Lookup
          ↓
        Map + Year + Playstyle
          ↓
        Popularity Recommendation
    """

    if dataset is None:
        dataset = load_dataset()

    df = prepare_dataset(dataset)

    normalized_agents = sorted(
        agent.strip().lower()
        for agent in agents
    )

    if len(normalized_agents) != 5:
        raise ValueError(
            "Composition must contain exactly 5 agents."
        )

    if len(set(normalized_agents)) != 5:
        raise ValueError(
            "Composition cannot contain duplicate agents."
        )

    playstyle = get_playstyle(
        normalized_agents
    )

    historical = find_exact_composition(
        df=df,
        map_name=map_name,
        year=year,
        agents=normalized_agents,
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    fallback = historical is None

    if historical is not None:

        # ----------------------------------------------------
        # Exact composition exists.
        # Recommend compositions from the same
        # Map + Year + Playstyle context.
        # ----------------------------------------------------

        recommendations = get_recommendations(
            df=df,
            map_name=map_name,
            year=year,
            playstyle=playstyle["name"],
            limit=recommendation_limit,
        )

        recommendation_source = (
            "MAP_YEAR_PLAYSTYLE"
        )

    else:

        # ----------------------------------------------------
        # Exact composition does not exist.
        # Use fallback hierarchy.
        # ----------------------------------------------------

        (
            recommendations,
            recommendation_source,
        ) = get_fallback_recommendations(
            df=df,
            map_name=map_name,
            year=year,
            playstyle=playstyle["name"],
            limit=recommendation_limit,
        )

    return {
        "input": {
            "map": map_name,
            "year": int(year),
            "agents": normalized_agents,
        },

        "playstyle": {
            "name": playstyle["name"],
            "pattern": playstyle["pattern"],
        },

        "historical": (
            historical
            if historical is not None
            else {
                "found": False,
                "map": map_name,
                "year": int(year),
                "agents": normalized_agents,
            }
        ),

        "recommendations": recommendations,

        "fallback": fallback,

        "fallback_source": (
            recommendation_source
            if fallback
            else None
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERAL ANALYSIS SERVICE TEST")
    print("=" * 60)

    test_cases = [

        {
            "name": "EXACT COMPOSITION",
            "map_name": "Abyss",
            "year": 2024,
            "agents": [
                "cypher",
                "jett",
                "kayo",
                "omen",
                "sova",
            ],
            "expected_found": True,
            "expected_fallback": False,
        },

        {
            "name": "NO EXACT COMPOSITION",
            "map_name": "Abyss",
            "year": 2024,
            "agents": [
                "breach",
                "fade",
                "jett",
                "kayo",
                "omen",
            ],
            "expected_found": False,
            "expected_fallback": True,
        },
        
        {
            "name": "FALSE COMPOSITION",
            "map_name": "Abyss",
            "year": 2027,
            "agents": [
                "brimstone",
                "clove",
                "miks",
                "viper",
                "omen",
            ],
            "expected_found": False,
            "expected_fallback": True,
        },

    ]

    for index, test in enumerate(test_cases, start=1):

        print("\n")
        print("=" * 60)
        print(f"TEST CASE {index}: {test['name']}")
        print("=" * 60)

        result = analyze_composition(
            map_name=test["map_name"],
            year=test["year"],
            agents=test["agents"],
        )

        print("\nINPUT")
        print(result["input"])

        print("\nPLAYSTYLE")
        print(result["playstyle"])

        print("\nHISTORICAL")
        print(result["historical"])

        print("\nRECOMMENDATIONS")

        for recommendation in result["recommendations"]:
            print(recommendation)

        print("\nFALLBACK")
        print(result["fallback"])

        print("\nSOURCE")
        print(result["fallback_source"])
        
        historical_found = result["historical"]["found"]
        fallback = result["fallback"]

        print("\nVALIDATION")

        if historical_found == test["expected_found"]:
            print("[PASS] Historical lookup")
        else:
            print("[FAIL] Historical lookup")

        if fallback == test["expected_fallback"]:
            print("[PASS] Fallback behavior")
        else:
            print("[FAIL] Fallback behavior")
            