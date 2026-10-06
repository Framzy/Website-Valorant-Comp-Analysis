from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from development.backend.config import GENERAL_DATASET_PATH
from development.backend.constants import (
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

    name = PLAYSTYLE_MAP.get(
        pattern,
        "UNCLASSIFIED",
    )

    match name:
        case "STANDARD":
            desc = (
                "Ini adalah gaya main paling aman dan fleksibel yang sering "
                "kamu lihat di game turnamen. Komposisinya sangat seimbang. "
                "Kamu punya Duelist untuk maju duluan, Initiator untuk cari "
                "info musuh, Controller untuk tutup pandangan (smoke), dan "
                "Sentinel untuk jaga lini belakang atau site. Cocok untuk "
                "tim yang ingin bermain rapi tanpa taktik yang terlalu "
                "aneh-aneh."
            )

        case "CONTROL":
            desc = (
                "Gaya main ini berfokus pada membuat musuh kebingungan dan "
                "mempersempit jarak pandang mereka. Dengan adanya dua smoker "
                "di tim, kamu bisa menutup banyak sudut pandang musuh "
                "sekaligus, melakukan fake site (mengecoh musuh), dan "
                "mengendalikan pergerakan peta dengan sangat leluasa. Cocok "
                "buat kamu yang suka main taktis dan menang lewat strategi, "
                "bukan cuma adu tembak."
            )

        case "AGGRESSIVE":
            desc = (
                "Ini adalah gaya main gas pol dan penuh aksi. Dengan "
                "mengandalkan dua Duelist, tim kamu punya kekuatan tempur "
                "(firepower) dan mobilitas yang sangat tinggi. Tujuannya "
                "cuma satu: serang site secepat mungkin, culik musuh, dan "
                "menangkan adu tembak sejak awal ronde. Sangat cocok untuk "
                "pemain yang suka main cepat, bar-bar, dan percaya diri "
                "dengan kemampuan aim mereka."
            )

        case "UTILITY_HEAVY":
            desc = (
                "Gaya main ini mengutamakan sabar, jebakan, dan penggunaan "
                "skill (utilitas). Karena seringkali tidak punya Duelist "
                "atau justru punya dua Sentinel, tim kamu akan bermain "
                "sangat lambat saat menyerang, namun menjadi benteng "
                "pertahanan yang mustahil ditembus saat bertahan. Kamu akan "
                "menang dengan cara memancing musuh masuk ke area jebakan "
                "atau menahan site hingga waktu habis. Cocok untuk pemain "
                "yang penyabar dan suka menyiksa mental musuh dengan setup "
                "ability."
            )

        case _:
            desc = (
                "Komposisi ini memiliki role pattern yang belum "
                "diklasifikasikan ke dalam playstyle tertentu,"
                "karena memiliki pattern yang tidak seimbang dan jarang dimainkan."
            )

    return {
        "pattern": pattern,
        "description": desc,
        "name": name,
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

    composition_key = build_composition_key(agents)

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

    if map_name is not None:
        candidates = candidates[
            candidates["Map"].str.lower()
            == map_name.lower()
        ]

    if year is not None:
        candidates = candidates[
            candidates["Year"] == year
        ]

    if playstyle is not None:
        candidates = candidates[
            candidates["Playstyle"] == playstyle
        ]

    if candidates.empty:
        return []

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
            "year": int(row["Year"]),
            "agents": row["Agent"],
            "role_pattern": row["Role Pattern"],
            "playstyle": row["Playstyle"],
            "total_maps": int(row["Total Maps Played"]),
            "pick_rate": float(row["Pick Rate"]),
            "winrate": float(row["Winrate"]),
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

    recommendations = get_recommendations(
        df=df,
        limit=limit,
    )

    if recommendations:
        return (
            recommendations,
            "GLOBAL",
        )

    return [], None


# ============================================================
# MAIN SERVICE
# ============================================================

class GeneralAnalysisService:
    """
    Production service for General Analysis.

    The dataset is loaded and prepared once when the
    service is initialized, then reused for every request.
    """

    def __init__(
        self, 
        dataset_path=GENERAL_DATASET_PATH
    ):
        self.dataset_path = Path(dataset_path)

        dataset = load_dataset(self.dataset_path)
        self.dataset = prepare_dataset(dataset)

    def analyze_composition(
        self,
        map_name,
        year,
        agents,
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
            df=self.dataset,
            map_name=map_name,
            year=year,
            agents=normalized_agents,
        )

        fallback = historical is None

        if historical is not None:
            recommendations = get_recommendations(
                df=self.dataset,
                map_name=map_name,
                year=year,
                playstyle=playstyle["name"],
                limit=recommendation_limit,
            )

            recommendation_source = (
                "MAP_YEAR_PLAYSTYLE"
            )

        else:
            (
                recommendations,
                recommendation_source,
            ) = get_fallback_recommendations(
                df=self.dataset,
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
                "description": playstyle["description"],
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
