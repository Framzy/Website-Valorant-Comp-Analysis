"""
Team Prediction V2 - Inference Feature Builder
================================================

Builds the numeric/context features required by the existing Team V2
feature_engineering.py from application input:

    Team + Map + Year + 5 Agents

Important:
- This module does NOT change feature_engineering.py.
- It does NOT use Winrate as an input to Composition Strength.
- Composition Strength is rebuilt from the same components/formula used by
  build_team_dataset.py.
- For a new/unseen composition, composition-specific played maps = 0.
  Historical team/map/year pattern and agent familiarity can still contribute
  to Composition Strength.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMPOSITION_COLUMNS = ["Team", "Map", "Year", "Agent"]


def normalize_agents(agents: Iterable[str]) -> list[str]:
    """Normalize agent names and return a stable sorted composition."""
    normalized = sorted(str(agent).strip().lower() for agent in agents)

    if len(normalized) != 5:
        raise ValueError("Exactly 5 agents are required.")

    if len(set(normalized)) != 5:
        raise ValueError("Duplicate agents are not allowed.")

    return normalized


def ensure_agent_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Agent column to list representation if needed."""
    result = df.copy()

    if result["Agent"].dtype == object:
        sample = result["Agent"].dropna().iloc[0] if not result.empty else None
        if isinstance(sample, str):
            result["Agent"] = result["Agent"].apply(
                lambda value: ast.literal_eval(value)
                if isinstance(value, str)
                else value
            )

    return result


def build_role_counts(agents: list[str], agent_role_map: dict[str, str]) -> dict[str, int]:
    """Build the same four role counts used by Team V2."""
    counts = {
        "controller": 0,
        "duelist": 0,
        "initiator": 0,
        "sentinel": 0,
    }

    for agent in agents:
        if agent not in agent_role_map:
            raise ValueError(f"Unknown agent: {agent}")
        role = agent_role_map[agent]
        if role not in counts:
            raise ValueError(f"Unsupported role '{role}' for agent '{agent}'")
        counts[role] += 1

    return counts


def build_role_pattern(role_counts: dict[str, int]) -> str:
    """Build the same Role Pattern string used by Team V2."""
    return (
        f"{role_counts['duelist']}D-"
        f"{role_counts['initiator']}I-"
        f"{role_counts['controller']}C-"
        f"{role_counts['sentinel']}S"
    )


def _context_rows(df: pd.DataFrame, team: str, map_name: str, year: int) -> pd.DataFrame:
    return df[
        (df["Team"] == team)
        & (df["Map"] == map_name)
        & (df["Year"] == year)
    ].copy()


def _composition_match(agent_value, target_agents: list[str]) -> bool:
    if isinstance(agent_value, str):
        agent_value = ast.literal_eval(agent_value)
    return normalize_agents(agent_value) == target_agents


def find_exact_composition(
    df: pd.DataFrame,
    team: str,
    map_name: str,
    year: int,
    agents: list[str],
) -> pd.DataFrame:
    """Find the exact Team + Map + Year + 5-agent composition, if present."""
    target_agents = normalize_agents(agents)
    context = _context_rows(df, team, map_name, year)

    if context.empty:
        return context

    mask = context["Agent"].apply(
        lambda value: _composition_match(value, target_agents)
    )
    return context.loc[mask].copy()


def calculate_team_overall_wr(df: pd.DataFrame, team: str) -> float:
    """Match build_team_dataset.py: mean composition Winrate by Team."""
    rows = df[df["Team"] == team]
    if rows.empty:
        raise ValueError(f"No Team V2 data found for team '{team}'.")
    return round(float(rows["Winrate"].mean()), 4)


def calculate_team_map_wr(df: pd.DataFrame, team: str, map_name: str) -> float:
    """Match build_team_dataset.py: mean composition Winrate by Team + Map."""
    rows = df[(df["Team"] == team) & (df["Map"] == map_name)]
    if rows.empty:
        raise ValueError(
            f"No Team V2 data found for team '{team}' on map '{map_name}'."
        )
    return round(float(rows["Winrate"].mean()), 4)


def calculate_agent_familiarity(
    context: pd.DataFrame,
    agents: list[str],
) -> tuple[float, float]:
    """
    Match the build pipeline's Agent Usage Mean/Min logic.

    For each selected agent:
        Agent Played = sum(Total Maps Played) across context compositions
        Agent Usage = Agent Played / Context Played

    Context Played is the sum of Total Maps Played for Team + Map + Year.
    """
    context_played = float(context["Total Maps Played"].sum())
    denominator = max(context_played, 1.0)

    usage = []
    for agent in agents:
        agent_played = 0.0
        for _, row in context.iterrows():
            composition = row["Agent"]
            if isinstance(composition, str):
                composition = ast.literal_eval(composition)
            if agent in [str(a).strip().lower() for a in composition]:
                agent_played += float(row["Total Maps Played"])
        usage.append(agent_played / denominator)

    return round(float(np.mean(usage)), 4), round(float(np.min(usage)), 4)


def calculate_composition_strength(
    df: pd.DataFrame,
    team: str,
    map_name: str,
    year: int,
    agents: list[str],
    agent_role_map: dict[str, str],
) -> dict[str, float | int | str]:
    """
    Rebuild Composition Strength using the same components and weights as
    build_team_dataset.py.

    No Winrate or prediction is used by this calculation.
    """
    agents = normalize_agents(agents)
    context = _context_rows(df, team, map_name, year)

    if context.empty:
        raise ValueError(
            f"No context data found for {team} / {map_name} / {year}."
        )

    context_played = float(context["Total Maps Played"].sum())

    exact = find_exact_composition(df, team, map_name, year, agents)
    if len(exact) > 1:
        raise ValueError("Multiple rows found for the same exact composition.")

    composition_played = (
        float(exact.iloc[0]["Total Maps Played"]) if len(exact) == 1 else 0.0
    )

    # Same global max-played denominator used by build_team_dataset.py.
    global_max_played = float(df["Total Maps Played"].max())
    if global_max_played <= 0:
        reliability = 0.0
    else:
        reliability = float(
            np.log1p(composition_played) / np.log1p(global_max_played)
        )

    usage_ratio = round(
        composition_played / context_played if context_played > 0 else 0.0,
        4,
    )
    reliability = round(reliability, 4)
    effective_usage = round(usage_ratio * reliability, 4)

    role_counts = build_role_counts(agents, agent_role_map)
    role_pattern = build_role_pattern(role_counts)

    pattern_played = float(
        context.loc[context["Role Pattern"] == role_pattern, "Total Maps Played"].sum()
    )
    pattern_usage = round(
        pattern_played / context_played if context_played > 0 else 0.0,
        4,
    )

    agent_usage_mean, agent_usage_min = calculate_agent_familiarity(
        context,
        agents,
    )

    composition_strength = (
        (
            effective_usage * 0.50
            + pattern_usage * 0.25
            + agent_usage_mean * 0.15
            + agent_usage_min * 0.10
        )
        * 100
    )
    composition_strength = float(np.round(composition_strength, 2))

    return {
        "context_played": context_played,
        "composition_played": composition_played,
        "usage_ratio": usage_ratio,
        "reliability": reliability,
        "effective_usage": effective_usage,
        "pattern_played": pattern_played,
        "pattern_usage": pattern_usage,
        "agent_usage_mean": agent_usage_mean,
        "agent_usage_min": agent_usage_min,
        "composition_strength": round(composition_strength, 2),
        "role_pattern": role_pattern,
        "duelist_count": role_counts["duelist"],
        "initiator_count": role_counts["initiator"],
        "controller_count": role_counts["controller"],
        "sentinel_count": role_counts["sentinel"],
    }


def prepare_inference_row(
    df: pd.DataFrame,
    team: str,
    map_name: str,
    year: int,
    agents: Iterable[str],
    agent_role_map: dict[str, str],
) -> pd.DataFrame:
    """
    Create the one-row input expected by the existing feature_engineering.py.

    The three numeric model features are calculated here rather than copied
    from a historical target row.
    """
    agents = normalize_agents(agents)
    context = _context_rows(df, team, map_name, year)

    if context.empty:
        raise ValueError(
            f"No Team V2 context found for {team} / {map_name} / {year}."
        )

    strength = calculate_composition_strength(
        df,
        team,
        map_name,
        year,
        agents,
        agent_role_map,
    )

    role_counts = build_role_counts(agents, agent_role_map)

    row = pd.DataFrame(
        [
            {
                "Team": team,
                "Map": map_name,
                "Year": year,
                "Agent": agents,
                "Role Pattern": strength["role_pattern"],
                "Duelist Count": role_counts["duelist"],
                "Initiator Count": role_counts["initiator"],
                "Controller Count": role_counts["controller"],
                "Sentinel Count": role_counts["sentinel"],
                "Team Overall WR": calculate_team_overall_wr(df, team),
                "Team Map WR": calculate_team_map_wr(df, team, map_name),
                "Composition Strength": strength["composition_strength"],
            }
        ]
    )

    return row
