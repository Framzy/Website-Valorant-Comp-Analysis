"""
Team Prediction V2 Service
==========================

Application-level orchestration only.

Flow:
    application input
        -> validation
        -> exact historical composition lookup
        -> historical row OR inference feature builder
        -> existing feature_engineering.encode_features()
        -> trained Team V2 model
        -> prediction

This module intentionally contains NO Flask/API route logic.
"""

from __future__ import annotations

import ast
import json
import math
import joblib
import pandas as pd

from pathlib import Path
from typing import Iterable

from backend.config import (
    TEAM_DATASET_PATH,
    MODEL_TEAM_DIR,
)

from backend.constants import (
    EXPECTED_FEATURE_COUNT,
    AGENT_ROLE_MAP,
)

from backend.ml.team.inference_feature_builder import (
    calculate_composition_strength,
    find_exact_composition,
    prepare_inference_row,
    normalize_agents,
)
from backend.ml.team.feature_encoder import encode_features
from backend.ml.team.comp_strength_hierarchy import strength_for_new_row

# Used only if metadata.json has no "blend" section (older model artifacts).
DEFAULT_BLEND = {
    "prior_map_weight": 0.3,
    "lambda": 0.5,
    "deviation_cap": 0.15,
    "range_z": 1.28,
    "range_prior_maps": 10,
}

EVIDENCE_NOTES = {
    "low": "Data historis terbatas. Perlakukan sebagai perkiraan kasar.",
    "medium": "Data historis cukup untuk perkiraan, bukan kepastian.",
    "high": "Data historis relatif kuat. Tetap merupakan perkiraan.",
}

NEVER_OBSERVED_NOTE = (
    "Role pattern ini belum pernah dipilih tim pro manapun di data turnamen. "
    "Prediksi diberi penurunan wajar (dibatasi, bukan radikal) karena tim pro "
    "tidak memilih role pattern secara acak."
)

# Used only if metadata.json has no "pattern_adjust" section.
DEFAULT_PATTERN_ADJUST = {
    "common_share": 0.10,
    "common_pts": 0.02,
    "span": 0.05,
    "never_pts": 0.08,
    "max_pts": 0.10,
    "team_novelty_k": 3,
}


def prevalence_label(share: float) -> str:
    """Plain-language rarity of a role pattern across all pro maps."""
    if share <= 0:
        return "belum pernah dipilih tim pro manapun"
    if share < 0.01:
        return "sangat jarang dipakai tim pro"
    if share < 0.10:
        return "jarang dipakai tim pro"
    return "umum dipakai tim pro"


def pattern_penalty(adoption: dict, cfg: dict) -> tuple[float, str]:
    """
    Bounded penalty (win-probability points, >= 0) for an unusual role pattern.

    Follows the pro-scene logic in config.PATTERN_ADJUST:
      pattern common in pro play  -> tiny penalty even if this team never used it
      pattern rare in pro play    -> larger, scales with rarity (log scale)
      pattern nobody ever played  -> never_pts
    and it fades toward 0 the more this team has played the pattern itself.
    """
    share = float(adoption["global_share"])
    team_maps = int(adoption["team_maps"])

    if adoption["global_maps"] == 0:
        base, tier = cfg["never_pts"], "never_played_by_any_team"
    elif share >= cfg["common_share"]:
        base, tier = cfg["common_pts"], "common_pattern"
    else:
        rarity = min(1.0, max(0.0, (-math.log10(share) - 1.0) / 3.0))
        base = cfg["common_pts"] + cfg["span"] * rarity
        tier = "rare_pattern"

    k = float(cfg["team_novelty_k"])
    penalty = min(base * k / (k + team_maps), cfg["max_pts"])

    if team_maps > 0 and penalty < 0.005:
        tier = "team_regular_pattern"
    elif team_maps == 0 and tier != "never_played_by_any_team":
        tier = "team_new_" + tier
    return penalty, tier


class TeamPredictionService:
    """Orchestrates Team Prediction V2 inference."""

    def __init__(
        self,
        dataset_path=TEAM_DATASET_PATH,
        model_path=MODEL_TEAM_DIR / "team_model_v2.joblib",
        encoders_path=MODEL_TEAM_DIR / "encoders.joblib",
        feature_names_path=MODEL_TEAM_DIR / "feature_names.joblib",
        metadata_path=MODEL_TEAM_DIR / "metadata.json",
    ):
        # Load once when the service is created.
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)
        self.feature_names_path = Path(feature_names_path)

        self.dataset = pd.read_csv(self.dataset_path)
        self.model = joblib.load(self.model_path)
        self.encoders = joblib.load(self.encoders_path)
        self.feature_names = joblib.load(self.feature_names_path)

        self.blend = self._load_blend(Path(metadata_path))
        self.pattern_adjust = self._load_section(
            Path(metadata_path), "pattern_adjust", DEFAULT_PATTERN_ADJUST
        )

        self._prepare_dataset()
        self._validate_artifacts()

    @staticmethod
    def _load_section(metadata_path: Path, key: str, default: dict) -> dict:
        """Read one parameter section written by training (metadata.json)."""
        section = dict(default)
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                section.update(json.load(file).get(key, {}))
        except (OSError, ValueError):
            pass
        return section

    @staticmethod
    def _load_blend(metadata_path: Path) -> dict:
        """Read blend parameters written by training (single source of truth)."""
        blend = dict(DEFAULT_BLEND)
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                blend.update(json.load(file).get("blend", {}))
        except (OSError, ValueError):
            pass
        return blend

    def _evidence(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
        strength: dict,
    ) -> dict:
        """
        How much historical data supports this prediction.

        `strength` is the dict already returned by calculate_composition_strength
        (Tahap 4), so evidence counts and the hierarchy behind Composition
        Strength always agree -- both pool across years the same way.
        """
        ds = self.dataset
        maps = "Total Maps Played"

        team_map = ds[(ds["Team"] == team) & (ds["Map"] == map_name)]
        context = team_map[team_map["Year"] == year]

        team_map_played = int(team_map[maps].sum())
        context_played = int(context[maps].sum())

        ev = strength["evidence"]
        composition_played = ev["exact_played"]           # pooled across years
        team_pattern_played = ev["team_pattern_played"]
        role_pattern_played = ev["global_pattern_played"]  # this playstyle, anywhere
        never_observed = strength["role_pattern_never_observed"]

        if never_observed:
            level = "low"
        elif team_map_played < 5:
            level = "low"
        elif composition_played == 0:
            level = "low" if team_pattern_played < 5 else "medium"
        elif composition_played >= 3 and team_map_played >= 10:
            level = "high"
        else:
            level = "medium"

        note = NEVER_OBSERVED_NOTE if never_observed else EVIDENCE_NOTES[level]

        levels = strength["levels"]
        rarity_discount = round(
            levels["rarity_adjusted_anchor"] - levels["global_wr"], 4
        )

        return {
            "level": level,
            "team_map_played": team_map_played,
            "context_played": context_played,
            "composition_played": composition_played,
            "team_pattern_played": team_pattern_played,
            "role_pattern_played": role_pattern_played,
            "role_pattern_never_observed": never_observed,
            "role_pattern_rarity_discount": rarity_discount,
            "meta_prevalence": self._meta_prevalence(strength["pattern_adoption"]),
            "note": note,
        }

    @staticmethod
    def _meta_prevalence(adoption: dict) -> dict:
        """How common this role pattern is in pro play (counts only)."""
        return {
            "role_pattern_share_pct": round(adoption["global_share"] * 100, 2),
            "label": prevalence_label(adoption["global_share"]),
            "rank": adoption["global_rank"],
            "n_patterns": adoption["n_patterns"],
            "team_maps_with_pattern": adoption["team_maps"],
        }

    def _blend_prediction(
        self,
        raw: float,
        prepared: pd.DataFrame,
        team_map_played: int,
    ) -> tuple[float, float, dict]:
        """
        final = prior + lambda * clip(model - prior, -cap, +cap)

        The model can only nudge the team-strength prior, so one agent
        swap cannot move the result by more than lambda * cap.
        """
        b = self.blend
        row = prepared.iloc[0]

        w = float(b["prior_map_weight"])
        prior = w * float(row["Team Map WR"]) + (1 - w) * float(row["Team Overall WR"])

        cap = float(b["deviation_cap"])
        deviation = max(-cap, min(cap, raw - prior))
        final = max(0.0, min(1.0, prior + float(b["lambda"]) * deviation))

        se = math.sqrt(
            final * (1 - final) / (team_map_played + float(b["range_prior_maps"]))
        )
        half = float(b["range_z"]) * se
        rng = {
            "low": round(max(0.0, final - half), 4),
            "high": round(min(1.0, final + half), 4),
            "level": 0.8,
        }
        return final, prior, rng

    def _prepare_dataset(self) -> None:
        """Normalize the dataset Agent column once at service startup."""
        if "Agent" not in self.dataset.columns:
            raise ValueError("Team dataset is missing the 'Agent' column.")

        # No dtype check: newer pandas reads text columns as "str", not
        # "object", which would silently skip the conversion.
        self.dataset["Agent"] = self.dataset["Agent"].apply(
            lambda value: ast.literal_eval(value)
            if isinstance(value, str)
            else value
        )

    def _validate_artifacts(self) -> None:
        """Fail early if the model artifacts do not match V2's 116-feature contract."""
        model_features = getattr(self.model, "n_features_in_", None)
        feature_name_count = len(self.feature_names)

        if model_features != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Unexpected model feature count: {model_features}. "
                f"Expected {EXPECTED_FEATURE_COUNT}."
            )

        if feature_name_count != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Unexpected feature_names count: {feature_name_count}. "
                f"Expected {EXPECTED_FEATURE_COUNT}."
            )

        if model_features != feature_name_count:
            raise ValueError(
                "Model and feature_names are incompatible."
            )

        required_encoders = {"team", "map", "year", "agent"}
        missing = required_encoders - set(self.encoders.keys())

        if missing:
            raise ValueError(
                f"Missing inference encoders: {sorted(missing)}"
            )

    @staticmethod
    def _validate_input(
        team: str,
        map_name: str,
        year: int,
        agents: Iterable[str],
    ) -> tuple[int, list[str]]:
        """Validate application-level input and return normalized year/agents."""
        if not isinstance(team, str) or not team.strip():
            raise ValueError("Team is required.")

        if not isinstance(map_name, str) or not map_name.strip():
            raise ValueError("Map is required.")

        if year is None:
            raise ValueError("Year is required.")

        try:
            normalized_year = int(year)
        except (TypeError, ValueError):
            raise ValueError("Year must be an integer.")

        if agents is None:
            raise ValueError("Agents are required.")

        try:
            normalized_agents = normalize_agents(agents)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

        unknown_agents = [
            agent for agent in normalized_agents
            if agent not in AGENT_ROLE_MAP
        ]

        if unknown_agents:
            raise ValueError(
                f"Unknown agents: {', '.join(unknown_agents)}"
            )

        return normalized_year, normalized_agents

    def _validate_context(
        self,
        team: str,
        map_name: str,
        year: int,
    ) -> None:
        """Ensure the selected Team + Map + Year exists in the Team V2 dataset."""
        context = self.dataset[
            (self.dataset["Team"] == team)
            & (self.dataset["Map"] == map_name)
            & (self.dataset["Year"] == year)
        ]

        if context.empty:
            raise ValueError(
                f"No Team V2 data for {team} / {map_name} / {year}."
            )

    @staticmethod
    def _build_historical_row(
        exact: pd.DataFrame,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
    ) -> pd.DataFrame:
        """
        Build the feature-engineering input from an exact historical row.

        Winrate is the ML target. It is returned only as historical metadata
        and is never used to calculate Composition Strength.
        """
        row = exact.iloc[0]

        return pd.DataFrame(
            [{
                "Team": team,
                "Map": map_name,
                "Year": year,
                "Agent": agents,
                "Role Pattern": row["Role Pattern"],
                "Duelist Count": int(row["Duelist Count"]),
                "Initiator Count": int(row["Initiator Count"]),
                "Controller Count": int(row["Controller Count"]),
                "Sentinel Count": int(row["Sentinel Count"]),
                "Team Overall WR": float(row["Team Overall WR"]),
                "Team Map WR": float(row["Team Map WR"]),
                "Composition Strength": float(row["Composition Strength"]),
            }]
        )

    def _prepare_features(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
    ) -> tuple[pd.DataFrame, bool, float | None]:
        """
        Choose the correct feature-preparation path.

        FOUND:
            Use the exact historical composition row.

        NOT FOUND:
            Use inference_feature_builder to calculate the features for
            an unseen composition from the existing Team + Map + Year context.
        """
        exact = find_exact_composition(
            self.dataset,
            team,
            map_name,
            year,
            agents,
        )

        if len(exact) > 1:
            raise ValueError(
                "Multiple rows found for the same exact composition."
            )

        if len(exact) == 1:
            row = self._build_historical_row(
                exact,
                team,
                map_name,
                year,
                agents,
            )
            historical_winrate = float(exact.iloc[0]["Winrate"])
            return row, True, historical_winrate

        row = prepare_inference_row(
            df=self.dataset,
            team=team,
            map_name=map_name,
            year=year,
            agents=agents,
            agent_role_map=AGENT_ROLE_MAP,
        )

        return row, False, None

    def get_available_teams(
        self,
        year: int,
        map_name: str,
    ) -> list[str]:
        """Return teams whose Team + Map + Year context has >= 3 maps.

        Year and Map are supplied by the shared options layer; this method
        applies the Team-specific eligibility rule only.
        """
        try:
            normalized_year = int(year)
        except (TypeError, ValueError):
            raise ValueError("Year must be an integer.")

        if not isinstance(map_name, str) or not map_name.strip():
            raise ValueError("Map is required.")

        normalized_map = map_name.strip().lower()
        context = self.dataset[
            (self.dataset["Year"] == normalized_year)
            & (self.dataset["Map"].str.lower() == normalized_map)
        ]

        if context.empty:
            raise ValueError(
                f"No Team V2 data for {map_name} / {normalized_year}."
            )

        context_played = context.groupby("Team")["Total Maps Played"].sum()
        eligible = context_played[context_played >= 3]

        if eligible.empty:
            raise ValueError(
                f"No eligible teams for {map_name} / {normalized_year}."
            )

        return sorted(str(team) for team in eligible.index if pd.notna(team))

        # ========================================================
    # BEST HISTORICAL COMPOSITION
    # ========================================================

    def _get_best_composition(
        self,
        team: str,
        map_name: str,
        year: int,
    ) -> dict:
        """
        Return the most-used historical composition
        for Team + Map + Year.

        Ranking:
            1. Total Maps Played DESC
            2. Winrate DESC
            3. Composition Strength DESC

        This is a historical reference.
        It does not mean the composition is objectively
        the strongest composition.
        """

        context = self.dataset[
            (self.dataset["Team"] == team)
            & (self.dataset["Map"] == map_name)
            & (self.dataset["Year"] == year)
        ].copy()

        if context.empty:
            return {
                "found": False,
            }

        context = context.sort_values(
            by=[
                "Total Maps Played",
                "Winrate",
                "Composition Strength",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        row = context.iloc[0]

        agents = [
            str(agent)
            for agent in row["Agent"]
        ]

        winrate = float(
            row["Winrate"]
        )

        return {
            "found": True,
            "agents": agents,
            "role_pattern": str(
                row["Role Pattern"]
            ),
            "maps_played": int(
                row["Total Maps Played"]
            ),
            "winrate": winrate,
            "percentage": round(
                winrate * 100,
                2,
            ),
            "composition_strength": round(
                float(
                    row["Composition Strength"]
                ),
                2,
            ),
        }

    def predict(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: Iterable[str],
    ) -> dict:
        """Run one Team Prediction V2 inference request."""
        year, agents = self._validate_input(
            team=team,
            map_name=map_name,
            year=year,
            agents=agents,
        )

        self._validate_context(team, map_name, year)

        best_composition = self._get_best_composition(
            team=team,
            map_name=map_name,
            year=year,
        )

        prepared, historical_found, historical_winrate = (
            self._prepare_features(
                team=team,
                map_name=map_name,
                year=year,
                agents=agents,
            )
        )

        encoded = encode_features(prepared, self.encoders)

        if encoded.shape != (1, EXPECTED_FEATURE_COUNT):
            raise ValueError(
                f"Invalid inference feature shape: {encoded.shape}. "
                f"Expected (1, {EXPECTED_FEATURE_COUNT})."
            )

        raw_prediction = float(self.model.predict(encoded)[0])

        # Single source of truth for both Composition Strength (already
        # baked into `prepared`/`encoded` above) and evidence counts, so
        # the two always agree (Tahap 4).
        strength = calculate_composition_strength(
            self.dataset, team, map_name, year, agents, AGENT_ROLE_MAP,
        )
        # Defensive: an older inference_feature_builder.py may not pass
        # `pattern_adoption` through; derive it from the same hierarchy.
        if "pattern_adoption" not in strength:
            strength["pattern_adoption"] = strength_for_new_row(
                self.dataset,
                team,
                map_name,
                strength["role_pattern"],
                "|".join(normalize_agents(agents)),
            )["pattern_adoption"]

        evidence = self._evidence(team, map_name, year, agents, strength)

        blended, prior, blended_range = self._blend_prediction(
            raw_prediction,
            prepared,
            evidence["team_map_played"],
        )

        # Role pattern adjustment (Tahap 6): bounded and reported separately.
        penalty, tier = pattern_penalty(
            strength["pattern_adoption"], self.pattern_adjust
        )
        prediction = max(0.0, blended - penalty)
        prediction_range = {
            "low": round(max(0.0, blended_range["low"] - penalty), 4),
            "high": round(max(0.0, blended_range["high"] - penalty), 4),
            "level": blended_range["level"],
        }

        return {
            "input": {
                "team": team,
                "map": map_name,
                "year": year,
                "agents": agents,
            },
            "prediction": {
                "winrate": prediction,
                "percentage": round(prediction * 100, 2),
                "model_winrate": round(raw_prediction, 4),
                "prior_winrate": round(prior, 4),
                "range": prediction_range,
                "winrate_before_adjustment": round(blended, 4),
                "pattern_adjustment": {
                    "points": round(-penalty * 100, 2),
                    "tier": tier,
                    "reason": prevalence_label(strength["pattern_adoption"]["global_share"])
                    + (
                        "; belum pernah dipakai tim ini"
                        if strength["pattern_adoption"]["team_maps"] == 0
                        else f"; dipakai tim ini {strength['pattern_adoption']['team_maps']} map"
                    ),
                },
            },
            "confidence": evidence,
            "composition": {
                "role_pattern": str(prepared.iloc[0]["Role Pattern"]),
                "duelist_count": int(prepared.iloc[0]["Duelist Count"]),
                "initiator_count": int(prepared.iloc[0]["Initiator Count"]),
                "controller_count": int(prepared.iloc[0]["Controller Count"]),
                "sentinel_count": int(prepared.iloc[0]["Sentinel Count"]),
            },
            "historical": {
                "found": historical_found,
                "winrate": historical_winrate,
                "composition_strength": round(
                    float(prepared.iloc[0]["Composition Strength"]),
                    2,
                ),
            },
            "best_composition": best_composition,
            "inference": {
                "feature_count": int(encoded.shape[1]),
            },
        }
