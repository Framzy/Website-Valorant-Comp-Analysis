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
    find_exact_composition,
    prepare_inference_row,
    normalize_agents,
)
from backend.ml.team.feature_encoder import encode_features

class TeamPredictionService:
    """Orchestrates Team Prediction V2 inference."""

    def __init__(
        self,
        dataset_path=TEAM_DATASET_PATH,
        model_path=MODEL_TEAM_DIR / "team_model_v2.joblib",
        encoders_path=MODEL_TEAM_DIR / "encoders.joblib",
        feature_names_path=MODEL_TEAM_DIR / "feature_names.joblib",
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

        self._prepare_dataset()
        self._validate_artifacts()

    def _prepare_dataset(self) -> None:
        """Normalize the dataset Agent column once at service startup."""
        if "Agent" not in self.dataset.columns:
            raise ValueError("Team dataset is missing the 'Agent' column.")

        if self.dataset["Agent"].dtype == object:
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

        prediction = float(self.model.predict(encoded)[0])

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
            },
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
            "inference": {
                "feature_count": int(encoded.shape[1]),
            },
        }
