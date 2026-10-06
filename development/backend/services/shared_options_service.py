"""
Shared Options Service
======================

Provides Year and Map options shared by General Analysis V2
and Team Prediction V2.

The options are based on the common availability of the
General V2 and Team V2 datasets.
"""

from __future__ import annotations

import pandas as pd

from development.backend.config import (
    GENERAL_DATASET_PATH,
    TEAM_DATASET_PATH,
)


class SharedOptionsService:
    """Provides shared Year and Map options for the API."""

    def __init__(
        self,
        general_dataset_path=GENERAL_DATASET_PATH,
        team_dataset_path=TEAM_DATASET_PATH,
    ):
        self.general_dataset_path = general_dataset_path
        self.team_dataset_path = team_dataset_path

        self.general_dataset = pd.read_csv(self.general_dataset_path)
        self.team_dataset = pd.read_csv(self.team_dataset_path)

        self._validate_datasets()

    def _validate_datasets(self) -> None:
        """Fail early if required shared option columns are missing."""
        required_columns = {"Year", "Map"}

        missing_general = required_columns - set(self.general_dataset.columns)
        missing_team = required_columns - set(self.team_dataset.columns)

        if missing_general:
            raise ValueError(
                f"General dataset is missing columns: {sorted(missing_general)}"
            )

        if missing_team:
            raise ValueError(
                f"Team dataset is missing columns: {sorted(missing_team)}"
            )

    def get_available_years(self) -> list[int]:
        """
        Return years available in both General V2 and Team V2 datasets.
        """
        general_years = set(
            int(year)
            for year in self.general_dataset["Year"].dropna().unique()
        )
        team_years = set(
            int(year)
            for year in self.team_dataset["Year"].dropna().unique()
        )

        return sorted(general_years & team_years)

    def get_available_maps(self, year: int) -> list[str]:
        """
        Return maps available in both datasets for the selected year.
        """
        try:
            normalized_year = int(year)
        except (TypeError, ValueError):
            raise ValueError("Year must be an integer.")

        general_maps = set(
            str(map_name)
            for map_name in self.general_dataset.loc[
                self.general_dataset["Year"] == normalized_year, "Map"
            ].dropna().unique()
        )

        team_maps = set(
            str(map_name)
            for map_name in self.team_dataset.loc[
                self.team_dataset["Year"] == normalized_year, "Map"
            ].dropna().unique()
        )

        common_maps = general_maps & team_maps

        if not common_maps:
            raise ValueError(
                f"No shared map data for year {normalized_year}."
            )

        return sorted(common_maps)
