"""
Team Prediction V2 Service
==========================

Application-level orchestration only.

Flow:
    application input
        -> validation
        -> exact historical composition lookup
        -> historical row OR inference feature builder
        -> feature encoding
        -> trained Team V2 model
        -> raw prediction
        -> prior + model blend
        -> role-pattern adjustment
        -> confidence / evidence
        -> final prediction

This module intentionally contains NO Flask/API route logic.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from training.team.config import (
    AGENT_ROLE_MAP,
    TEAM_DATASET_PATH,
    MODEL_TEAM_DIR,
)

from training.team.inference_feature_builder import (
    calculate_composition_strength,
    find_exact_composition,
    prepare_inference_row,
    normalize_agents,
)

from training.team.feature_encoder import encode_features
from training.team.comp_strength_hierarchy import strength_for_new_row


EXPECTED_FEATURE_COUNT = 116


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

# Used only if metadata.json has no "blend" section.
DEFAULT_BLEND = {
    "prior_map_weight": 0.3,
    "lambda": 0.5,
    "deviation_cap": 0.15,
    "range_z": 1.28,
    "range_prior_maps": 10,
}


EVIDENCE_NOTES = {
    "low": (
        "Data historis terbatas. "
        "Perlakukan sebagai perkiraan kasar."
    ),
    "medium": (
        "Data historis cukup untuk perkiraan, "
        "bukan kepastian."
    ),
    "high": (
        "Data historis relatif kuat. "
        "Tetap merupakan perkiraan."
    ),
}


NEVER_OBSERVED_NOTE = (
    "Role pattern ini belum pernah dipilih tim pro manapun "
    "di data turnamen. Prediksi diberi penurunan wajar "
    "(dibatasi, bukan radikal) karena tim pro tidak memilih "
    "role pattern secara acak."
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


# ============================================================
# ROLE PATTERN HELPERS
# ============================================================

def prevalence_label(share: float) -> str:
    """
    Plain-language rarity of a role pattern across all pro maps.
    """

    if share <= 0:
        return "belum pernah dipilih tim pro manapun"

    if share < 0.01:
        return "sangat jarang dipakai tim pro"

    if share < 0.10:
        return "jarang dipakai tim pro"

    return "umum dipakai tim pro"


def pattern_penalty(
    adoption: dict,
    cfg: dict,
) -> tuple[float, str]:
    """
    Bounded penalty (win-probability points, >= 0)
    for an unusual role pattern.

    Logic:

        common pattern
            -> small penalty

        rare pattern
            -> larger penalty based on rarity

        never observed
            -> never_pts

    Penalty decreases as the selected team gains more
    experience with the role pattern.
    """

    share = float(adoption["global_share"])
    team_maps = int(adoption["team_maps"])

    if adoption["global_maps"] == 0:
        base = cfg["never_pts"]
        tier = "never_played_by_any_team"

    elif share >= cfg["common_share"]:
        base = cfg["common_pts"]
        tier = "common_pattern"

    else:
        rarity = min(
            1.0,
            max(
                0.0,
                (-math.log10(share) - 1.0) / 3.0,
            ),
        )

        base = (
            cfg["common_pts"]
            + cfg["span"] * rarity
        )

        tier = "rare_pattern"

    k = float(cfg["team_novelty_k"])

    penalty = min(
        base * k / (k + team_maps),
        cfg["max_pts"],
    )

    if team_maps > 0 and penalty < 0.005:
        tier = "team_regular_pattern"

    elif (
        team_maps == 0
        and tier != "never_played_by_any_team"
    ):
        tier = "team_new_" + tier

    return penalty, tier


# ============================================================
# SERVICE
# ============================================================

class TeamPredictionService:
    """
    Orchestrates Team Prediction V2 inference.
    """

    def __init__(
        self,
        dataset_path=TEAM_DATASET_PATH,
        model_path=MODEL_TEAM_DIR / "team_model_v2.joblib",
        encoders_path=MODEL_TEAM_DIR / "encoders.joblib",
        feature_names_path=MODEL_TEAM_DIR / "feature_names.joblib",
        metadata_path=MODEL_TEAM_DIR / "metadata.json",
    ):
        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)
        self.feature_names_path = Path(feature_names_path)
        self.metadata_path = Path(metadata_path)

        # ----------------------------------------------------
        # Load dataset / model / artifacts once
        # ----------------------------------------------------

        self.dataset = pd.read_csv(
            self.dataset_path
        )

        self.model = joblib.load(
            self.model_path
        )

        self.encoders = joblib.load(
            self.encoders_path
        )

        self.feature_names = joblib.load(
            self.feature_names_path
        )

        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        self.blend = self._load_blend(
            self.metadata_path
        )

        self.pattern_adjust = self._load_section(
            self.metadata_path,
            "pattern_adjust",
            DEFAULT_PATTERN_ADJUST,
        )

        # ----------------------------------------------------
        # Prepare / validate
        # ----------------------------------------------------

        self._prepare_dataset()
        self._validate_artifacts()

    # ========================================================
    # METADATA
    # ========================================================

    @staticmethod
    def _load_section(
        metadata_path: Path,
        key: str,
        default: dict,
    ) -> dict:
        """
        Read one parameter section from metadata.json.

        Falls back to default values when:
        - file does not exist
        - JSON is invalid
        - section does not exist
        """

        section = dict(default)

        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                section.update(
                    json.load(file).get(
                        key,
                        {},
                    )
                )

        except (
            OSError,
            ValueError,
        ):
            pass

        return section

    @staticmethod
    def _load_blend(
        metadata_path: Path,
    ) -> dict:
        """
        Read blend parameters written by training.

        metadata.json is the single source of truth.
        """

        blend = dict(DEFAULT_BLEND)

        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                blend.update(
                    json.load(file).get(
                        "blend",
                        {},
                    )
                )

        except (
            OSError,
            ValueError,
        ):
            pass

        return blend

    # ========================================================
    # EVIDENCE / CONFIDENCE
    # ========================================================

    def _evidence(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
        strength: dict,
    ) -> dict:
        """
        Determine how much historical data supports
        the prediction.
        """

        ds = self.dataset
        maps = "Total Maps Played"

        team_map = ds[
            (ds["Team"] == team)
            & (ds["Map"] == map_name)
        ]

        context = team_map[
            team_map["Year"] == year
        ]

        team_map_played = int(
            team_map[maps].sum()
        )

        context_played = int(
            context[maps].sum()
        )

        ev = strength["evidence"]

        composition_played = ev[
            "exact_played"
        ]

        team_pattern_played = ev[
            "team_pattern_played"
        ]

        role_pattern_played = ev[
            "global_pattern_played"
        ]

        never_observed = strength[
            "role_pattern_never_observed"
        ]

        # ----------------------------------------------------
        # Confidence level
        # ----------------------------------------------------

        if never_observed:
            level = "low"

        elif team_map_played < 5:
            level = "low"

        elif composition_played == 0:
            if team_pattern_played < 5:
                level = "low"
            else:
                level = "medium"

        elif (
            composition_played >= 3
            and team_map_played >= 10
        ):
            level = "high"

        else:
            level = "medium"

        # ----------------------------------------------------
        # Note
        # ----------------------------------------------------

        note = (
            NEVER_OBSERVED_NOTE
            if never_observed
            else EVIDENCE_NOTES[level]
        )

        # ----------------------------------------------------
        # Rarity adjustment information
        # ----------------------------------------------------

        levels = strength["levels"]

        rarity_discount = round(
            levels["rarity_adjusted_anchor"]
            - levels["global_wr"],
            4,
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
            "meta_prevalence": self._meta_prevalence(
                strength["pattern_adoption"]
            ),
            "note": note,
        }

    @staticmethod
    def _meta_prevalence(
        adoption: dict,
    ) -> dict:
        """
        Describe how common the role pattern is
        in pro play.
        """

        return {
            "role_pattern_share_pct": round(
                adoption["global_share"] * 100,
                2,
            ),
            "label": prevalence_label(
                adoption["global_share"]
            ),
            "rank": adoption["global_rank"],
            "n_patterns": adoption["n_patterns"],
            "team_maps_with_pattern": adoption["team_maps"],
        }

    # ========================================================
    # PREDICTION BLENDING
    # ========================================================

    def _blend_prediction(
        self,
        raw: float,
        prepared: pd.DataFrame,
        team_map_played: int,
    ) -> tuple[float, float, dict]:
        """
        Blend the model prediction with the team prior.

        final =
            prior
            + lambda * clip(
                model - prior,
                -cap,
                +cap
            )

        This prevents a sparse composition from producing
        an excessively large prediction change.
        """

        blend = self.blend
        row = prepared.iloc[0]

        # ----------------------------------------------------
        # Prior
        # ----------------------------------------------------

        prior_map_weight = float(
            blend["prior_map_weight"]
        )

        prior = (
            prior_map_weight
            * float(row["Team Map WR"])
            + (
                1 - prior_map_weight
            )
            * float(row["Team Overall WR"])
        )

        # ----------------------------------------------------
        # Model deviation
        # ----------------------------------------------------

        deviation_cap = float(
            blend["deviation_cap"]
        )

        deviation = max(
            -deviation_cap,
            min(
                deviation_cap,
                raw - prior,
            ),
        )

        final = (
            prior
            + float(blend["lambda"])
            * deviation
        )

        final = max(
            0.0,
            min(
                1.0,
                final,
            ),
        )

        # ----------------------------------------------------
        # Prediction range
        # ----------------------------------------------------

        se = math.sqrt(
            final * (1 - final)
            / (
                team_map_played
                + float(
                    blend["range_prior_maps"]
                )
            )
        )

        half = (
            float(blend["range_z"])
            * se
        )

        prediction_range = {
            "low": round(
                max(
                    0.0,
                    final - half,
                ),
                4,
            ),
            "high": round(
                min(
                    1.0,
                    final + half,
                ),
                4,
            ),
            "level": 0.8,
        }

        return (
            final,
            prior,
            prediction_range,
        )

    # ========================================================
    # DATASET PREPARATION
    # ========================================================

    def _prepare_dataset(self) -> None:
        """
        Normalize the dataset Agent column once
        at service startup.
        """

        if "Agent" not in self.dataset.columns:
            raise ValueError(
                "Team dataset is missing "
                "the 'Agent' column."
            )

        # Do not rely on pandas dtype here.
        #
        # Newer pandas versions can represent text columns
        # differently, so always normalize string values.
        self.dataset["Agent"] = (
            self.dataset["Agent"].apply(
                lambda value:
                    ast.literal_eval(value)
                    if isinstance(value, str)
                    else value
            )
        )

    # ========================================================
    # ARTIFACT VALIDATION
    # ========================================================

    def _validate_artifacts(self) -> None:
        """
        Fail early if model artifacts do not match
        the Team V2 116-feature contract.
        """

        model_features = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        feature_name_count = len(
            self.feature_names
        )

        # ----------------------------------------------------
        # Model feature count
        # ----------------------------------------------------

        if model_features != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Unexpected model feature count: "
                f"{model_features}. "
                f"Expected "
                f"{EXPECTED_FEATURE_COUNT}."
            )

        # ----------------------------------------------------
        # Feature names count
        # ----------------------------------------------------

        if feature_name_count != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Unexpected feature_names count: "
                f"{feature_name_count}. "
                f"Expected "
                f"{EXPECTED_FEATURE_COUNT}."
            )

        # ----------------------------------------------------
        # Model vs feature names
        # ----------------------------------------------------

        if (
            model_features
            != feature_name_count
        ):
            raise ValueError(
                "Model and feature_names "
                "are incompatible."
            )

        # ----------------------------------------------------
        # Required encoders
        # ----------------------------------------------------

        required_encoders = {
            "team",
            "map",
            "year",
            "agent",
        }

        missing = (
            required_encoders
            - set(self.encoders.keys())
        )

        if missing:
            raise ValueError(
                "Missing inference encoders: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_input(
        team: str,
        map_name: str,
        year: int,
        agents: Iterable[str],
    ) -> tuple[int, list[str]]:
        """
        Validate application-level input and return
        normalized year and agents.
        """

        if (
            not isinstance(team, str)
            or not team.strip()
        ):
            raise ValueError(
                "Team is required."
            )

        if (
            not isinstance(map_name, str)
            or not map_name.strip()
        ):
            raise ValueError(
                "Map is required."
            )

        if year is None:
            raise ValueError(
                "Year is required."
            )

        try:
            normalized_year = int(year)

        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                "Year must be an integer."
            )

        if agents is None:
            raise ValueError(
                "Agents are required."
            )

        try:
            normalized_agents = (
                normalize_agents(agents)
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                str(exc)
            ) from exc

        unknown_agents = [
            agent
            for agent in normalized_agents
            if agent not in AGENT_ROLE_MAP
        ]

        if unknown_agents:
            raise ValueError(
                "Unknown agents: "
                + ", ".join(
                    unknown_agents
                )
            )

        return (
            normalized_year,
            normalized_agents,
        )

    # ========================================================
    # CONTEXT VALIDATION
    # ========================================================

    def _validate_context(
        self,
        team: str,
        map_name: str,
        year: int,
    ) -> None:
        """
        Ensure Team + Map + Year exists
        in the Team V2 dataset.
        """

        context = self.dataset[
            (self.dataset["Team"] == team)
            & (
                self.dataset["Map"]
                == map_name
            )
            & (
                self.dataset["Year"]
                == year
            )
        ]

        if context.empty:
            raise ValueError(
                f"No Team V2 data for "
                f"{team} / "
                f"{map_name} / "
                f"{year}."
            )

    # ========================================================
    # HISTORICAL FEATURE ROW
    # ========================================================

    @staticmethod
    def _build_historical_row(
        exact: pd.DataFrame,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
    ) -> pd.DataFrame:
        """
        Build the feature-engineering input from
        an exact historical row.

        Winrate is the ML target.

        It is returned only as historical metadata
        and is never used to calculate Composition Strength.
        """

        row = exact.iloc[0]

        return pd.DataFrame(
            [
                {
                    "Team": team,
                    "Map": map_name,
                    "Year": year,
                    "Agent": agents,
                    "Role Pattern": row[
                        "Role Pattern"
                    ],
                    "Duelist Count": int(
                        row["Duelist Count"]
                    ),
                    "Initiator Count": int(
                        row["Initiator Count"]
                    ),
                    "Controller Count": int(
                        row["Controller Count"]
                    ),
                    "Sentinel Count": int(
                        row["Sentinel Count"]
                    ),
                    "Team Overall WR": float(
                        row["Team Overall WR"]
                    ),
                    "Team Map WR": float(
                        row["Team Map WR"]
                    ),
                    "Composition Strength": float(
                        row["Composition Strength"]
                    ),
                }
            ]
        )

    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def _prepare_features(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: list[str],
    ) -> tuple[
        pd.DataFrame,
        bool,
        float | None,
    ]:
        """
        Choose the correct feature-preparation path.

        FOUND:
            Use exact historical composition row.

        NOT FOUND:
            Build inference features for an unseen composition.
        """

        exact = find_exact_composition(
            self.dataset,
            team,
            map_name,
            year,
            agents,
        )

        # ----------------------------------------------------
        # Duplicate protection
        # ----------------------------------------------------

        if len(exact) > 1:
            raise ValueError(
                "Multiple rows found for "
                "the same exact composition."
            )

        # ----------------------------------------------------
        # Historical composition
        # ----------------------------------------------------

        if len(exact) == 1:
            row = self._build_historical_row(
                exact,
                team,
                map_name,
                year,
                agents,
            )

            historical_winrate = float(
                exact.iloc[0]["Winrate"]
            )

            return (
                row,
                True,
                historical_winrate,
            )

        # ----------------------------------------------------
        # Unseen composition
        # ----------------------------------------------------

        row = prepare_inference_row(
            df=self.dataset,
            team=team,
            map_name=map_name,
            year=year,
            agents=agents,
            agent_role_map=AGENT_ROLE_MAP,
        )

        return (
            row,
            False,
            None,
        )

    # ========================================================
    # AVAILABLE TEAMS
    # ========================================================

    def get_available_teams(
        self,
        year: int,
        map_name: str,
    ) -> list[str]:
        """
        Return teams whose Team + Map + Year context
        has at least 3 maps.

        Year and Map are supplied by the options layer.
        This method applies the Team-specific
        eligibility rule.
        """

        try:
            normalized_year = int(year)

        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                "Year must be an integer."
            )

        if (
            not isinstance(map_name, str)
            or not map_name.strip()
        ):
            raise ValueError(
                "Map is required."
            )

        normalized_map = (
            map_name.strip().lower()
        )

        context = self.dataset[
            (
                self.dataset["Year"]
                == normalized_year
            )
            & (
                self.dataset["Map"]
                .str.lower()
                == normalized_map
            )
        ]

        if context.empty:
            raise ValueError(
                f"No Team V2 data for "
                f"{map_name} / "
                f"{normalized_year}."
            )

        context_played = (
            context.groupby("Team")[
                "Total Maps Played"
            ].sum()
        )

        eligible = context_played[
            context_played >= 3
        ]

        if eligible.empty:
            raise ValueError(
                f"No eligible teams for "
                f"{map_name} / "
                f"{normalized_year}."
            )

        return sorted(
            str(team)
            for team in eligible.index
            if pd.notna(team)
        )

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

        Definition:

            Primary:
                Total Maps Played DESC

            Tie-breaker:
                Winrate DESC

            Tie-breaker:
                Composition Strength DESC

        This is a historical reference.

        It does NOT mean the composition is objectively
        the strongest composition.
        """

        context = self.dataset[
            (self.dataset["Team"] == team)
            & (
                self.dataset["Map"]
                == map_name
            )
            & (
                self.dataset["Year"]
                == year
            )
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

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        team: str,
        map_name: str,
        year: int,
        agents: Iterable[str],
    ) -> dict:
        """
        Run one Team Prediction V2 inference request.
        """

        # ----------------------------------------------------
        # 1. Validate input
        # ----------------------------------------------------

        year, agents = self._validate_input(
            team=team,
            map_name=map_name,
            year=year,
            agents=agents,
        )

        # ----------------------------------------------------
        # 2. Validate context
        # ----------------------------------------------------

        self._validate_context(
            team,
            map_name,
            year,
        )

        # ----------------------------------------------------
        # 3. Historical best composition
        # ----------------------------------------------------

        best_composition = (
            self._get_best_composition(
                team=team,
                map_name=map_name,
                year=year,
            )
        )

        # ----------------------------------------------------
        # 4. Prepare inference features
        # ----------------------------------------------------

        (
            prepared,
            historical_found,
            historical_winrate,
        ) = self._prepare_features(
            team=team,
            map_name=map_name,
            year=year,
            agents=agents,
        )

        # ----------------------------------------------------
        # 5. Encode
        # ----------------------------------------------------

        encoded = encode_features(
            prepared,
            self.encoders,
        )

        if encoded.shape != (
            1,
            EXPECTED_FEATURE_COUNT,
        ):
            raise ValueError(
                f"Invalid inference feature "
                f"shape: {encoded.shape}. "
                f"Expected "
                f"(1, {EXPECTED_FEATURE_COUNT})."
            )

        # ----------------------------------------------------
        # 6. Raw model prediction
        # ----------------------------------------------------

        raw_prediction = float(
            self.model.predict(encoded)[0]
        )

        # ----------------------------------------------------
        # 7. Composition Strength + evidence
        # ----------------------------------------------------

        strength = (
            calculate_composition_strength(
                self.dataset,
                team,
                map_name,
                year,
                agents,
                AGENT_ROLE_MAP,
            )
        )

        # Defensive compatibility:
        # older inference_feature_builder versions
        # may not return pattern_adoption.

        if "pattern_adoption" not in strength:
            strength[
                "pattern_adoption"
            ] = strength_for_new_row(
                self.dataset,
                team,
                map_name,
                strength["role_pattern"],
                "|".join(
                    normalize_agents(
                        agents
                    )
                ),
            )[
                "pattern_adoption"
            ]

        # ----------------------------------------------------
        # 8. Evidence / confidence
        # ----------------------------------------------------

        evidence = self._evidence(
            team,
            map_name,
            year,
            agents,
            strength,
        )

        # ----------------------------------------------------
        # 9. Blend raw prediction with prior
        # ----------------------------------------------------

        (
            blended,
            prior,
            blended_range,
        ) = self._blend_prediction(
            raw_prediction,
            prepared,
            evidence["team_map_played"],
        )

        # ----------------------------------------------------
        # 10. Role-pattern adjustment
        # ----------------------------------------------------

        penalty, tier = pattern_penalty(
            strength["pattern_adoption"],
            self.pattern_adjust,
        )

        prediction = max(
            0.0,
            blended - penalty,
        )

        prediction_range = {
            "low": round(
                max(
                    0.0,
                    blended_range["low"]
                    - penalty,
                ),
                4,
            ),
            "high": round(
                max(
                    0.0,
                    blended_range["high"]
                    - penalty,
                ),
                4,
            ),
            "level": blended_range["level"],
        }

        # ----------------------------------------------------
        # 11. Final response
        # ----------------------------------------------------

        return {
            "input": {
                "team": team,
                "map": map_name,
                "year": year,
                "agents": agents,
            },

            "prediction": {
                "winrate": prediction,
                "percentage": round(
                    prediction * 100,
                    2,
                ),

                "model_winrate": round(
                    raw_prediction,
                    4,
                ),

                "prior_winrate": round(
                    prior,
                    4,
                ),

                "range": prediction_range,

                "winrate_before_adjustment": round(
                    blended,
                    4,
                ),

                "pattern_adjustment": {
                    "points": round(
                        -penalty * 100,
                        2,
                    ),

                    "tier": tier,

                    "reason": (
                        prevalence_label(
                            strength[
                                "pattern_adoption"
                            ][
                                "global_share"
                            ]
                        )
                        + (
                            "; belum pernah dipakai tim ini"
                            if strength[
                                "pattern_adoption"
                            ][
                                "team_maps"
                            ] == 0
                            else
                            f"; dipakai tim ini "
                            f"{strength['pattern_adoption']['team_maps']} map"
                        )
                    ),
                },
            },

            "confidence": evidence,

            "composition": {
                "role_pattern": str(
                    prepared.iloc[0][
                        "Role Pattern"
                    ]
                ),

                "duelist_count": int(
                    prepared.iloc[0][
                        "Duelist Count"
                    ]
                ),

                "initiator_count": int(
                    prepared.iloc[0][
                        "Initiator Count"
                    ]
                ),

                "controller_count": int(
                    prepared.iloc[0][
                        "Controller Count"
                    ]
                ),

                "sentinel_count": int(
                    prepared.iloc[0][
                        "Sentinel Count"
                    ]
                ),
            },

            "historical": {
                "found": historical_found,

                "winrate": historical_winrate,

                "composition_strength": round(
                    float(
                        prepared.iloc[0][
                            "Composition Strength"
                        ]
                    ),
                    2,
                ),
            },

            "best_composition": best_composition,

            "inference": {
                "feature_count": int(
                    encoded.shape[1]
                ),
            },
        }