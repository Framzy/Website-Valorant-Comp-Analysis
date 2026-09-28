"""
Configuration for Valorant Predictor V2 Training
"""

from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    ROOT_DIR
    / "dataset"
    / "valorant_dataset_all.csv"
)
DATASET_DIR = (
    ROOT_DIR
    / "dataset"
)
TEAM_DATASET_PATH = (
    ROOT_DIR
    / "dataset"
    / "valorant_dataset_team_v2.csv"
)

MODEL_TEAM_DIR = ROOT_DIR / "models" / "team_prediction_v2"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# RANDOM
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# DATASET
# ==========================================================


TRAIN_TEST_SPLIT = 0.20

CV_FOLDS = 5

FEATURE_COLUMNS = [

    "Team",

    "Map",

    "Year",

    "Agent",

    "Role Pattern",

    "Duelist Count",

    "Initiator Count",

    "Controller Count",

    "Sentinel Count",

    "Team Overall WR",

    "Team Map WR",

    "Composition Strength",

]

TARGET_COLUMN = "Winrate"

# ==========================================================
# XGBOOST
# ==========================================================

# Regularized (Tahap 2). Chosen with evaluate_stability.py: lower honest
# MAE than the legacy params and ~3x less sensitive to a single agent swap.
XGB_PARAMS = {

    "n_estimators": 200,

    "max_depth": 3,

    "min_child_weight": 10,

    "learning_rate": 0.05,

    "subsample": 0.8,

    "colsample_bytree": 0.6,

    "reg_lambda": 5.0,

    "random_state": RANDOM_STATE

}

# Previous parameters, kept only for comparison in evaluate_stability.py
XGB_PARAMS_LEGACY = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# ==========================================================
# SERVING BLEND (stored in metadata.json, read by the backend)
# ==========================================================
# final = prior + LAMBDA * clip(model - prior, -CAP, +CAP)
# prior = MAP_WEIGHT * Team Map WR + (1 - MAP_WEIGHT) * Team Overall WR
BLEND = {
    "prior_map_weight": 0.3,
    "lambda": 0.5,
    "deviation_cap": 0.15,
    "range_z": 1.28,          # ~80% uncertainty range
    "range_prior_maps": 10,   # pseudo-maps, same as WR_SHRINKAGE_K
}

# ==========================================================
# ROLE ORDER
# ==========================================================

ROLE_ORDER = [
    "duelist",
    "initiator",
    "controller",
    "sentinel"
]

# ==========================================================
# AGENT ROLE MAP
# ==========================================================

AGENT_ROLE_MAP = {

    # Duelist
    "iso": "duelist",
    "jett": "duelist",
    "raze": "duelist",
    "reyna": "duelist",
    "yoru": "duelist",
    "neon": "duelist",
    "phoenix": "duelist",
    "waylay": "duelist",

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
    "vyse": "sentinel"

}

# ==========================================================
# Training Dataset Filter
# ==========================================================

MIN_TOURNAMENT = 40
MIN_YEAR = 2

# Shrinkage strength (in maps) for Team Overall WR / Team Map WR.
# (wins + K * global_wr) / (maps + K)
WR_SHRINKAGE_K = 10

# ==========================================================
# COMPOSITION STRENGTH (hierarchical, Tahap 4)
# ==========================================================
# Composition Strength now measures historical PERFORMANCE (win rate),
# not usage frequency. Because most compositions have very few maps
# (median 2), it backs off through a chain of progressively broader,
# progressively more data-rich contexts, each shrunk toward the next:
#
#   L1 exact composition   (Team + Map + exact 5 agents, all years)
#   L2 Team + Map + Role Pattern    <- "how this team wins on this map"
#   L3 Team + Role Pattern          <- "how this team likes to play"
#   L4 Role Pattern (global)        <- "how this playstyle performs, at all"
#   L5 Team Overall WR              <- final fallback
#
# Each level: value = (wins + K * value_of_next_level) / (maps + K).
# If a level has zero maps, it collapses exactly to the level above it,
# so an unusual pattern a team has never tried (or that nobody in the
# dataset has ever played, e.g. 5 Sentinel) does NOT get an invented
# penalty: it honestly falls back to broader evidence, down to the
# team's own baseline. The service layer (team_prediction_service.py)
# separately reports when Level 4 has zero maps, so the product can
# tell the user "this playstyle has never been observed" rather than
# implying the number reflects real analysis of it.
CS_SHRINKAGE_K = {
    "exact": 15,              # L1: noisiest level, shrink hardest
    "team_map_pattern": 10,   # L2
    "team_pattern": 10,       # L3
    "global_pattern": 10,     # L4
}
