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
GENERAL_DATASET_PATH = (
    ROOT_DIR
    / "dataset"
    / "valorant_dataset_general_v2.csv"
)

MODEL_GENERAL_DIR = ROOT_DIR / "models" / "general_prediction_v2"
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

    "Map",

    "Year",

    "Agent",

    "Role Pattern",

    "Duelist Count",

    "Initiator Count",

    "Controller Count",

    "Sentinel Count",

    "Composition Strength",

]

TARGET_COLUMN = "Winrate"

# ==========================================================
# XGBOOST
# ==========================================================

XGB_PARAMS = {

    "n_estimators": 300,

    "max_depth": 6,

    "learning_rate": 0.05,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "random_state": RANDOM_STATE

}

# ==========================================================
# Training Dataset Filter
# ==========================================================

MIN_TOURNAMENT = 40
MIN_YEAR = 2
