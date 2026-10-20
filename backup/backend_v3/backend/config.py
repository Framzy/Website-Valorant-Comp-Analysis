from pathlib import Path

# BASE DIRECTORY

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# DATASET

GENERAL_DATASET_PATH = (
    DATA_DIR / "valorant_dataset_general_v2.csv"
)

TEAM_DATASET_PATH = (
    DATA_DIR / "valorant_dataset_team_v2.csv"
)

# MODEL

MODEL_TEAM_DIR = (
    MODEL_DIR / "team_prediction_v2"
)