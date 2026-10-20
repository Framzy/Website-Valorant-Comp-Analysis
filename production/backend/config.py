from pathlib import Path

# BASE DIRECTORY

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "ml" / "artifacts"

# DATASET

GENERAL_DATASET_PATH = (
    DATA_DIR / "valorant_dataset_general_v2.csv"
)

TEAM_DATASET_PATH = (
    DATA_DIR / "valorant_dataset_team_v2.csv"
)

# ARTIFACT

ARTIFACT_TEAM_DIR = (
    ARTIFACT_DIR / "team"
)