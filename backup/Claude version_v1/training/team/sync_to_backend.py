"""
Sync Team V2 artifacts to the backend (no backend refactor needed).

Usage (from the folder that contains both `training/` and `backend/`):
    python -m training.team.sync_to_backend

Copies, as ONE unit (dataset + model must always ship together):
    dataset/valorant_dataset_team_v2.csv -> backend/data/
    models/team_prediction_v2/*.joblib, metadata.json -> backend/models/team_prediction_v2/
    team/inference_feature_builder.py, feature_encoder.py -> backend/ml/team/
        (only the import line of feature_encoder is rewritten)
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # folder holding training/ and backend/
T, B = ROOT / "training", ROOT / "backend"

meta = json.loads((T / "models/team_prediction_v2/metadata.json").read_text())
if "evaluation_protocol" not in meta:
    raise SystemExit("metadata.json has no evaluation_protocol -> model was not "
                     "trained with the leak-free pipeline. Retrain first.")

copies = [(T / "dataset/valorant_dataset_team_v2.csv", B / "data/valorant_dataset_team_v2.csv")]
for name in ["team_model_v2.joblib", "encoders.joblib", "feature_names.joblib", "metadata.json"]:
    copies.append((T / "models/team_prediction_v2" / name,
                   B / "models/team_prediction_v2" / name))
for src, dst in copies:
    shutil.copy2(src, dst); print("copied", dst.relative_to(ROOT))

# ml/team files: keep backend import style
shutil.copy2(T / "team/inference_feature_builder.py", B / "ml/team/inference_feature_builder.py")
enc = (T / "team/feature_encoder.py").read_text(encoding="utf-8")
enc = enc.replace("EXPECTED_FEATURE_COUNT = 116",
                  "from backend.constants import (\n    EXPECTED_FEATURE_COUNT,\n)", 1)
(B / "ml/team/feature_encoder.py").write_text(enc, encoding="utf-8")

# comp_strength_hierarchy.py: backend keeps its own hand-maintained copy
# (no honest_strength / RANDOM_STATE, since backend never trains -- see
# backend/ml/team/comp_strength_hierarchy.py's own header comment). This
# sync only WARNS if CS_SHRINKAGE_K has drifted; it does not overwrite it,
# so update the backend copy's constant by hand if you change it here.
config_path = T / "team/config.py"
train_k = {"__file__": str(config_path)}
exec(
    compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"),
    train_k,
)
backend_cs = (B / "ml/team/comp_strength_hierarchy.py").read_text(encoding="utf-8")
for level, value in train_k["CS_SHRINKAGE_K"].items():
    if f'"{level}": {value},' not in backend_cs:
        print(f"[WARN] backend/ml/team/comp_strength_hierarchy.py CS_SHRINKAGE_K "
              f"looks out of date for '{level}' (expected {value}). Update it by hand.")

print("synced ml/team/*.py")
