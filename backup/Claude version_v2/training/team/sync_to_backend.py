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
import ast
import json
import re
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
# The backend must NOT import from the training package (deployments do not
# ship training/). Copy the builder but point its import at the backend copy
# of comp_strength_hierarchy.py.
builder = (T / "team/inference_feature_builder.py").read_text(encoding="utf-8")
builder = builder.replace(
    "from training.team.comp_strength_hierarchy import",
    "from backend.ml.team.comp_strength_hierarchy import",
)
(B / "ml/team/inference_feature_builder.py").write_text(builder, encoding="utf-8")
enc = (T / "team/feature_encoder.py").read_text(encoding="utf-8")
enc = enc.replace("EXPECTED_FEATURE_COUNT = 116",
                  "from backend.constants import (\n    EXPECTED_FEATURE_COUNT,\n)", 1)
(B / "ml/team/feature_encoder.py").write_text(enc, encoding="utf-8")

# comp_strength_hierarchy.py: backend keeps its own hand-maintained copy
# (no honest_strength / RANDOM_STATE, since backend never trains -- see
# backend/ml/team/comp_strength_hierarchy.py's own header comment). This
# sync only WARNS if these constants have drifted; it does not overwrite
# them, so update the backend copy by hand if you change one here.
config_path = T / "team/config.py"
train_k = {"__file__": str(config_path)}
exec(
    compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"),
    train_k,
)
backend_cs = (B / "ml/team/comp_strength_hierarchy.py").read_text(encoding="utf-8")

# Parsed as data (not matched as text), so formatting differences --
# quote style, spacing, trailing commas, line breaks -- never cause a
# false-positive warning; only an actual value difference does.
CHECKS = [
    ("CS_SHRINKAGE_K", r"CS_SHRINKAGE_K\s*=\s*(\{.*?\})", True),
    ("CS_RARE_PATTERN_SHARE_THRESHOLD", r"CS_RARE_PATTERN_SHARE_THRESHOLD\s*=\s*([0-9.eE+-]+)", False),
    ("CS_RARE_PATTERN_DISCOUNT_CAP", r"CS_RARE_PATTERN_DISCOUNT_CAP\s*=\s*([0-9.eE+-]+)", False),
]

for name, pattern, is_dict in CHECKS:
    match = re.search(pattern, backend_cs, re.S)
    backend_val = None
    if match:
        try:
            backend_val = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            backend_val = None

    if backend_val is None:
        print(f"[WARN] Could not find/parse {name} in "
              "backend/ml/team/comp_strength_hierarchy.py. Verify it by hand.")
    elif backend_val != train_k[name]:
        print(f"[WARN] backend/ml/team/comp_strength_hierarchy.py {name} is out of date.\n"
              f"       training : {train_k[name]}\n"
              f"       backend  : {backend_val}\n"
              "       Update the backend copy by hand.")

# Guard: nothing under backend/ may import from training/.
leaks = [
    str(f.relative_to(ROOT))
    for f in list((B / "ml").rglob("*.py")) + list((B / "services").rglob("*.py"))
    if "from training" in f.read_text(encoding="utf-8") or "import training" in f.read_text(encoding="utf-8")
]
if leaks:
    print("[WARN] backend imports from the training package (will break without training/):")
    for f in leaks:
        print("       -", f)

print("synced ml/team/*.py")
