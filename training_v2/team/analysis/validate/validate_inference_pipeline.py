import joblib

from training_v2.team.config import MODEL_TEAM_DIR


def main():

    print("=" * 60)
    print("VALIDATE INFERENCE ARTIFACTS")
    print("=" * 60)

    encoders = joblib.load(
        MODEL_TEAM_DIR / "encoders.joblib"
    )

    feature_names = joblib.load(
        MODEL_TEAM_DIR / "feature_names.joblib"
    )

    model = joblib.load(
        MODEL_TEAM_DIR / "team_model_v2.joblib"
    )

    print()
    print("ENCODER CATEGORIES")
    print("--------------------------")

    print(
        f"Team  : "
        f"{len(encoders['team'].categories_[0])}"
    )

    print(
        f"Map   : "
        f"{len(encoders['map'].categories_[0])}"
    )

    print(
        f"Year  : "
        f"{len(encoders['year'].categories_[0])}"
    )

    print(
        f"Agent : "
        f"{len(encoders['agent'].classes_)}"
    )

    print()
    print("FEATURE CONTRACT")
    print("--------------------------")

    categorical_count = (
        len(encoders["team"].categories_[0])
        + len(encoders["map"].categories_[0])
        + len(encoders["year"].categories_[0])
        + len(encoders["agent"].classes_)
    )

    total_features = (
        categorical_count
        + 4   # role count
        + 8   # role pattern
        + 3   # numeric
    )

    print(f"Categorical : {categorical_count}")
    print(f"Role Count  : 4")
    print(f"Role Pattern: 8")
    print(f"Numeric     : 3")
    print(f"Total       : {total_features}")

    print()
    print("MODEL")
    print("--------------------------")

    print(
        f"Model Features : "
        f"{model.n_features_in_}"
    )

    print(
        f"Feature Names  : "
        f"{len(feature_names)}"
    )

    print()
    print("VALIDATION")
    print("--------------------------")

    if total_features != 116:
        raise ValueError(
            f"Expected 116 features, "
            f"got {total_features}"
        )

    if len(feature_names) != 116:
        raise ValueError(
            "feature_names.joblib "
            "does not contain 116 features."
        )

    if model.n_features_in_ != 116:
        raise ValueError(
            "Model does not expect "
            "116 features."
        )

    print("[PASS] Encoder feature count : 116")
    print("[PASS] Feature names count    : 116")
    print("[PASS] Model feature count    : 116")
    print("[PASS] Inference artifacts compatible")


if __name__ == "__main__":
    main()