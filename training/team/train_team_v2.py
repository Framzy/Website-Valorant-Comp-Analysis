"""
Train Team Model V2
===================
"""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from training.team.config import (
    MODEL_TEAM_DIR,
    BLEND,
    PATTERN_ADJUST,
    XGB_PARAMS,
)


from training.team.feature_engineering import (
    load_dataset,
    build_feature_pipeline,
)

from training.team.wr_features import apply_honest_wr
from training.team.comp_strength_hierarchy import honest_strength

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def split_dataset(
    X,
    y,
):
    """
    Split train and test dataset.
    """

    print("\n" + "=" * 60)
    print("TRAIN TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print(f"[INFO] Train : {X_train.shape}")

    print(f"[INFO] Test  : {X_test.shape}")

    return {

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

    }
       
def build_model():
    """
    Create XGBoost model.
    """

    print("\n" + "=" * 60)
    print("BUILD MODEL")
    print("=" * 60)

    model = XGBRegressor(

        objective="reg:squarederror",

        **XGB_PARAMS,

    )

    print("[INFO] Model Created")

    return model

def train_model(
    model,
    train_data,
    sample_weight=None,
):
    """
    Train XGBoost model.
    """

    print("\n" + "=" * 60)
    print("TRAIN MODEL")
    print("=" * 60)

    model.fit(

        train_data["X_train"],

        train_data["y_train"],

        sample_weight=sample_weight,

    )

    print("[INFO] Training Complete")

    return model

"""
Evaluate Model
====
"""
def predict_model(
    model,
    train_data,
):
    """
    Predict test dataset.
    """

    print("\n" + "=" * 60)
    print("PREDICT MODEL")
    print("=" * 60)

    prediction = model.predict(
        train_data["X_test"]
    )

    print(f"[INFO] Prediction : {len(prediction)}")

    return prediction

def evaluate_model(
    prediction,
    train_data,
):
    """
    Evaluate trained model.
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    mae = mean_absolute_error(
        train_data["y_test"],
        prediction,
    )

    rmse = float(np.sqrt(mean_squared_error(
        train_data["y_test"],
        prediction,
    )))

    r2 = r2_score(
        train_data["y_test"],
        prediction,
    )

    baseline_mae = mean_absolute_error(
        train_data["y_test"],
        np.full(len(train_data["y_test"]), train_data["y_train"].mean()),
    )

    print(f"[INFO] Baseline MAE (predict train mean) : {baseline_mae:.4f}")

    print(f"[INFO] MAE  : {mae:.4f}")

    print(f"[INFO] RMSE : {rmse:.4f}")

    print(f"[INFO] R²   : {r2:.4f}")

    return {

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

    }
    
def show_feature_importance(
    model,
    feature_names,
):
    """
    Show top feature importance.
    """

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_,

    })

    importance = (

        importance

        .sort_values(
            "Importance",
            ascending=False,
        )

        .head(20)

    )

    print(importance.to_string(index=False))

    return importance
    
def save_model(
    model,
):
    """
    Save trained model.
    """

    print("\n" + "=" * 60)
    print("SAVE MODEL")
    print("=" * 60)

    path = MODEL_TEAM_DIR / "team_model_v2.joblib"

    joblib.dump(
        model,
        path,
    )

    print(f"[INFO] Saved : {path}")
    
def save_encoders(
    encoders,
):
    """
    Save encoders.
    """

    path = MODEL_TEAM_DIR / "encoders.joblib"
    
    joblib.dump(
        encoders,
        path,
    )

    print(f"[INFO] Saved : {path}")
    
def save_feature_names(
    feature_names,
):
    """
    Save feature names.
    """

    path = MODEL_TEAM_DIR / "feature_names.joblib"

    joblib.dump(
        feature_names,
        path,
    )

    print(f"[INFO] Saved : {path}")

def save_metadata(
    pipeline,
    train_data,
    evaluation,
):
    """
    Save model metadata.
    """

    metadata = {

        "model_name": "Team Prediction V2",

        "version": "2.0",

        "status": "Production Baseline",

        "algorithm": "XGBoost Regressor",

        "prediction_target": "Winrate",

        "primary_score": "Composition Strength",

        "evaluation_protocol": "honest_wr_v1 (OOF train WR, train-only test WR)",

        "blend": BLEND,

        "pattern_adjust": PATTERN_ADJUST,

        "feature_count": len(
            pipeline["feature_names"]
        ),

        "train_size": len(
            train_data["X_train"]
        ),

        "test_size": len(
            train_data["X_test"]
        ),

        "evaluation": {

            "mae": round(
                evaluation["mae"],
                4,
            ),

            "rmse": round(
                evaluation["rmse"],
                4,
            ),

            "r2": round(
                evaluation["r2"],
                4,
            ),

        }

    }

    path = MODEL_TEAM_DIR / "metadata.json"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            metadata,

            file,

            indent=4,

        )

    print(f"[INFO] Saved : {path}")

def main():

    df = load_dataset()

    pipeline = build_feature_pipeline(df)

    train_data = split_dataset(

        pipeline["X"],

        pipeline["y"],

    )

    # Leak-free WR features (train: out-of-fold, test: train-only).
    train_data = apply_honest_wr(
        train_data,
        df,
        pipeline["feature_names"],
    )

    # Leak-free Composition Strength (Tahap 4): same idea, since the
    # hierarchical chain also folds in each row's own Winrate.
    cs_col = pipeline["feature_names"].index("composition_strength")
    cs_train, cs_test = honest_strength(
        df,
        train_data["y_train"].index,
        train_data["y_test"].index,
    )
    train_data["X_train"][:, cs_col] = cs_train
    train_data["X_test"][:, cs_col] = cs_test

    model = build_model()

    # Rows with more maps are more reliable targets.
    sample_weight = np.sqrt(
        df.loc[train_data["y_train"].index, "Total Maps Played"].to_numpy()
    )

    model = train_model(

        model,

        train_data,

        sample_weight=sample_weight,

    )

    prediction = predict_model(
    model,
    train_data,
)

    evaluation = evaluate_model(
        prediction,
        train_data,
    )

    print()

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(type(model))

    print()

    print(evaluation)
    
    importance = show_feature_importance(

        model,

        pipeline["feature_names"],

    )

    save_model(model)

    save_encoders(
        pipeline["encoders"]
    )

    save_feature_names(
        pipeline["feature_names"]
    )

    save_metadata(

        pipeline,

        train_data,

        evaluation,

    )
    
if __name__ == "__main__":

    main()