# Team Prediction V2

## Overview

Team Prediction V2 is the professional team composition prediction model used in the Valorant Predictor project.

Unlike the first version, this model is built on a complete preprocessing pipeline and a domain-specific scoring system called **Composition Strength**.

The objective of this model is **not** to directly estimate team strength using machine learning.

Instead, the project follows a two-stage approach:

1. Build a human-designed Composition Strength score using historical statistics.
2. Validate that score by predicting Winrate with XGBoost.

This allows Composition Strength to remain the primary recommendation score while Winrate serves as an objective validation target.

---

## Project Pipeline

Raw Dataset

↓

Data Cleaning

↓

Composition Reconstruction

↓

Dataset Aggregation

↓

Feature Engineering

↓

Composition Strength Calculation

↓

XGBoost Training

↓

Winrate Prediction

↓

Model Validation

---

## Input Features

### Categorical

- Team
- Map
- Meta (Year)
- Agent Composition

### Role Features

- Duelist Count
- Initiator Count
- Controller Count
- Sentinel Count

### Role Pattern

Examples:

- 1D-2I-1C-1S
- 1D-2I-2C-0S
- 2D-1I-2C-0S

### Numerical Features

- Team Overall Winrate
- Team Map Winrate
- Composition Strength

---

## Target

Although the project focuses on **Composition Strength**, the machine learning model predicts:

```
Winrate
```

Reason:

The dataset does not provide ground-truth labels for Composition Strength.

Winrate is the only observable performance metric available, making it suitable as a supervised learning target.

Composition Strength is therefore treated as a domain-engineered feature rather than the prediction target.

---

## Composition Strength

Composition Strength is calculated before machine learning.

It represents how suitable a composition is for a specific:

- Team
- Map
- Meta

The score is derived from several historical indicators such as:

- Context Usage
- Usage Ratio
- Reliability
- Role Pattern Usage
- Composition Familiarity

Machine learning is then used only to validate whether this score correlates with actual competitive performance.

---

## Machine Learning

Model:

```
XGBoost Regressor
```

Prediction Target:

```
Winrate
```

---

## Evaluation Results

Dataset Size

```
2197 compositions
```

Evaluation Metrics

| Metric | Result |
| ------ | ------ |
| MAE    | 0.2453 |
| RMSE   | 0.3171 |
| R²     | 0.1981 |

Baseline Comparison

| Predictor     | MAE    |
| ------------- | ------ |
| Mean Baseline | 0.2995 |
| XGBoost       | 0.2453 |

Improvement over baseline

```
18.09%
```

---

## Feature Importance

One of the most important engineered features is:

```
Composition Strength
```

This indicates that the handcrafted scoring system contributes meaningful information to predicting team performance.

---

## Current Status

```
Status:
Production Baseline
```

The current version is considered stable and ready for backend integration.

Future improvements should focus primarily on obtaining richer datasets rather than redesigning the existing pipeline.

---

## Known Limitations

The model does not include:

- Opponent information
- Player statistics
- Patch notes
- Economy data
- Round-level information
- Side-specific performance

These limitations originate from the available dataset rather than the model architecture.

---

## Future Improvements

Potential future improvements include:

- Larger competitive dataset
- Patch-aware modeling
- Opponent-aware prediction
- Player-level statistics
- Ensemble models
- Meta evolution tracking

These improvements are optional and are not required for the current production baseline.
