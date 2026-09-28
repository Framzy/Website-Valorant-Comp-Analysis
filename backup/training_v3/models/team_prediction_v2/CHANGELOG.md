# Changelog

All notable changes to Team Prediction V2 are documented here.

---

# Version 2.0

Status

```
Production Baseline
```

Date

```
July 2026
```

---

## Major Changes

### Dataset

- Rebuilt preprocessing pipeline from scratch.
- Normalized professional team names.
- Aggregated raw match records into composition-level dataset.
- Reconstructed compositions containing more than five agents.
- Removed duplicated training compositions.

---

### Historical Statistics

Added historical context features:

- Team Overall Winrate
- Team Map Winrate
- Context Played
- Usage Ratio
- Reliability

---

### Role Analysis

Added role-based features.

Role Counts

- Duelist
- Initiator
- Controller
- Sentinel

Role Pattern examples

- 1D-2I-1C-1S
- 1D-2I-2C-0S
- 2D-1I-2C-0S

---

### Composition Strength

Introduced Composition Strength as the primary recommendation score.

Unlike Winrate, this score is generated through domain knowledge rather than machine learning.

Composition Strength combines multiple historical indicators into a single score representing how suitable a composition is for a specific team, map, and meta.

---

### Feature Engineering

Current feature groups:

- Team
- Map
- Meta
- Agent Composition
- Role Count
- Role Pattern
- Team Statistics
- Composition Strength

---

### Machine Learning

Migrated from TensorFlow (V1) to XGBoost (V2).

Reason:

- Better performance on tabular data.
- Faster training.
- Easier maintenance.
- Better interpretability.

Prediction target:

```
Winrate
```

Composition Strength remains a handcrafted feature.

---

### Validation

Added validation modules:

- Strength Validation
- Prediction Validation
- Feature Importance
- Baseline Comparison

Final evaluation indicates:

- Pipeline successfully validated.
- Model performs better than baseline predictor.
- Composition Strength contributes meaningful predictive information.

---

## Design Philosophy

The project intentionally separates:

Knowledge Engineering

↓

Composition Strength

from

Machine Learning

↓

Winrate Prediction

Machine learning is used to validate the handcrafted score rather than replace it.

This keeps the recommendation logic interpretable while still benefiting from supervised learning.

---

## Lessons Learned

Several important conclusions were reached during development.

### 1.

Better preprocessing contributes more than changing algorithms.

### 2.

Historical context is more valuable than raw composition frequency.

### 3.

Dataset quality limits achievable model performance.

### 4.

Not every improvement justifies additional complexity.

### 5.

A stable and maintainable pipeline is preferable to chasing marginal accuracy gains.

---

## Final Decision

Team Prediction V2 is officially frozen as the production baseline.

Future development should prioritize:

- Backend integration
- Frontend integration
- General Prediction V2

rather than continuing to optimize Team Prediction V2.

Future improvements should only be made when significantly richer datasets become available.
