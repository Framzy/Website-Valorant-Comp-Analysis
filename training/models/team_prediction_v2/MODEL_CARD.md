# Model Card

# Team Prediction V2

Version

```
2.0
```

Status

```
Production Baseline
```

Last Updated

```
July 2026
```

---

# Model Overview

Team Prediction V2 is the machine learning model used by Valorant Predictor to evaluate professional team compositions.

The project combines handcrafted domain knowledge with supervised machine learning.

Instead of allowing machine learning to determine team strength directly, the project first constructs a Composition Strength score using historical statistics.

Machine learning is then used to validate whether this handcrafted score is informative by predicting historical winrate.

---

# Intended Use

The model is intended to:

- Evaluate professional Valorant team compositions.
- Support draft recommendation.
- Estimate expected historical performance.
- Validate Composition Strength against historical match results.

---

# Out of Scope

This model is NOT designed to:

- Predict live esports matches.
- Predict future tournament winners.
- Replace analyst judgment.
- Evaluate player mechanics.
- Evaluate opponent-specific strategies.

---

# Inputs

Categorical

- Team
- Map
- Meta (Year)
- Agent Composition

Role Features

- Duelist Count
- Initiator Count
- Controller Count
- Sentinel Count

Role Pattern

Examples

- 1D-2I-1C-1S
- 1D-2I-2C-0S
- 2D-1I-2C-0S

Historical Features

- Team Overall Winrate
- Team Map Winrate
- Composition Strength

---

# Output

The machine learning model predicts:

```
Winrate
```

The application presents:

```
Composition Strength
```

Composition Strength is the primary recommendation score used by the application.

---

# Dataset

Source

Professional Valorant Match Dataset

Processed Samples

```
2197
```

Agents

```
29
```

Teams

```
57
```

Maps

```
12
```

Meta Versions

```
3
```

---

# Model

Algorithm

```
XGBoost Regressor
```

Training Strategy

```
Supervised Regression
```

Prediction Target

```
Winrate
```

---

# Evaluation

MAE

```
0.2453
```

RMSE

```
0.3171
```

R²

```
0.1981
```

Baseline MAE

```
0.2995
```

Improvement

```
18.09%
```

---

# Feature Importance

Important feature groups include:

- Team Statistics
- Composition Strength
- Agent Composition
- Role Features

Composition Strength consistently contributes meaningful predictive information.

---

# Assumptions

The model assumes:

- Historical professional matches represent strategic tendencies.
- Team identity remains meaningful within the selected meta.
- Historical composition usage reflects practical team preferences.

---

# Limitations

Current dataset does not contain:

- Opponent information
- Player statistics
- Patch version
- Economy
- Round history
- Side-specific data

These limitations reduce the maximum achievable prediction accuracy.

---

# Ethical Considerations

The model should be treated as a decision-support tool.

Predictions are probabilistic and should not be interpreted as guaranteed competitive outcomes.

Human strategic analysis should always take precedence.

---

# Maintenance

Current Status

```
Frozen
```

Future updates should prioritize richer datasets instead of increasing model complexity.
