"""
Stability & blend experiment (Tahap 2) - READ ONLY, saves nothing.

    python -m training.team.evaluate_stability

Uses the same split and the same leak-free WR features as train_team_v2.py.
Compares: baselines, current XGB params, a regularized XGB, and a
"prior + capped model deviation" blend, and measures how much one
same-role agent swap moves each prediction.
"""
import ast
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from training.team.config import (
    XGB_PARAMS, XGB_PARAMS_LEGACY, AGENT_ROLE_MAP, BLEND,
)
from training.team.feature_engineering import load_dataset, build_feature_pipeline
from training.team.train_team_v2 import split_dataset
from training.team.wr_features import apply_honest_wr, honest_wr

REG_PARAMS = XGB_PARAMS          # now the regularized set
CAP = BLEND["deviation_cap"]     # max |model - prior| allowed in the blend
LAMBDAS = (0.0, 0.25, 0.5, 1.0)


def main():
    df = load_dataset()
    pipe = build_feature_pipeline(df)
    names = list(pipe["feature_names"])
    data = apply_honest_wr(split_dataset(pipe["X"], pipe["y"]), df, names)
    Xtr, Xte = data["X_train"], data["X_test"]
    ytr, yte = data["y_train"].to_numpy(), data["y_test"].to_numpy()
    tr_idx, te_idx = data["y_train"].index, data["y_test"].index
    w_te = df.loc[te_idx, "Total Maps Played"].to_numpy()
    w_tr = np.sqrt(df.loc[tr_idx, "Total Maps Played"].to_numpy())

    _, wr_te = honest_wr(df, tr_idx, te_idx)
    prior = BLEND["prior_map_weight"] * wr_te[:, 1] + (1 - BLEND["prior_map_weight"]) * wr_te[:, 0]      # team-map / team-overall

    def mae(p):
        return mean_absolute_error(yte, p), np.average(abs(p - yte), weights=w_te)

    rows = {"mean of train": np.full(len(yte), ytr.mean()), "prior only": prior}
    models = {}
    for label, params, sw in (
        ("XGB current params", XGB_PARAMS_LEGACY, None),
        ("XGB regularized + sqrt(maps) weights", REG_PARAMS, w_tr),
    ):
        m = XGBRegressor(**params)
        m.fit(Xtr, ytr, sample_weight=sw)
        models[label] = m
        rows[label] = m.predict(Xte)

    reg = models["XGB regularized + sqrt(maps) weights"]
    for lam in LAMBDAS[1:]:
        dev = np.clip(rows["XGB regularized + sqrt(maps) weights"] - prior, -CAP, CAP)
        rows[f"blend: prior + {lam} * capped(model - prior)"] = np.clip(prior + lam * dev, 0, 1)

    print("\n" + "=" * 78)
    print(f"{'variant':52s} {'MAE':>7s} {'MAE(w)':>8s}   (w = weighted by maps)")
    print("=" * 78)
    for k, p in rows.items():
        a, b = mae(p)
        print(f"{k:52s} {a:7.4f} {b:8.4f}")

    # ---- same-role agent swap sensitivity (all other features fixed) ----
    agent_cols = {n[len("agent_"):]: i for i, n in enumerate(names) if n.startswith("agent_")}
    role_of = {a: AGENT_ROLE_MAP[a] for a in agent_cols}
    rng = np.random.default_rng(0)
    lists = df.loc[te_idx, "Agent"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v).tolist()
    X2 = Xte.copy()
    for r, ags in enumerate(lists):
        old = ags[rng.integers(len(ags))]
        pool = [a for a in agent_cols if role_of[a] == role_of[old] and a not in ags]
        if not pool:
            continue
        new = pool[rng.integers(len(pool))]
        X2[r, agent_cols[old]] = 0
        X2[r, agent_cols[new]] = 1

    print("\nSame-role single-agent swap -> |change in prediction| (test rows)")
    print(f"{'variant':52s} {'mean':>6s} {'p90':>6s} {'max':>6s}")
    for label, m in models.items():
        d = np.abs(m.predict(X2) - m.predict(Xte))
        print(f"{label:52s} {d.mean():6.3f} {np.quantile(d, .9):6.3f} {d.max():6.3f}")
    for lam in LAMBDAS[1:]:
        f = lambda X: np.clip(prior + lam * np.clip(reg.predict(X) - prior, -CAP, CAP), 0, 1)
        d = np.abs(f(X2) - f(Xte))
        print(f"{f'blend lambda={lam}':52s} {d.mean():6.3f} {np.quantile(d, .9):6.3f} {d.max():6.3f}")


if __name__ == "__main__":
    main()
