"""
Regression check for Team Prediction (run after every retrain / sync).

    python -m backend.check_team_service

Verifies (1) the API contract is intact (legacy keys + new additive keys),
(2) the 116-feature contract, and (3) that swapping one agent for another of
the SAME role cannot move the prediction by more than lambda * 2 * cap.
"""
import random

from backend.constants import AGENT_ROLE_MAP
from backend.services.team_prediction_service import TeamPredictionService

LEGACY_KEYS = {"input", "prediction", "composition", "historical", "inference"}
NEW_KEYS = {"confidence"}
PRED_KEYS = {"winrate", "percentage", "model_winrate", "prior_winrate", "range"}


def check_contract(res):
    assert LEGACY_KEYS | NEW_KEYS <= set(res), f"missing keys: {(LEGACY_KEYS | NEW_KEYS) - set(res)}"
    assert PRED_KEYS <= set(res["prediction"]), "prediction keys missing"
    assert res["inference"]["feature_count"] == 116
    p, r = res["prediction"]["winrate"], res["prediction"]["range"]
    assert 0 <= r["low"] <= p <= r["high"] <= 1, "range does not contain prediction"
    assert res["confidence"]["level"] in {"low", "medium", "high"}


def main():
    svc = TeamPredictionService()
    b = svc.blend
    bound = b["lambda"] * 2 * b["deviation_cap"] + 1e-9

    print("== G2 Esports / Ascent / 2026 (the reported case) ==")
    hist = svc.dataset[
        (svc.dataset.Team == "G2 Esports") & (svc.dataset.Map == "Ascent") & (svc.dataset.Year == 2026)
    ].iloc[0]["Agent"]
    base = list(hist)
    swapped = sorted("deadlock" if a == "vyse" else a for a in base)
    outs = {}
    for label, ag in (("historical", base), ("vyse->deadlock", swapped)):
        try:
            r = svc.predict("G2 Esports", "Ascent", 2026, ag)
        except ValueError as exc:
            print(f"  {label}: skipped ({exc})"); continue
        check_contract(r)
        outs[label] = r
        p = r["prediction"]
        print(f"  {label:15s} final={p['percentage']:5.1f}%  model={p['model_winrate']*100:5.1f}%  "
              f"prior={p['prior_winrate']*100:5.1f}%  range={p['range']['low']*100:.0f}-{p['range']['high']*100:.0f}%  "
              f"evidence={r['confidence']['level']}")
    if len(outs) == 2:
        d = abs(outs["historical"]["prediction"]["winrate"] - outs["vyse->deadlock"]["prediction"]["winrate"])
        print(f"  |delta| = {d*100:.1f} points")
        assert d <= bound, "reported case still unstable"

    print("\n== Same-role swap sweep ==")
    rnd = random.Random(0)
    rows = svc.dataset.sample(min(300, len(svc.dataset)), random_state=0)
    deltas, levels = [], {"low": 0, "medium": 0, "high": 0}
    for _, row in rows.iterrows():
        ag = list(row["Agent"])
        old = rnd.choice(ag)
        pool = [a for a, r in AGENT_ROLE_MAP.items() if r == AGENT_ROLE_MAP[old] and a not in ag]
        if not pool:
            continue
        new = sorted(a if a != old else rnd.choice(pool) for a in ag)
        try:
            a = svc.predict(row["Team"], row["Map"], int(row["Year"]), ag)
            c = svc.predict(row["Team"], row["Map"], int(row["Year"]), new)
        except ValueError:
            continue
        check_contract(a); check_contract(c)
        deltas.append(abs(a["prediction"]["winrate"] - c["prediction"]["winrate"]))
        levels[c["confidence"]["level"]] += 1
    print(f"  n={len(deltas)}  mean={sum(deltas)/len(deltas):.4f}  max={max(deltas):.4f}  bound={bound:.4f}")
    print(f"  evidence level of swapped compositions: {levels}")
    assert max(deltas) <= bound, "swap sensitivity exceeds bound"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
