"""
Team Prediction V2 - Analysis Test
==================================

Run from Claude Version root:

    python -m training.team.analysis.test.analysis_test_team_service

Purpose:
- Inspect actual service output, not only regression assertions.
- Compare historical compositions across 2024/2025/2026.
- Isolate Year/context effect using the SAME composition.
- Test same-role agent swap sensitivity.
- Test multiple Sentinel alternatives.
- Test an unobserved role pattern (5 Sentinel).
- Inspect confidence, range, prior, raw model output,
  context support and composition support.
- Comparef {team} and Paper Rex behavior.
- Verify the 116-feature contract and validation.

This test does NOT modify datasets, models, or source code.
"""

from __future__ import annotations

from training.team.constants import AGENT_ROLE_MAP
from training.team.service.team_prediction_service import TeamPredictionService


def summarize(label, result):
    i = result["input"]
    p = result["prediction"]
    c = result["confidence"]
    b = result["best_composition"]
    comp = result["composition"]

    print(f"\n--- {label} ---")
    print(f"Team              : {i['team']}")
    print(f"Map               : {i['map']}")
    print(f"Year              : {i['year']}")
    print(f"Agents            : {', '.join(i['agents'])}")
    print(f"Prediction final  : {p['percentage']:.2f}%")
    print(f"Raw model         : {p['model_winrate'] * 100:.2f}%")
    print(f"Prior             : {p['prior_winrate'] * 100:.2f}%")
    print(
        f"Range             : "
        f"{p['range']['low'] * 100:.1f}% - "
        f"{p['range']['high'] * 100:.1f}%"
    )
    print(f"Confidence        : {c['level']}")
    print(f"Team-map played   : {c['team_map_played']}")
    print(f"Context played    : {c['context_played']}")
    print(f"Composition maps  : {c['composition_played']}")
    print(f"Role pattern      : {comp['role_pattern']}")
    print(
        f"Composition str.  : "
        f"{result['historical']['composition_strength']}"
    )
    print(f"Historical found  : {result['historical']['found']}")
    print(f"Best Historical Composition              : {b}")
    print(f"Note              : {c['note']}")


def get_historical_row(svc, team, map_name, year):
    context = svc.dataset[
        (svc.dataset["Team"] == team)
        & (svc.dataset["Map"] == map_name)
        & (svc.dataset["Year"] == year)
    ]

    if context.empty:
        return None

    return context.iloc[0]


def predict_historical(svc, team, map_name, year):
    row = get_historical_row(svc, team, map_name, year)

    if row is None:
        print(
            f"{team} / {map_name} / {year}: "
            "context tidak ditemukan."
        )
        return None

    agents = list(row["Agent"])

    result = svc.predict(
        team,
        map_name,
        year,
        agents,
    )

    return row, result


def main():
    svc = TeamPredictionService()

    print("=" * 80)
    print("TEAM PREDICTION V2 - COMPREHENSIVE ANALYSIS TEST")
    print("=" * 80)

    # ==================================================================
    # TEST 1
    # Historical composition masing-masing tahun
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 1] HISTORICAL COMPOSITION PER YEAR")
    print("=" * 80)

    historical_results = {}
    team = "Paper Rex"
    map_name = "Ascent"

    for year in [2024, 2025, 2026]:
        data = predict_historical(
            svc,
            team,
            map_name,
            year,
        )

        if data is None:
            continue

        row, result = data

        historical_results[year] = {
            "row": row,
            "result": result,
            "agents": list(row["Agent"]),
        }

        summarize(
            f"{team} {map_name} {year} - historical",
            result,
        )

    # ==================================================================
    # TEST 2
    # SAME COMPOSITION across different years
    #
    # Tujuan:
    # mengisolasi pengaruh Year/context.
    #
    # Composition 2026 digunakan sebagai input yang sama untuk:
    # 2024, 2025, 2026.
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 2] SAME COMPOSITION ACROSS YEARS")
    print("=" * 80)

    if 2026 not in historical_results:
        print(f"{team} {map_name} 2026 tidak tersedia.")
    else:
        reference_agents = historical_results[2026]["agents"]

        print(
            "\nReference composition (taken from 2026):"
        )
        print(", ".join(reference_agents))

        same_comp_results = {}

        for year in [2024, 2025, 2026]:
            try:
                result = svc.predict(
                    team,
                    map_name,
                    year,
                    reference_agents,
                )

                same_comp_results[year] = result

                summarize(
                    f"{team} same composition -> context {year}",
                    result,
                )

            except ValueError as exc:
                print(
                    f"{team} / {map_name} / {year}: {exc}"
                )

        if len(same_comp_results) >= 2:
            print("\n--- SAME COMPOSITION YEAR COMPARISON ---")

            for year, result in same_comp_results.items():
                prediction = result["prediction"]["percentage"]
                prior = result["prediction"]["prior_winrate"] * 100
                raw = result["prediction"]["model_winrate"] * 100

                print(
                    f"{year}: "
                    f"final={prediction:.2f}% | "
                    f"raw={raw:.2f}% | "
                    f"prior={prior:.2f}% | "
                    f"context={result['confidence']['context_played']} | "
                    f"historical={result['historical']['found']} | "
                    f"confidence={result['confidence']['level']}"
                )

    # ==================================================================
    # TEST 3
    # Same-role swap
    #
    # Gunakan COMPOSITION 2026 secara eksplisit.
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 3] SAME-ROLE AGENT SWAP")
    print("=" * 80)

    if 2026 in historical_results:
        base_agents = historical_results[2026]["agents"]

        print(
            "\nBase composition:"
        )
        print(", ".join(base_agents))

        swapped_agents = [
            "deadlock" if agent == "vyse" else agent
            for agent in base_agents
        ]

        base_result = svc.predict(
            team,
            map_name,
            2026,
            base_agents,
        )

        swapped_result = svc.predict(
            team,
            map_name,
            2026,
            swapped_agents,
        )

        summarize(
            f"{team} 2026 - original",
            base_result,
        )

        summarize(
            f"{team} 2026 - Vyse -> Deadlock",
            swapped_result,
        )

        delta = abs(
            base_result["prediction"]["winrate"]
            - swapped_result["prediction"]["winrate"]
        )

        print(
            f"\nPrediction delta: "
            f"{delta * 100:.2f} percentage points"
        )

    # ==================================================================
    # TEST 4
    # Sentinel alternatives
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 4] SAME-ROLE SENTINEL ALTERNATIVES")
    print("=" * 80)

    if 2026 in historical_results:
        base_agents = historical_results[2026]["agents"]

        base_result = svc.predict(
            team,
            map_name,
            2026,
            base_agents,
        )

        base_prediction = (
            base_result["prediction"]["winrate"]
        )

        sentinel_pool = [
            agent
            for agent, role in AGENT_ROLE_MAP.items()
            if role == "sentinel"
        ]

        for sentinel in sentinel_pool:
            if sentinel == "vyse":
                continue

            candidate = [
                sentinel if agent == "vyse" else agent
                for agent in base_agents
            ]

            try:
                result = svc.predict(
                    team,
                    map_name,
                    2026,
                    candidate,
                )

                prediction = (
                    result["prediction"]["winrate"]
                )

                delta = abs(
                    prediction - base_prediction
                )

                print(
                    f"{sentinel:10s} -> "
                    f"{prediction * 100:6.2f}% "
                    f"(delta={delta * 100:5.2f} pts, "
                    f"confidence={result['confidence']['level']})"
                )

            except ValueError:
                continue

    # ==================================================================
    # TEST 5
    # 5 Sentinel
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 5] UNOBSERVED ROLE PATTERN - 5 SENTINEL")
    print("=" * 80)

    five_sentinel = [
        "deadlock",
        "killjoy",
        "sage",
        "cypher",
        "vyse",
    ]

    five_sentinel_result = svc.predict(
        team,
        map_name,
        2026,
        five_sentinel,
    )

    summarize(
        f"{team} 2026 - 5 Sentinel",
        five_sentinel_result,
    )

    # ==================================================================
    # TEST 6
    # Paper Rex
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 6] PAPER REX - HISTORICAL VS SAME-ROLE SWAP")
    print("=" * 80)

    pr_data = predict_historical(
        svc,
        "Paper Rex",
        map_name,
        2026,
    )

    if pr_data is None:
        print("Paper Rex / Ascent / 2026 tidak ditemukan.")
    else:
        pr_row, pr_historical = pr_data
        pr_agents = list(pr_row["Agent"])

        summarize(
            "Paper Rex historical",
            pr_historical,
        )

        # Cari Sentinel dalam composition.
        sentinel_index = None

        for index, agent in enumerate(pr_agents):
            if AGENT_ROLE_MAP.get(agent) == "sentinel":
                sentinel_index = index
                break

        if sentinel_index is not None:
            replacement = "deadlock"

            if replacement == pr_agents[sentinel_index]:
                replacement = "killjoy"

            pr_swap_agents = pr_agents.copy()
            pr_swap_agents[sentinel_index] = replacement

            pr_swap = svc.predict(
                "Paper Rex",
                map_name,
                2026,
                pr_swap_agents,
            )

            summarize(
                "Paper Rex same-role swap",
                pr_swap,
            )

            delta = abs(
                pr_historical["prediction"]["winrate"]
                - pr_swap["prediction"]["winrate"]
            )

            print(
                f"\nPaper Rex prediction delta: "
                f"{delta * 100:.2f} percentage points"
            )

    # ==================================================================
    # TEST 7
    # Context support comparison
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 7] CONTEXT VS COMPOSITION SUPPORT")
    print("=" * 80)

    for year, data in historical_results.items():
        result = data["result"]
        confidence = result["confidence"]

        print(
            f"{year}: "
            f"context={confidence['context_played']} maps | "
            f"composition={confidence['composition_played']} maps | "
            f"confidence={confidence['level']} | "
            f"historical={result['historical']['found']}"
        )

    # ==================================================================
    # TEST 8
    # Contract
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 8] SERVICE CONTRACT")
    print("=" * 80)

    if 2026 in historical_results:
        result = historical_results[2026]["result"]

        assert result["inference"]["feature_count"] == 116
        assert "confidence" in result
        assert "range" in result["prediction"]

        low = result["prediction"]["range"]["low"]
        value = result["prediction"]["winrate"]
        high = result["prediction"]["range"]["high"]

        assert 0 <= low <= value <= high <= 1

        print("Feature count : 116")
        print("Range valid   : True")
        print("Confidence    : True")

    # ==================================================================
    # TEST 9
    # Validation
    # ==================================================================
    print("\n" + "=" * 80)
    print("[TEST 9] INPUT VALIDATION")
    print("=" * 80)

    try:
        svc.predict(
            team,
            map_name,
            2026,
            ["astra", "phoenix"],
        )
    except ValueError as exc:
        print(
            f"Invalid agent count rejected: {exc}"
        )

    try:
        svc.predict(
            team,
            map_name,
            2026,
            [
                "unknown_agent",
                "phoenix",
                "sova",
                "vyse",
                "yoru",
            ],
        )
    except ValueError as exc:
        print(
            f"Unknown agent rejected: {exc}"
        )

    # ==================================================================
    # FINAL
    # ==================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()