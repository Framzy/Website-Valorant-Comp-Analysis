"""
Team Prediction V2 - Service Test
=================================

Run from the project root:

    python -m training_v2.team.analysis.test_team_prediction_service

This test intentionally checks:
1. Service initialization + 116-feature artifact contract.
2. Historical composition path (FOUND).
3. Unseen composition path (NOT FOUND -> inference feature builder).
4. Basic input validation.

The test does not require Flask/API code.
"""

from training_v2.team.service.team_prediction_service import (
    TeamPredictionService,
)


def print_result(label: str, result: dict) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print("historical found :", result["historical"]["found"])
    print("historical WR    :", result["historical"]["winrate"])
    print("strength         :", result["historical"]["composition_strength"])
    print("role pattern     :", result["composition"]["role_pattern"])
    print("prediction       :", result["prediction"]["percentage"], "%")
    print("feature count    :", result["inference"]["feature_count"])


def main() -> None:
    print("INITIALIZE TEAM PREDICTION SERVICE")
    service = TeamPredictionService()
    print("[PASS] Service initialized")
    print(f"[PASS] Feature contract: {service.model.n_features_in_} features")

    # ------------------------------------------------------
    # 1. Historical composition: should use FOUND path.
    # ------------------------------------------------------
    historical = service.predict(
        team="Paper Rex",
        map_name="Abyss",
        year=2024,
        agents=["breach", "kayo", "neon", "omen", "phoenix"],
    )

    print_result("TEST 1 - HISTORICAL COMPOSITION", historical)

    assert historical["historical"]["found"] is True
    assert historical["inference"]["feature_count"] == 116
    assert historical["composition"]["role_pattern"] == "2D-2I-1C-0S"

    print("[PASS] Historical composition path")

    # ------------------------------------------------------
    # 2. Unseen composition: should use fallback builder.
    # ------------------------------------------------------
    unseen = service.predict(
        team="Paper Rex",
        map_name="Abyss",
        year=2024,
        agents=["breach", "kayo", "neon", "omen", "jett"],
    )

    print_result("TEST 2 - UNSEEN COMPOSITION", unseen)

    assert unseen["historical"]["found"] is False
    assert unseen["historical"]["winrate"] is None
    assert unseen["inference"]["feature_count"] == 116
    assert unseen["composition"]["role_pattern"] == "2D-2I-1C-0S"

    print("[PASS] Unseen composition fallback path")

    # ------------------------------------------------------
    # 3. Basic validation.
    # ------------------------------------------------------
    print("\nTEST 3 - INPUT VALIDATION")
    try:
        service.predict(
            team="100 Thieves",
            map_name="Abyss",
            year=2024,
            agents=["jett", "omen", "sova", "cypher"],
        )
    except ValueError as exc:
        print("[PASS] Invalid agent count rejected:", exc)
    else:
        raise AssertionError("Invalid agent count was not rejected.")

    try:
        service.predict(
            team="100 Thieves",
            map_name="Abyss",
            year=2024,
            agents=["jett", "omen", "sova", "cypher", "unknown_agent"],
        )
    except ValueError as exc:
        print("[PASS] Unknown agent rejected:", exc)
    else:
        raise AssertionError("Unknown agent was not rejected.")

    print("\nALL SERVICE TESTS PASSED")


if __name__ == "__main__":
    main()
