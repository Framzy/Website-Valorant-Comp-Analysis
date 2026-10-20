"""Automated API regression tests for Valorant Predictor.

Run from the production directory:
    python -m unittest backend.tests.test_api -v

These tests use Flask's test client and mock service methods so API routing,
request validation, response formatting, and HTTP status codes can be tested
without running real ML inference for every test.
"""

import unittest
from unittest.mock import patch

from backend.app import create_app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config.update(TESTING=True)
        cls.client = cls.app.test_client()

    def test_get_available_years(self):
        with patch(
            "backend.controllers.options_controller.option_service.get_available_years",
            return_value=[2024, 2025, 2026],
        ):
            response = self.client.get("/api/options/years")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"years": [2024, 2025, 2026]})

    def test_get_available_maps_success(self):
        with patch(
            "backend.controllers.options_controller.option_service.get_available_maps",
            return_value=["Abyss", "Ascent"],
        ) as service:
            response = self.client.get("/api/options/maps?year=2024")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"year": 2024, "maps": ["Abyss", "Ascent"]},
        )
        service.assert_called_once_with("2024")

    def test_get_available_maps_missing_year(self):
        response = self.client.get("/api/options/maps")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Year is required."})

    def test_get_available_maps_invalid_year(self):
        with patch(
            "backend.controllers.options_controller.option_service.get_available_maps",
            side_effect=ValueError("Year must be an integer."),
        ):
            response = self.client.get("/api/options/maps?year=abc")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Year must be an integer."},
        )

    def test_get_team_teams_success(self):
        with patch(
            "backend.controllers.options_controller.team_service.get_available_teams",
            return_value=["FNATIC", "G2 Esports"],
        ) as service:
            response = self.client.get(
                "/api/team/options/teams?year=2024&map=Abyss"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "year": 2024,
                "map": "Abyss",
                "teams": ["FNATIC", "G2 Esports"],
            },
        )
        service.assert_called_once_with(year="2024", map_name="Abyss")

    def test_get_team_teams_missing_parameters(self):
        response = self.client.get("/api/team/options/teams?map=Abyss")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Missing required query parameters: year"},
        )

    def test_general_analysis_success(self):
        expected = {"input": {"map": "Abyss", "year": 2024, "agents": ["jett"]}}
        with patch(
            "backend.controllers.general_controller.general_service.analyze_composition",
            return_value=expected,
        ) as service:
            response = self.client.post(
                "/api/general/analyze",
                json={"map": "Abyss", "year": 2024, "agents": ["jett"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        service.assert_called_once_with(
            map_name="Abyss",
            year=2024,
            agents=["jett"],
        )

    def test_general_analysis_non_object_body(self):
        response = self.client.post(
            "/api/general/analyze",
            json=["not", "an", "object"],
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Request body must be a JSON object."},
        )

    def test_general_analysis_missing_field(self):
        response = self.client.post(
            "/api/general/analyze",
            json={"map": "Abyss", "year": 2024},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Missing required fields: agents"},
        )

    def test_team_prediction_success(self):
        expected = {"prediction": {"winrate": 0.5, "percentage": 50.0}}
        with patch(
            "backend.controllers.team_controller.team_service.predict",
            return_value=expected,
        ) as service:
            response = self.client.post(
                "/api/team/predict",
                json={
                    "team": "FNATIC",
                    "map": "Abyss",
                    "year": 2024,
                    "agents": ["cypher", "jett", "kayo", "omen", "sova"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)
        service.assert_called_once_with(
            team="FNATIC",
            map_name="Abyss",
            year=2024,
            agents=["cypher", "jett", "kayo", "omen", "sova"],
        )

    def test_team_prediction_missing_field(self):
        response = self.client.post(
            "/api/team/predict",
            json={"team": "FNATIC", "map": "Abyss", "year": 2024},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Missing required fields: agents"},
        )

    def test_unknown_endpoint(self):
        response = self.client.get("/api/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_wrong_method(self):
        response = self.client.get("/api/team/predict")

        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
