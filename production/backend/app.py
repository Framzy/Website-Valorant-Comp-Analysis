from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.services.shared_options_service import SharedOptionsService
from backend.services.general_analysis_service import GeneralAnalysisService
from backend.services.team_prediction_service import TeamPredictionService


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# SERVICES
# ============================================================

option_service = SharedOptionsService()
general_service = GeneralAnalysisService()
team_service = TeamPredictionService()


# ============================================================
# SHARED OPTION API
# ============================================================

@app.get("/api/options/years")
def get_available_years():
    """Return years shared by General V2 and Team V2."""
    try:
        years = option_service.get_available_years()
        return jsonify({"years": years}), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


@app.get("/api/options/maps")
def get_available_maps():
    """Return maps shared by General V2 and Team V2 for a year."""
    try:
        year = request.args.get("year")

        if year is None:
            return jsonify({"error": "Year is required."}), 400

        maps = option_service.get_available_maps(year)

        return jsonify({
            "year": int(year),
            "maps": maps,
        }), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


# ============================================================
# GENERAL ANALYSIS API
# ============================================================

@app.post("/api/general/analyze")
def analyze_general():
    """Analyze a General composition."""
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        required_fields = ("map", "year", "agents")
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": (
                    "Missing required fields: "
                    + ", ".join(missing_fields)
                )
            }), 400

        result = general_service.analyze_composition(
            map_name=data["map"],
            year=data["year"],
            agents=data["agents"],
        )

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


# ============================================================
# TEAM-SPECIFIC OPTIONS API
# ============================================================

@app.get("/api/team/options/teams")
def get_team_teams():
    """Return available Team V2 teams for a year and map."""
    try:
        year = request.args.get("year")
        map_name = request.args.get("map")

        missing_fields = []

        if year is None:
            missing_fields.append("year")

        if map_name is None:
            missing_fields.append("map")

        if missing_fields:
            return jsonify({
                "error": (
                    "Missing required query parameters: "
                    + ", ".join(missing_fields)
                )
            }), 400

        teams = team_service.get_available_teams(
            year=year,
            map_name=map_name,
        )

        return jsonify({
            "year": int(year),
            "map": map_name,
            "teams": teams,
        }), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


# ============================================================
# TEAM PREDICTION API
# ============================================================

@app.post("/api/team/predict")
def predict_team():
    """Run Team Prediction V2."""
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        required_fields = ("team", "map", "year", "agents")
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": (
                    "Missing required fields: "
                    + ", ".join(missing_fields)
                )
            }), 400

        result = team_service.predict(
            team=data["team"],
            map_name=data["map"],
            year=data["year"],
            agents=data["agents"],
        )

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run()