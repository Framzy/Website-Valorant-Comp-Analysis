from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from backend.services.general_analysis_service import GeneralAnalysisService
from backend.services.team_prediction_service import TeamPredictionService


# ============================================================
# APP CONFIGURATION
# ============================================================

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "app"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    template_folder=str(FRONTEND_DIR),
)

CORS(app)

# ============================================================
# SERVICES
# ============================================================

general_service = GeneralAnalysisService()
team_service = TeamPredictionService()


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
# TEAM OPTIONS API
# ============================================================

@app.get("/api/team/options/years")
def get_team_years():
    """Return available Team V2 years."""
    try:
        years = team_service.get_available_years()
        return jsonify({"years": years}), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


@app.get("/api/team/options/maps")
def get_team_maps():
    """Return available Team V2 maps for a year."""
    try:
        year = request.args.get("year")

        if year is None:
            return jsonify({"error": "Year is required."}), 400

        maps = team_service.get_available_maps(year)

        return jsonify({
            "year": int(year),
            "maps": maps,
        }), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


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
# FRONTEND
# ============================================================

@app.get("/")
def serve_frontend():
    """Serve the frontend application."""
    return send_from_directory(
        FRONTEND_DIR,
        "index.html",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
