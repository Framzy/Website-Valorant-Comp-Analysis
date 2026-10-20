from flask import jsonify, request

from backend.services.shared_options_service import SharedOptionsService
from backend.services.team_prediction_service import TeamPredictionService

option_service = SharedOptionsService()
team_service = TeamPredictionService()


def get_available_years():
    """Return years shared by General V2 and Team V2."""
    try:
        years = option_service.get_available_years()
        return jsonify({"years": years}), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        return jsonify({"error": "Internal server error."}), 500


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
