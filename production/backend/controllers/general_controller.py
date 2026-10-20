from flask import jsonify, request

from backend.services.general_analysis_service import GeneralAnalysisService

general_service = GeneralAnalysisService()


def analyze_general():
    """Analyze a General composition."""
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        required_fields = ("map", "year", "agents")
        missing_fields = [field for field in required_fields if field not in data]

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
