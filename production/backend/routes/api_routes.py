from flask import Blueprint

from backend.controllers.general_controller import analyze_general
from backend.controllers.options_controller import (
    get_available_maps,
    get_available_years,
    get_team_teams,
)
from backend.controllers.team_controller import predict_team

api_bp = Blueprint("api", __name__)

api_bp.add_url_rule(
    "/api/options/years",
    view_func=get_available_years,
    methods=["GET"],
)
api_bp.add_url_rule(
    "/api/options/maps",
    view_func=get_available_maps,
    methods=["GET"],
)
api_bp.add_url_rule(
    "/api/general/analyze",
    view_func=analyze_general,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/api/team/options/teams",
    view_func=get_team_teams,
    methods=["GET"],
)
api_bp.add_url_rule(
    "/api/team/predict",
    view_func=predict_team,
    methods=["POST"],
)
