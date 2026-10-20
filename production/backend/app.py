from flask import Flask
from flask_cors import CORS

from backend.routes.api_routes import api_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(api_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run()
