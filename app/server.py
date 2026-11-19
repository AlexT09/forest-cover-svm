"""Servicio único desplegable: API FastAPI (/health, /predict, /docs) + dashboard Dash en "/".

Arranque:  uvicorn app.server:app --host 0.0.0.0 --port $PORT
"""
import os

from fastapi.middleware.wsgi import WSGIMiddleware

from app.dashboard import create_dash
from src.forest.api import app

dash_app = create_dash(requests_pathname_prefix="/", api_base=os.environ.get("PUBLIC_URL", ""))
# Las rutas de la API se registraron antes, así que tienen prioridad sobre el montaje del dashboard.
app.mount("/", WSGIMiddleware(dash_app.server))
