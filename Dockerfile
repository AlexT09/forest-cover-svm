# Imagen mínima y sin privilegios para la API + dashboard.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /srv
# Dependencias primero para aprovechar la caché de capas
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
# Código de la biblioteca, la app y los artefactos versionados del modelo
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os, urllib.request as u; u.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1
# Railway/Render inyectan PORT; en local se usa 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
