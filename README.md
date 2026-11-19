# Clasificación multiclase de cobertura forestal a gran escala
### Kernels, validación por bloques, incertidumbre calibrada y operación monitoreada

Clasificación multiclase de las 7 coberturas del conjunto **Forest CoverType (UCI)**: 581 012 parcelas y 54 variables.

**Alcance:** núcleo obligatorio (fases 0–5 y 9) más las extensiones **6** (calibración y predicción conforme) y
**10** (monitoreo y deriva).

**Dashboard desplegado:** <https://forest-cover-svm.onrender.com> · API: [`/health`](https://forest-cover-svm.onrender.com/health), `/predict`, documentación en [`/docs`](https://forest-cover-svm.onrender.com/docs) (plan gratuito: la primera visita puede tardar ≈ 1 min)

| Entregable | Ubicación |
|---|---|
| Notebook ejecutado (flujo completo con salidas y figuras) | [`notebooks/clasificacion_cobertura_forestal.ipynb`](notebooks/clasificacion_cobertura_forestal.ipynb) |
| Biblioteca modular | `src/forest/` (`features`, `validation`, `models`, `calibration`, `drift`, `api`) |
| API FastAPI (`/health`, `/predict`) | `src/forest/api.py` |
| Dashboard Dash/Plotly | `app/dashboard.py` (se sirve junto con la API en `app/server.py`) |
| Tarjeta del modelo | `artifacts/model_card.md` |
| Configuración, semillas y entorno | `config/config.yaml`, `requirements.lock`, `requirements-dev.lock` (dependencias directas en `requirements*.in`) |
| Pruebas y CI | `tests/`, `.github/workflows/ci.yml`, `Dockerfile` |

## Estructura

```
forest-cover-svm/
├── config/config.yaml          # protocolo: particiones, semillas, tamaños, umbrales
├── data/                       # covtype.data.gz (no se sube a Git; SHA-256 en config.yaml)
├── notebooks/clasificacion_cobertura_forestal.ipynb
├── src/forest/                 # biblioteca
├── app/dashboard.py, server.py # dashboard + servicio único (API + dashboard)
├── artifacts/                  # modelo calibrado, conformal.json, referencia de deriva, results.json, model_card.md
├── reports/figures/            # figuras exportadas por el notebook
├── scripts/                    # build_notebook.py, run_notebook.py
├── tests/                      # no fuga, contrato de la API y unidades de la biblioteca
├── Dockerfile, .github/workflows/ci.yml, requirements.lock
```

## Reproducir

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.lock
# Copiar covtype.data.gz (UCI) en data/ (ver data/README.md) — el notebook verifica su SHA-256
python scripts/build_notebook.py                      # genera el .ipynb desde el código fuente
python scripts/run_notebook.py                        # ejecuta todo y guarda las salidas (≈80 min en un portátil de 4 núcleos)
FOREST_PROFILE=smoke python scripts/run_notebook.py   # prueba rápida de punta a punta (≈9 min, no sobrescribe el notebook)
pytest --cov=src                                      # pruebas
uvicorn app.server:app --port 8000                    # API + dashboard en http://localhost:8000
```

Las ejecuciones quedan registradas en MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Protocolo (resumen)
- **Métrica primaria:** macro-F1. Secundarias: balanced accuracy, MCC, recall por clase y log-loss.
- **Prueba:** bloques pseudo-espaciales completos (KMeans k=20 sobre elevación, distancia a vías y a focos de incendio), 1 de 5 pliegues de `StratifiedGroupKFold`.
- **Reservas del desarrollo:** CAL (10 000 filas, calibración conforme) e IID (20 000 filas, prueba dentro de distribución).
- **Campeón:** mayor macro-F1 en validación anidada 5×3 por bloques; si la diferencia con el segundo no es significativa
  (t corregida de Nadeau–Bengio, p ≥ 0.05), el más barato de servir (latencia de predicción). Solo con datos de desarrollo.

## Despliegue (Render)
El `Dockerfile` construye un servicio único (API + dashboard) que escucha en `$PORT`; `render.yaml` lo declara como
servicio web gratuito con comprobación de salud en `/health`.

1. En [render.com](https://render.com) → **New → Blueprint** → elegir este repositorio (lee `render.yaml`).
2. Al terminar la construcción, el servicio queda en `https://<nombre>.onrender.com` (dashboard en `/`, API en `/docs`).
3. Opcional: variable `PUBLIC_URL=https://<nombre>.onrender.com` para que la pestaña "API" del dashboard muestre la URL real.

En el plan gratuito el servicio se suspende tras 15 min sin tráfico; la primera visita tarda ≈ 1 min en despertarlo.

## Limitaciones declaradas
Los bloques son pseudo-espaciales, porque el conjunto no trae coordenadas. La SVM exacta se entrena con una submuestra por su costo ≈ cuadrático. La memoria del equipo de desarrollo (7 GB) limita algunos tamaños, documentados en `config/config.yaml`.
