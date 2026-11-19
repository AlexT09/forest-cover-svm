"""Dashboard (Dash/Plotly) de clasificación de cobertura forestal.

Pestañas: predicción por lotes (con probabilidades y conjunto conforme), monitoreo de deriva,
resultados del notebook, tarjeta del modelo y documentación de la API.
"""
from __future__ import annotations

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from src.forest import drift as D
from src.forest.features import CLASS_NAMES, CONTINUOUS, FEATURES, TARGET
from src.forest.validation import scores

ART = Path(os.environ.get("FOREST_ARTIFACTS", "artifacts"))
CLASSES = np.arange(1, 8)
NAMES = [CLASS_NAMES[c] for c in CLASSES]
PALETTE = px.colors.qualitative.Safe
COLOR = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(NAMES)}
MAX_ROWS = 1000
TEMPLATE = "plotly_white"


# ----------------------------------------------------------------------------- artefactos
@lru_cache(maxsize=None)
def model(which: str):
    return joblib.load(ART / ("model.joblib" if which == "model" else "model_alt.joblib"))


@lru_cache(maxsize=None)
def conformal() -> dict:
    p = ART / "conformal.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=None)
def results() -> dict:
    p = ART / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=None)
def reference() -> pd.DataFrame:
    return pd.read_csv(ART / "reference.csv.gz")


@lru_cache(maxsize=None)
def sample_batch() -> pd.DataFrame:
    return pd.read_csv(ART / "sample_batch.csv")


def model_label(which: str) -> str:
    c = conformal().get(which, {})
    name = c.get("name", "campeón" if which == "model" else "alternativo")
    role = "Campeón" if which == "model" else "Alternativo"
    return f"{role}: {name} calibrado ({c.get('method', '—')})"


def predict(df: pd.DataFrame, which: str) -> pd.DataFrame:
    m = model(which)
    proba = m.predict_proba(df[FEATURES])
    pred = m.classes_[proba.argmax(axis=1)]
    out = pd.DataFrame(proba, columns=[CLASS_NAMES[c] for c in m.classes_])
    out.insert(0, "predicción", [CLASS_NAMES[c] for c in pred])
    out.insert(1, "clase", pred)
    out.insert(2, "confianza", proba.max(axis=1))
    q = conformal().get(which, {}).get("qhat")
    if q is not None:
        sets = proba >= 1 - q
        sets[np.arange(len(proba)), proba.argmax(axis=1)] = True
        out.insert(3, "conjunto conforme", [", ".join(CLASS_NAMES[c] for c in m.classes_[s]) for s in sets])
        out.insert(4, "tamaño conjunto", sets.sum(axis=1))
        if TARGET in df:
            idx = np.searchsorted(m.classes_, df[TARGET].to_numpy())
            out["cubierto"] = sets[np.arange(len(idx)), idx]
    if TARGET in df:
        out.insert(1, "real", [CLASS_NAMES.get(int(c), "?") for c in df[TARGET]])
    return out


# ----------------------------------------------------------------------------- carga de archivos
def parse_upload(contents: str, filename: str) -> tuple[pd.DataFrame | None, str]:
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        df = pd.read_csv(io.BytesIO(raw))
        if not set(FEATURES) <= set(df.columns):
            df = pd.read_csv(io.BytesIO(raw), header=None)
            if df.shape[1] in (54, 55):
                df.columns = FEATURES + ([TARGET] if df.shape[1] == 55 else [])
            else:
                return None, f"{filename}: se esperan las 54 columnas del conjunto (con o sin encabezado)."
        msg = f"{filename}: {len(df):,} filas leídas."
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            msg += f" Se procesan las primeras {MAX_ROWS} (mismo límite que la API)."
        return df[FEATURES + ([TARGET] if TARGET in df else [])], msg
    except Exception as e:  # noqa: BLE001
        return None, f"No se pudo leer {filename}: {e}"


# ----------------------------------------------------------------------------- componentes
def kpi(title: str, value: str, sub: str = "", color: str = "primary"):
    return dbc.Card(dbc.CardBody([
        html.Div(title, className="text-muted small text-uppercase"),
        html.H3(value, className=f"text-{color} mb-0"),
        html.Div(sub, className="small text-muted"),
    ]), className="h-100 shadow-sm")


def empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 14, "color": "#888"})
    fig.update_layout(template=TEMPLATE, xaxis_visible=False, yaxis_visible=False, height=300)
    return fig


def tab_predict():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("1. Datos"),
                dcc.Upload(id="upload", multiple=False, children=html.Div([
                    "Arrastra o ", html.A("selecciona un CSV", className="link-primary"),
                    html.Div("54 variables de Covertype (opcional: Cover_Type para medir aciertos). Máx. 1 000 filas.",
                             className="small text-muted")]),
                    style={"borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
                           "padding": "18px", "textAlign": "center", "cursor": "pointer"}),
                dbc.ButtonGroup([
                    dbc.Button("Lote de ejemplo (TEST)", id="btn-sample", color="secondary", outline=True, size="sm"),
                    dbc.Button("Ejemplo con deriva (+0.5 DE elevación)", id="btn-drift", color="warning", outline=True, size="sm"),
                ], className="mt-2"),
                html.Div(id="upload-msg", className="small mt-2"),
            ], md=6),
            dbc.Col([
                html.H5("2. Modelo"),
                dbc.RadioItems(id="which", value="model", options=[
                    {"label": model_label("model"), "value": "model"},
                    {"label": model_label("model_alt"), "value": "model_alt"}]),
                html.Div(id="conf-info", className="small text-muted mt-2"),
                dbc.Button("Descargar predicciones (CSV)", id="btn-download", color="primary", size="sm", className="mt-3"),
                dcc.Download(id="download"),
            ], md=6),
        ], className="mb-3"),
        dbc.Row(id="kpis", className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="fig-dist"), md=6),
            dbc.Col(dcc.Graph(id="fig-conf"), md=6),
        ]),
        dbc.Row([dbc.Col(dcc.Graph(id="fig-heat"), md=12)]),
        html.H5("Detalle por fila", className="mt-3"),
        dash_table.DataTable(id="table", page_size=12, sort_action="native", filter_action="native",
                             style_table={"overflowX": "auto"},
                             style_cell={"fontFamily": "Inter, system-ui, sans-serif", "fontSize": 13, "padding": "4px"},
                             style_header={"fontWeight": "600", "backgroundColor": "#f3f5f9"}),
    ], fluid=True, className="py-3")


def tab_drift():
    return dbc.Container([
        dbc.Alert("Compara el lote cargado con la referencia del modelo (10 000 parcelas de calibración, muestra de 5 000). "
                  "Política: alerta si PSI > 0.2 en alguna variable continua, si KS es significativa tras Holm (α=0.05) "
                  "o si la divergencia JS de las clases predichas supera 0.05.", color="light", className="small"),
        dbc.Row([
            dbc.Col([html.Label("Simular desplazamiento de la elevación (desviaciones estándar)"),
                     dcc.Slider(id="delta", min=-1, max=1, step=0.05, value=0,
                                marks={v: str(v) for v in [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]})], md=8),
            dbc.Col(html.Div(id="drift-badge", className="mt-3"), md=4),
        ]),
        dbc.Row(id="drift-kpis", className="g-3 my-2"),
        dbc.Row([dbc.Col(dcc.Graph(id="fig-psi"), md=7), dbc.Col(dcc.Graph(id="fig-classdist"), md=5)]),
        dbc.Row([dbc.Col(dcc.Graph(id="fig-elev"), md=7), dbc.Col(html.Div(id="ks-table"), md=5)]),
    ], fluid=True, className="py-3")


def _records(key):
    return pd.DataFrame(results().get(key) or [])


def tab_results():
    r = results()
    if "test_metrics" not in r:
        return dbc.Alert("Aún no hay resultados: ejecuta el notebook para generar artifacts/results.json.", color="warning")
    figs = []

    tm = _records("test_metrics")
    ci = _records("test_ci")
    c1 = ci[ci.metric == "macro_f1"].sort_values("estimate")
    f = go.Figure(go.Scatter(x=c1.estimate, y=c1.modelo, mode="markers", marker={"size": 11},
                             error_x={"type": "data", "symmetric": False, "array": c1.ci_high - c1.estimate,
                                      "arrayminus": c1.estimate - c1.ci_low}))
    f.update_layout(template=TEMPLATE, title="Macro-F1 en TEST (bloques no vistos) con IC 95 % bootstrap", height=330,
                    xaxis_title="macro-F1")
    figs.append(dcc.Graph(figure=f))

    vs = _records("validation_schemes")
    if len(vs):
        f = px.bar(vs, x="model", y="mean", color="scheme", barmode="group", error_y="std", template=TEMPLATE,
                   title="PI2 · Macro-F1 según el esquema de validación (media ± DE)",
                   labels={"mean": "macro-F1", "model": "", "scheme": "esquema"})
        figs.append(dcc.Graph(figure=f))

    sc = _records("scalability")
    if len(sc):
        bmap = {d["kernel"]: d["b (ajuste)"] for d in r.get("complexity_fit", [])}
        sc["serie"] = sc.kernel.map(lambda k: f"{k} (b = {bmap.get(k, float('nan')):.2f})")
        f = px.line(sc, x="n", y="fit_s", color="serie", markers=True, log_x=True, log_y=True, template=TEMPLATE,
                    title="PI1 · Tiempo de ajuste de la SVM exacta (log–log): log t = a + b log n",
                    labels={"fit_s": "segundos", "serie": "kernel"})
        figs.append(dcc.Graph(figure=f))

    ap = _records("approximations")
    if len(ap):
        f = px.scatter(ap, x="fit_s", y="macro_f1", color="familia", text="modelo", log_x=True, template=TEMPLATE,
                       title="PI1 · Compromiso costo–desempeño (validación por bloques)",
                       labels={"fit_s": "tiempo de ajuste (s)", "macro_f1": "macro-F1"})
        f.update_traces(textposition="top center", textfont_size=9)
        figs.append(dcc.Graph(figure=f))

    cc = _records("conformal_class")
    if len(cc):
        f = px.bar(cc, x="clase", y="cobertura", color="conjunto", barmode="group", facet_col="modelo", template=TEMPLATE,
                   title="PI3 · Cobertura conforme por clase (α = 0.1)", category_orders={"clase": NAMES})
        f.add_hline(y=0.9, line_dash="dash", line_color="red")
        figs.append(dcc.Graph(figure=f))

    dp = _records("drift_power")
    if len(dp):
        m = dp.melt(id_vars="escenario", value_vars=["alerta PSI", "alerta KS", "alerta JS", "alerta"])
        f = px.bar(m, x="escenario", y="value", color="variable", barmode="group", template=TEMPLATE,
                   title="PI4 · Tasa de alerta por ventana de 500 peticiones", labels={"value": "tasa", "variable": "detector"})
        figs.append(dcc.Graph(figure=f))

    tables = []
    for title, key, cols in [
        ("Métricas en TEST e IID", "test_metrics", None),
        ("Validación anidada 5×3 por bloques", "nested", ["modelo", "pliegue", "macro_f1", "balanced_accuracy", "mcc", "tiempo_s"]),
        ("Calibración", "calibration", None),
        ("Predicción conforme", "conformal", None),
        ("Tiempo hasta la alerta", "drift_tta", None),
        ("Nadeau–Bengio (validación por bloques)", "nadeau_bengio", None),
        ("McNemar en TEST", "mcnemar", None),
    ]:
        d = _records(key)
        if not len(d):
            continue
        d = d[cols] if cols else d
        tables.append(html.H6(title, className="mt-4"))
        tables.append(dash_table.DataTable(
            data=d.round(4).to_dict("records"), columns=[{"name": c, "id": c} for c in d.columns],
            style_table={"overflowX": "auto"}, page_size=12,
            style_cell={"fontSize": 12, "fontFamily": "Inter, system-ui, sans-serif", "padding": "3px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f3f5f9"}))

    head = dbc.Row([
        dbc.Col(kpi("Modelo campeón", r.get("champion", "—"), "validación anidada por bloques + desempate por costo"), md=4),
        dbc.Col(kpi("macro-F1 TEST", f"{tm.iloc[0]['macro-F1 TEST']:.3f}", f"modelo {tm.iloc[0]['modelo']}"), md=4),
        dbc.Col(kpi("Particiones", f"{sum((r.get('partitions') or {}).values()):,}", "filas en DEV-TRAIN, CAL, IID y TEST"), md=4),
    ], className="g-3 mb-2")
    return dbc.Container([head] + [dbc.Row([dbc.Col(g, md=6) for g in figs[i:i + 2]]) for i in range(0, len(figs), 2)]
                         + tables, fluid=True, className="py-3")


def tab_card():
    p = ART / "model_card.md"
    return dbc.Container(dcc.Markdown(p.read_text(encoding="utf-8") if p.exists() else "Sin tarjeta.", className="py-3"), fluid=True)


def tab_api(base: str):
    ex = '{"instances": [[2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, 1, 0, 0, 0' + ", 0" * 40 + "]]}"
    ex = ex.replace(", 0" * 40 + "]]", ", 0" * 28 + ", 1" + ", 0" * 11 + "]]")
    return dbc.Container(dcc.Markdown(f"""
### API de inferencia (FastAPI)
El mismo servicio expone la API REST junto al dashboard:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| POST | `/predict` | Lote de 1 a 1 000 filas con 54 valores: clase, probabilidades calibradas y conjunto conforme |
| GET | `/docs` | Documentación interactiva (Swagger) |

```bash
curl -X POST {base}/predict -H "Content-Type: application/json" \\
     -d '{ex}'
```
""", className="py-3"), fluid=True)


# ----------------------------------------------------------------------------- aplicación
def create_dash(requests_pathname_prefix: str = "/", api_base: str = "") -> Dash:
    app = Dash(__name__, requests_pathname_prefix=requests_pathname_prefix, title="Cobertura forestal · Clasificación multiclase",
               external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
    r = results()
    subtitle = (f"Campeón: {r.get('champion', '—')} · alternativo: {r.get('alternative', '—')} · "
                "validación por bloques pseudo-espaciales · probabilidades calibradas · predicción conforme α = 0.1")
    app.layout = html.Div([
        dbc.Navbar(dbc.Container([
            html.Div([html.H4("Clasificación multiclase de cobertura forestal", className="text-white mb-0"),
                      html.Div(subtitle, className="small text-white-50")]),
        ], fluid=True), color="primary", dark=True, className="mb-2"),
        dcc.Store(id="batch"),
        dbc.Tabs([
            dbc.Tab(tab_predict(), label="Predicción por lotes"),
            dbc.Tab(tab_drift(), label="Monitoreo de deriva"),
            dbc.Tab(tab_results(), label="Resultados del proyecto"),
            dbc.Tab(tab_card(), label="Tarjeta del modelo"),
            dbc.Tab(tab_api(api_base or "https://<servicio>"), label="API"),
        ], className="px-2"),
    ])

    @app.callback(Output("batch", "data"), Output("upload-msg", "children"),
                  Input("upload", "contents"), Input("btn-sample", "n_clicks"), Input("btn-drift", "n_clicks"),
                  State("upload", "filename"), prevent_initial_call=False)
    def load_batch(contents, n1, n2, filename):
        from dash import ctx
        return load_batch_logic(ctx.triggered_id, contents, filename)

    @app.callback(Output("kpis", "children"), Output("fig-dist", "figure"), Output("fig-conf", "figure"),
                  Output("fig-heat", "figure"), Output("table", "data"), Output("table", "columns"),
                  Output("conf-info", "children"), Input("batch", "data"), Input("which", "value"))
    def update_predictions(data, which):
        return predictions_view(data, which)

    @app.callback(Output("download", "data"), Input("btn-download", "n_clicks"), State("batch", "data"),
                  State("which", "value"), prevent_initial_call=True)
    def download(n, data, which):
        if not data:
            return no_update
        df = pd.read_json(io.StringIO(data), orient="split")
        return dcc.send_data_frame(predict(df, which).to_csv, "predicciones_cobertura.csv", index=False)

    @app.callback(Output("drift-kpis", "children"), Output("drift-badge", "children"), Output("fig-psi", "figure"),
                  Output("fig-classdist", "figure"), Output("fig-elev", "figure"), Output("ks-table", "children"),
                  Input("batch", "data"), Input("delta", "value"))
    def update_drift(data, delta):
        return drift_view(data, delta)

    return app


def load_batch_logic(trig, contents, filename):
    if trig == "upload" and contents:
        df, msg = parse_upload(contents, filename or "archivo")
        if df is None:
            return no_update, dbc.Alert(msg, color="danger", className="py-1")
        return df.to_json(orient="split"), dbc.Alert(msg, color="success", className="py-1")
    df = sample_batch()
    if trig == "btn-drift":
        sd = results().get("drift_config", {}).get("sd_elevation", 280.0)
        df = D.shift_feature(df, "Elevation", 0.5, sd)
        return df.to_json(orient="split"), dbc.Alert("Lote de ejemplo con la elevación desplazada +0.5 DE "
                                                     "(revisa la pestaña de deriva).", color="warning", className="py-1")
    return df.to_json(orient="split"), dbc.Alert(f"Lote de ejemplo: {len(df)} parcelas de los bloques de prueba "
                                                 "(con etiqueta real).", color="info", className="py-1")

def predictions_view(data, which):
    c = conformal().get(which, {})
    info = (f"q̂ = {c.get('qhat', float('nan')):.3f} (α = {c.get('alpha', 0.1)}): el conjunto incluye toda clase con "
            f"p ≥ {1 - c.get('qhat', 0):.3f}. Garantía marginal de cobertura ≥ {1 - c.get('alpha', 0.1):.0%} bajo intercambiabilidad.")
    if not data:
        e = empty_fig("Carga un CSV o usa el lote de ejemplo")
        return [], e, e, e, [], [], info
    df = pd.read_json(io.StringIO(data), orient="split")
    out = predict(df, which)
    cards = [dbc.Col(kpi("Parcelas", f"{len(out):,}"), md=2),
             dbc.Col(kpi("Clase más frecuente", out["predicción"].mode()[0]), md=3),
             dbc.Col(kpi("Confianza media", f"{out['confianza'].mean():.2f}"), md=2)]
    if "tamaño conjunto" in out:
        cards.append(dbc.Col(kpi("Tamaño medio conjunto", f"{out['tamaño conjunto'].mean():.2f}",
                                 f"{(out['tamaño conjunto'] == 1).mean():.0%} con una sola clase"), md=2))
    if "real" in out:
        s = scores(df[TARGET].to_numpy(), out["clase"].to_numpy())
        sub = f"exactitud {s['accuracy']:.3f}"
        if "cubierto" in out:
            sub += f" · cobertura conforme {out['cubierto'].mean():.3f}"
        cards.append(dbc.Col(kpi("macro-F1 del lote", f"{s['macro_f1']:.3f}", sub, "success"), md=3))

    dist = out["predicción"].value_counts().reindex(NAMES, fill_value=0).reset_index()
    dist.columns = ["clase", "n"]
    f1 = px.bar(dist, x="clase", y="n", color="clase", color_discrete_map=COLOR, template=TEMPLATE,
                title="Distribución de clases predichas")
    if "real" in out:
        real = out["real"].value_counts().reindex(NAMES, fill_value=0)
        f1.add_scatter(x=NAMES, y=real.values, mode="markers", marker={"symbol": "diamond", "size": 11, "color": "black"},
                       name="real")
    f1.update_layout(showlegend=False, height=360, xaxis_title="")

    if "tamaño conjunto" in out:
        f2 = px.histogram(out, x="confianza", color=out["tamaño conjunto"].astype(str), nbins=25, template=TEMPLATE,
                          title="Confianza (p máx.) según el tamaño del conjunto conforme",
                          labels={"color": "tamaño"}, barmode="stack")
    else:
        f2 = px.histogram(out, x="confianza", nbins=25, template=TEMPLATE, title="Confianza (p máx.)")
    f2.update_layout(height=360)

    top = out.head(60)
    pcols = [n for n in NAMES if n in out]
    f3 = go.Figure(go.Heatmap(z=top[pcols].to_numpy().T, x=[f"#{i}" for i in top.index], y=pcols, colorscale="Blues",
                              zmin=0, zmax=1, colorbar={"title": "p"}))
    f3.update_layout(template=TEMPLATE, title="Probabilidades calibradas por clase (primeras 60 filas)", height=330)

    tbl = out.copy()
    tbl.insert(0, "fila", range(len(tbl)))
    for col in pcols + ["confianza"]:
        tbl[col] = tbl[col].round(3)
    tbl = tbl.drop(columns=["clase"])
    if "cubierto" in tbl:
        tbl["cubierto"] = tbl["cubierto"].map({True: "sí", False: "no"})
    return cards, f1, f2, f3, tbl.to_dict("records"), [{"name": col, "id": col} for col in tbl.columns], info

def drift_view(data, delta):
    if not data:
        e = empty_fig("Carga un lote en la pestaña de predicción")
        return [], "", e, e, e, ""
    df = pd.read_json(io.StringIO(data), orient="split")
    ref = reference()
    sd = results().get("drift_config", {}).get("sd_elevation", float(ref["Elevation"].std()))
    if delta:
        df = D.shift_feature(df, "Elevation", delta, sd)
    pred = model("model").predict(df[FEATURES])
    mon = D.DriftMonitor(ref[CONTINUOUS], CONTINUOUS, ref["pred"].to_numpy(), CLASSES)
    r = mon.check(df[CONTINUOUS], pred)
    tab = r["table"].sort_values("psi", ascending=False)

    badge = dbc.Alert("ALERTA de deriva" if r["alert"] else "Sin deriva relevante",
                      color="danger" if r["alert"] else "success", className="text-center fw-bold py-2")
    k = [dbc.Col(kpi("PSI máximo", f"{r['max_psi']:.3f}", r["psi_feature"], "danger" if r["psi_alert"] else "primary"), md=3),
         dbc.Col(kpi("p KS mín. (Holm)", f"{r['min_p_holm']:.2g}", "α = 0.05", "danger" if r["ks_alert"] else "primary"), md=3),
         dbc.Col(kpi("JS clases predichas", f"{r['js']:.3f}", "umbral 0.05", "danger" if r["js_alert"] else "primary"), md=3),
         dbc.Col(kpi("Tamaño del lote", f"{len(df):,}", "ventana de referencia: 500"), md=3)]

    f1 = px.bar(tab, x="psi", y="feature", orientation="h", template=TEMPLATE, title="PSI por variable continua",
                color=np.where(tab.psi > 0.2, "alerta", np.where(tab.psi > 0.1, "moderado", "estable")),
                color_discrete_map={"alerta": "#d62728", "moderado": "#ff7f0e", "estable": "#2ca02c"},
                labels={"feature": "", "color": ""})
    for x, dash_ in [(0.1, "dot"), (0.2, "dash"), (0.25, "solid")]:
        f1.add_vline(x=x, line_dash=dash_, line_color="gray")
    f1.update_layout(height=420, yaxis={"categoryorder": "total ascending"})

    pref = D.class_distribution(ref["pred"], CLASSES)
    pcur = D.class_distribution(pred, CLASSES)
    f2 = go.Figure([go.Bar(x=NAMES, y=pref, name="referencia"), go.Bar(x=NAMES, y=pcur, name="lote")])
    f2.update_layout(template=TEMPLATE, barmode="group", title=f"Clases predichas (JS = {r['js']:.3f})", height=420)

    f3 = go.Figure([go.Histogram(x=ref["Elevation"], name="referencia", histnorm="probability density", opacity=.6, nbinsx=40),
                    go.Histogram(x=df["Elevation"], name="lote", histnorm="probability density", opacity=.6, nbinsx=40)])
    f3.update_layout(template=TEMPLATE, barmode="overlay", title="Distribución de la elevación", height=360)

    t = tab[["feature", "psi", "ks_stat", "p_value", "p_holm"]].round(4)
    table = dash_table.DataTable(data=t.to_dict("records"), columns=[{"name": c, "id": c} for c in t.columns],
                                 style_cell={"fontSize": 12, "padding": "3px"},
                                 style_header={"fontWeight": "600", "backgroundColor": "#f3f5f9"},
                                 style_data_conditional=[{"if": {"filter_query": "{p_holm} < 0.05", "column_id": "p_holm"},
                                                          "color": "#d62728", "fontWeight": "600"}])
    return k, badge, f1, f2, f3, html.Div([html.H6("Kolmogórov–Smirnov con corrección de Holm"), table])


if __name__ == "__main__":
    create_dash().run(debug=False, port=int(os.environ.get("PORT", 8050)))
