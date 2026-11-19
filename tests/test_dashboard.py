"""Pruebas del dashboard y del servicio combinado (API + Dash) con los artefactos de juguete."""
import base64

from fastapi.testclient import TestClient

from tests.conftest import toy_frame


def test_dashboard_views():
    from app import dashboard as dsh

    data, msg = dsh.load_batch_logic(None, None, None)
    assert data and "ejemplo" in str(msg)
    data_d, _ = dsh.load_batch_logic("btn-drift", None, None)
    for which in ["model", "model_alt"]:
        cards, f1, f2, f3, rows, cols, info = dsh.predictions_view(data, which)
        assert len(rows) == 50 and any(c["id"] == "conjunto conforme" for c in cols) and "q̂" in info
    k, badge, *_ = dsh.drift_view(data_d, 0.5)
    assert len(k) == 4
    empty = dsh.predictions_view(None, "model")
    assert empty[4] == []
    csv = toy_frame(10).to_csv(index=False).encode()
    df, msg = dsh.parse_upload("data:text/csv;base64," + base64.b64encode(csv).decode(), "x.csv")
    assert df is not None and len(df) == 10
    bad, msg = dsh.parse_upload("data:text/csv;base64," + base64.b64encode(b"a,b\n1,2\n").decode(), "y.csv")
    assert bad is None
    assert dsh.tab_results() is not None and dsh.tab_card() is not None


def test_combined_server_serves_dashboard_and_api():
    from app.server import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.get("/")
        assert r.status_code == 200 and "Dash" in r.text or "_dash" in r.text
        assert client.get("/_dash-layout").status_code == 200
