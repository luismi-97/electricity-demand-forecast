import copy

import numpy as np
import pandas as pd
import pytest

from edf import db, evaluation as ev, serving, weather
from edf.config import CITIES
from edf.features import load_dataset
from edf.frozen import FROZEN
from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster
from edf.models.baselines import SeasonalNaive
from edf.train_final import train_final
from tests.conftest import make_df


def test_save_load_roundtrip_gives_identical_predictions(tmp_path):
    df = make_df("2021-01-01", "2023-12-31 23:00", noise=300.0)
    m = LGBMForecaster(n_estimators=40, num_leaves=15).fit(df)
    q = LGBMQuantileForecaster(n_estimators=40, num_leaves=15).fit(df)
    m.save(tmp_path / "m.joblib"); q.save(tmp_path / "q.joblib")
    m2, q2 = LGBMForecaster.load(tmp_path / "m.joblib"), LGBMQuantileForecaster.load(tmp_path / "q.joblib")
    day = pd.Timestamp("2023-12-31")
    from edf.timeutils import cutoff_for_day, target_index_for_day
    tgt, cut = target_index_for_day(day), cutoff_for_day(day)
    hist, exog = df.loc[:cut], df.reindex(tgt).drop(columns="load")
    np.testing.assert_allclose(m.predict(hist, tgt, exog), m2.predict(hist, tgt, exog))
    pd.testing.assert_frame_equal(q.predict(hist, tgt, exog), q2.predict(hist, tgt, exog))


def test_nn_save_load_roundtrip(tmp_path):
    pytest.importorskip("tensorflow")
    from edf.models.nn import NNForecaster
    df = make_df("2021-01-01", "2023-12-31 23:00", noise=300.0)
    nn = NNForecaster(hidden=(16,), epochs=3, patience=2).fit(df)
    nn.save(tmp_path / "nn")
    nn2 = NNForecaster.load(tmp_path / "nn")
    from edf.timeutils import cutoff_for_day, target_index_for_day
    day = pd.Timestamp("2023-12-31"); tgt, cut = target_index_for_day(day), cutoff_for_day(day)
    hist, exog = df.loc[:cut], df.reindex(tgt).drop(columns="load")
    np.testing.assert_allclose(nn.predict(hist, tgt, exog), nn2.predict(hist, tgt, exog), rtol=1e-4, atol=1e-2)


def test_train_final_writes_artifacts_and_meta(artifacts):
    out, meta = artifacts
    for f in ("lgbm.joblib", "quantile.joblib", "meta.json"):
        assert (out / f).exists()
    # CQR shift may be negative when the raw quantile interval is already conservative (as on this synthetic data)
    assert abs(meta["conformal_shift_mwh"]) < 2000 and meta["with_nn"] is False
    assert meta["trained_through_utc"].startswith("2024-12-31")


def test_predictor_replays_a_day_like_the_backtest(warehouse, artifacts):
    path, df = warehouse
    out, _ = artifacts
    p = serving.Predictor(out, path)
    day = pd.Timestamp("2024-12-18")
    res = p.predict_day(day)
    assert len(res) == 24
    assert (res["lower_mwh"] <= res["point_mwh"]).all() and (res["point_mwh"] <= res["upper_mwh"]).all()
    truth = df["load"].reindex(res["ts_utc"]).to_numpy()
    naive = ev.backtest(df, SeasonalNaive(168), "2024-12-18", "2024-12-18")
    assert np.abs(res["point_mwh"].to_numpy() - truth).mean() < np.abs(naive["y_pred"] - naive["y_true"]).mean() * 1.5


def test_predictor_refuses_days_without_data(warehouse, artifacts):
    path, _ = warehouse
    p = serving.Predictor(artifacts[0], path)
    with pytest.raises(serving.ForecastNotAvailable, match="needs data up to"):
        p.predict_day("2025-03-01")


def test_live_weather_mode_uses_forecast_frame(warehouse, artifacts, monkeypatch):
    path, df = warehouse
    day = pd.Timestamp("2024-12-18")
    from edf.timeutils import target_index_for_day
    tgt = target_index_for_day(day)

    def fake_frame(start, end, session=None):
        rows = [pd.DataFrame({"city": c, "ts_utc": tgt, "temperature_2m": 3.0, "shortwave_radiation": 0.0,
                              "wind_speed_10m": 4.0}) for c in CITIES]
        return pd.concat(rows, ignore_index=True)

    monkeypatch.setattr(weather, "fetch_forecast_frame", fake_frame)
    p = serving.Predictor(artifacts[0], path)
    live = p.predict_day(day, "forecast")
    arch = p.predict_day(day, "archive")
    assert len(live) == 24 and live["point_mwh"].notna().all()
    assert not np.allclose(live["point_mwh"], arch["point_mwh"])        # different weather -> different forecast


def test_weather_forecast_fetch_parses_response(monkeypatch):
    calls = []

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"], "temperature_2m": [1.0, 2.0],
                               "shortwave_radiation": [0.0, 0.0], "wind_speed_10m": [3.0, 4.0]}}

    class Sess:
        def get(self, url, params=None, timeout=None):
            calls.append(url); return Resp()

    frame = weather.fetch_forecast_frame("2025-01-01", "2025-01-01", session=Sess())
    assert set(frame["city"]) == set(CITIES) and len(frame) == 2 * len(CITIES)
    assert all("api.open-meteo.com/v1/forecast" in u for u in calls)


# ---- HTTP layer -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(warehouse, artifacts):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from edf.api import create_app
    return TestClient(create_app(serving.Predictor(artifacts[0], warehouse[0])))


def test_api_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/model").json()
    assert m["version"] == "v1" and m["conformal_alpha"] == 0.10


def test_api_forecast_ok(client):
    r = client.get("/forecast", params={"date": "2024-12-18"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hours"]) == 24 and body["interval"] == "90% conformalised"
    assert all(h["lower_mwh"] <= h["point_mwh"] <= h["upper_mwh"] for h in body["hours"])
    assert "OBSERVED weather" in body["note"]


def test_api_dst_day_has_23_hours(client):
    assert len(client.get("/forecast", params={"date": "2024-03-31"}).json()["hours"]) == 23


def test_api_errors(client):
    assert client.get("/forecast", params={"date": "2025-06-01"}).status_code == 409
    assert client.get("/forecast", params={"date": "not-a-date"}).status_code == 422
    assert client.get("/forecast", params={"date": "2024-12-18", "weather": "magic"}).status_code == 422
    assert client.get("/forecast").status_code == 422


def test_api_without_model_is_degraded_not_crashing(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from edf import api
    monkeypatch.setattr(api, "Predictor", lambda: (_ for _ in ()).throw(FileNotFoundError("no model")))
    c = TestClient(api.create_app())
    assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
    assert c.get("/forecast", params={"date": "2024-12-18"}).status_code == 503


def test_read_only_connection_does_not_create_files(tmp_path):
    import duckdb
    with pytest.raises(duckdb.IOException):
        db.connect(tmp_path / "missing.duckdb", read_only=True)


def test_several_processes_can_read_the_warehouse_at_once(warehouse):
    """Notebooks, the API and train_final only read: they must not lock each other out."""
    import subprocess, sys
    path, _ = warehouse
    code = ("import duckdb,sys; c=duckdb.connect(sys.argv[1], read_only=True); "
            "print(c.execute('select count(*) from smard_series').fetchone()[0])")
    holder = db.connect(path, read_only=True)           # this process keeps a read-only connection open
    try:
        out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=60)
    finally:
        holder.close()
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) > 1000


def test_an_open_writer_blocks_other_processes(tmp_path):
    """Documents the DuckDB behaviour behind the 'Conflicting lock' error: one open writer excludes everybody."""
    import subprocess, sys
    path = tmp_path / "w.duckdb"
    writer = db.connect(path)                            # read-write connection kept open
    try:
        code = "import duckdb,sys; duckdb.connect(sys.argv[1], read_only=True)"
        out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=60)
    finally:
        writer.close()
    assert out.returncode != 0 and "lock" in out.stderr.lower()
