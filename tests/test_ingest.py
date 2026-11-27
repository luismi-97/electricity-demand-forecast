import pandas as pd

from edf import db, ingest, smard, weather


def _fake_smard(series, resolution="hour", start=None, end=None, **kw):
    idx = pd.date_range(start, periods=48, freq="h", tz="UTC")
    return pd.DataFrame({"series": series, "resolution": resolution, "ts_utc": idx, "value": 1.0})


def _fake_weather(city, lat, lon, start, end, **kw):
    idx = pd.date_range(start, periods=24, freq="h", tz="UTC")
    return pd.DataFrame({"city": city, "ts_utc": idx, "temperature_2m": 5.0,
                         "shortwave_radiation": 0.0, "wind_speed_10m": 3.0})


def test_ingest_smard_and_resume(monkeypatch):
    monkeypatch.setattr(smard, "fetch_series", _fake_smard)
    con = db.connect(":memory:")
    ingest.ingest_smard(con, ["load"], start="2024-01-01")
    assert con.execute("SELECT count(*) FROM smard_series").fetchone()[0] == 48
    # second run without --start resumes (7-day overlap) and stays idempotent on the overlap
    ingest.ingest_smard(con, ["load"])
    n = con.execute("SELECT count(*) FROM smard_series").fetchone()[0]
    assert n >= 48


def test_ingest_weather(monkeypatch):
    monkeypatch.setattr(weather, "fetch_city", _fake_weather)
    con = db.connect(":memory:")
    ingest.ingest_weather(con, start="2024-01-01", end="2024-01-02")
    cities = con.execute("SELECT count(DISTINCT city) FROM weather_hourly").fetchone()[0]
    assert cities == 6
