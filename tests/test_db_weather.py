import pandas as pd

from edf import db, weather


def test_upsert_is_idempotent():
    con = db.connect(":memory:")
    df = pd.DataFrame({
        "series": ["load"] * 3, "resolution": ["hour"] * 3,
        "ts_utc": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "value": [1.0, 2.0, 3.0],
    })
    db.upsert(con, "smard_series", df)
    db.upsert(con, "smard_series", df)                       # same rows again
    df2 = df.assign(value=[10.0, 20.0, 30.0])
    db.upsert(con, "smard_series", df2)                      # corrected values
    n, total = con.execute("SELECT count(*), sum(value) FROM smard_series").fetchone()
    assert n == 3 and total == 60.0


def test_weather_parse_and_store():
    payload = {"hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
        "temperature_2m": [1.5, 1.2],
        "shortwave_radiation": [0.0, 0.0],
        "wind_speed_10m": [10.0, 11.0],
    }}
    df = weather.parse_response(payload, "berlin")
    assert str(df["ts_utc"].dt.tz) == "UTC"
    con = db.connect(":memory:")
    assert db.upsert(con, "weather_hourly", df) == 2
    assert con.execute("SELECT count(*) FROM weather_hourly").fetchone()[0] == 2
