import numpy as np
import pandas as pd

from edf import smard

H = 3_600_000  # one hour in ms
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)


def fake_get_json_factory():
    calls = []

    def fake(url, session=None, retries=3, pause=1.0):
        calls.append(url)
        if "index_hour" in url:
            return {"timestamps": [T0, T0 + 168 * H, T0 + 2 * 168 * H]}
        ts = int(url.rsplit("_", 1)[1].split(".")[0])
        rows = [[ts + i * H, 50_000.0 + i] for i in range(168)]
        rows[3][1] = None                      # SMARD may publish gaps as null
        return {"series": rows}
    return fake, calls


def test_fetch_series_basic(monkeypatch):
    fake, calls = fake_get_json_factory()
    monkeypatch.setattr(smard, "_get_json", fake)
    monkeypatch.setattr(smard.time, "sleep", lambda s: None)
    df = smard.fetch_series("load", start="2024-01-01", end="2024-01-14")
    assert list(df.columns) == ["series", "resolution", "ts_utc", "value"]
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert df["ts_utc"].is_monotonic_increasing and df["ts_utc"].is_unique
    assert df["value"].notna().all()
    assert len(df) == 2 * 168 - 2               # two chunks, one null dropped in each


def test_fetch_series_respects_window(monkeypatch):
    fake, calls = fake_get_json_factory()
    monkeypatch.setattr(smard, "_get_json", fake)
    monkeypatch.setattr(smard.time, "sleep", lambda s: None)
    df = smard.fetch_series("load", start="2024-01-09", end="2024-01-10")
    assert df["ts_utc"].min() >= pd.Timestamp("2024-01-09", tz="UTC")
    assert df["ts_utc"].max() <= pd.Timestamp("2024-01-10 23:59:59", tz="UTC")


def test_bad_resolution():
    import pytest
    with pytest.raises(ValueError):
        smard.fetch_series("load", resolution="minute")
