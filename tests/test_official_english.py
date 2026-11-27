import io

import numpy as np
import pandas as pd

from edf import db, official
from edf.ingest import ingest_official

SAMPLE = """Start date;End date;grid load [MWh] Calculated resolutions;Residual load [MWh] Calculated resolutions
Jan 1, 2023 12:00 AM;Jan 1, 2023 1:00 AM;41,792.50;2,798.75
Jan 1, 2023 1:00 AM;Jan 1, 2023 2:00 AM;39,621.00;886.25
Jan 1, 2023 2:00 AM;Jan 1, 2023 3:00 AM;38,240.75;-293.50
Jan 1, 2023 3:00 AM;Jan 1, 2023 4:00 AM;37,205.50;-645.75
Jan 1, 2023 4:00 AM;Jan 1, 2023 5:00 AM;37,326.75;-2.50
"""


def test_english_export_uses_grid_load_not_residual():
    df = official.load_official_csv(io.StringIO(SAMPLE))
    assert df["value"].tolist() == [41792.5, 39621.0, 38240.75, 37205.5, 37326.75]
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2022-12-31 23:00", tz="UTC")       # 12:00 AM CET
    assert (df["series"] == "official_forecast").all()


def test_english_export_value_col_override_and_negative_numbers():
    df = official.load_official_csv(io.StringIO(SAMPLE), value_col="Residual load [MWh] Calculated resolutions")
    assert df["value"].tolist() == [2798.75, 886.25, -293.5, -645.75, -2.5]


def test_english_export_dst_fall_back_and_pm_hours():
    rows = ["Start date;End date;grid load [MWh] Calculated resolutions"]
    for hh in ["12:00 AM", "1:00 AM", "2:00 AM", "2:00 AM", "3:00 AM", "1:00 PM"]:     # 29 Oct 2023: 02:00 twice
        rows.append(f"Oct 29, 2023 {hh};Oct 29, 2023 {hh};50,000.00")
    df = official.load_official_csv(io.StringIO("\n".join(rows) + "\n"))
    assert len(df) == 6 and df["ts_utc"].is_unique


def test_locale_detection():
    assert official._detect_decimal(pd.Series(["41,792.50", "1,234.00"])) == "."
    assert official._detect_decimal(pd.Series(["41.792,50", "1.234,00"])) == ","
    assert official._detect_decimal(pd.Series(["12.5", "3.25"])) == "."
    assert official._detect_decimal(pd.Series(["12,5", "3,25"])) == ","


def _daily_load(n_days=60, seed=0):
    idx = pd.date_range("2023-01-01", periods=24 * n_days, freq="h", tz="UTC")
    hour = np.asarray(idx.tz_convert("Europe/Berlin").hour)
    rng = np.random.default_rng(seed)
    return pd.Series(50000 + 8000 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 300, len(idx)), index=idx)


def test_alignment_check_detects_correct_and_shifted_timestamps():
    load = _daily_load()
    ok = official.alignment_check(load + 100, load)
    assert min(ok, key=ok.get) == 0
    shifted = load.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)           # forecast stamped one hour late
    bad = official.alignment_check(shifted, load)
    assert min(bad, key=bad.get) == -1          # moving the timestamps back by one hour realigns it
    assert official.alignment_check(load.iloc[:48], load.iloc[:48]) == {}


def test_ingest_logs_alignment(caplog):
    con = db.connect(":memory:")
    load = _daily_load()
    db.upsert(con, "smard_series", pd.DataFrame({"series": "load", "resolution": "hour",
                                                 "ts_utc": load.index, "value": load.to_numpy()}))
    loc = load.index.tz_convert("Europe/Berlin").tz_localize(None)
    lines = ["Start date;End date;grid load [MWh] Calculated resolutions"]
    for t, v in zip(loc, load.to_numpy() + 50):
        s = t.strftime("%b %-d, %Y %-I:%M %p")
        lines.append(f"{s};{s};{v:,.2f}")
    import logging
    with caplog.at_level(logging.INFO, logger="edf.ingest"):
        ingest_official(con, io.StringIO("\n".join(lines) + "\n"))
    assert "Alignment OK" in caplog.text