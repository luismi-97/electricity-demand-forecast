import io

import numpy as np
import pandas as pd
import pytest

from edf import db, evaluation as ev, official
from edf.features import load_official
from edf.ingest import ingest_official
from edf.models.baselines import SeasonalNaive
from edf.models.gbm import LGBMQuantileForecaster
from tests.conftest import make_df

HEADER = "Datum von;Datum bis;Gesamt [MWh] Berechnete Auflösungen\n"


def _csv(rows):
    return io.StringIO(HEADER + "\n".join(rows) + "\n")


def test_official_csv_hourly_and_timezone():
    df = official.load_official_csv(_csv(["01.01.2024 00:00;01.01.2024 01:00;51.234,5",
                                          "01.01.2024 01:00;01.01.2024 02:00;50.100,0",
                                          "01.01.2024 02:00;01.01.2024 03:00;-"]))
    assert list(df.columns) == ["series", "resolution", "ts_utc", "value"]
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")     # 00:00 CET
    assert df["value"].tolist() == [51234.5, 50100.0]                           # German decimals; '-' dropped
    assert (df["series"] == "official_forecast").all()


def test_official_csv_quarterhour_is_summed_and_incomplete_hours_dropped():
    rows = [f"01.06.2024 00:{m:02d};01.06.2024 00:{m + 15:02d};100,0" if m < 45 else "01.06.2024 00:45;01.06.2024 01:00;100,0"
            for m in (0, 15, 30, 45)]
    rows += ["01.06.2024 01:00;01.06.2024 01:15;100,0", "01.06.2024 01:15;01.06.2024 01:30;100,0"]   # only 2 of 4
    df = official.load_official_csv(_csv(rows), resolution="quarterhour")
    assert len(df) == 1 and df["value"].iloc[0] == 400.0


def test_official_csv_dst_fall_back_keeps_both_hours():
    rows = []
    for h in [0, 1, 2, 2, 3]:      # 27 Oct 2024: 02:00 occurs twice
        rows.append(f"27.10.2024 {h:02d}:00;27.10.2024 {h:02d}:59;{50000 + len(rows)},0")
    df = official.load_official_csv(_csv(rows))
    assert len(df) == 5 and df["ts_utc"].is_unique


def test_ingest_and_load_official_roundtrip():
    con = db.connect(":memory:")
    buf = _csv([f"01.01.2024 {h:02d}:00;01.01.2024 {h + 1:02d}:00;{50000 + h},0" for h in range(5)])
    ingest_official(con, buf)
    s = load_official(con)
    assert len(s) == 5 and str(s.index.tz) == "UTC"
    assert load_official(db.connect(":memory:")).empty


def test_quantile_model_intervals_and_coverage():
    df = make_df(noise=500.0)
    m = LGBMQuantileForecaster(n_estimators=120, learning_rate=0.1, num_leaves=31)
    res = ev.backtest(df, m, "2024-02-01", "2024-03-15", refit_every_days=60)
    assert {"q05", "q50", "q95", "y_pred"} <= set(res.columns)
    assert (res["q05"] <= res["q50"]).all() and (res["q50"] <= res["q95"]).all()
    cov = ev.interval_metrics(res)
    assert cov["nominal"] == pytest.approx(0.90)
    assert 0.78 < cov["coverage"] < 0.97
    pb = ev.pinball_loss(res)
    assert set(pb) == {"q05", "q50", "q95"} and all(v > 0 for v in pb.values())


def test_pinball_zero_for_perfect_forecast():
    res = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "q50": [1.0, 2.0, 3.0]})
    assert ev.pinball_loss(res)["q50"] == 0.0


def test_rolling_conformal_repairs_too_narrow_intervals():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz="UTC")
    y = 50000 + rng.normal(0, 500, len(idx))
    res = pd.DataFrame({"ts_utc": idx, "y_true": y, "q05": 50000 - 100.0, "q50": 50000.0, "q95": 50000 + 100.0})
    before = ev.interval_metrics(res)["coverage"]
    cal = ev.rolling_conformal(res, window_days=60, lag_days=2, alpha=0.10)
    after = ev.interval_metrics(cal, suffix="_c")
    assert before < 0.3
    assert 0.85 < after["coverage"] < 0.95
    assert cal[["q05_c", "q95_c"]].iloc[: 24 * 10].isna().all().all()       # no calibration history yet


def test_conformal_uses_only_past_days():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=24 * 120, freq="h", tz="UTC")
    res = pd.DataFrame({"ts_utc": idx, "y_true": 50000 + rng.normal(0, 500, len(idx)),
                        "q05": 49900.0, "q50": 50000.0, "q95": 50100.0})
    a = ev.rolling_conformal(res)
    res2 = res.copy()
    last_day = res2["ts_utc"].dt.tz_convert("Europe/Berlin").dt.normalize().max()
    res2.loc[res2["ts_utc"].dt.tz_convert("Europe/Berlin").dt.normalize() >= last_day - pd.Timedelta(days=1), "y_true"] = 1e9
    b = ev.rolling_conformal(res2)
    day = res["ts_utc"].dt.tz_convert("Europe/Berlin").dt.normalize()
    # the corrupted outcomes (last two days) must not influence the interval of the last day (lag_days=2)
    np.testing.assert_allclose(a.loc[day == last_day, "q05_c"].to_numpy(), b.loc[day == last_day, "q05_c"].to_numpy())


def test_block_bootstrap_detects_better_forecast_and_null():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24 * 120, freq="h", tz="UTC")
    y = 50000 + rng.normal(0, 300, len(idx))
    good = pd.DataFrame({"ts_utc": idx, "y_true": y, "y_pred": y + rng.normal(0, 200, len(idx))})
    bad = pd.DataFrame({"ts_utc": idx, "y_true": y, "y_pred": y + rng.normal(0, 800, len(idx))})
    r = ev.paired_block_bootstrap(good, bad, n_boot=300)
    assert r["diff_a_minus_b"] < 0 and r["ci95_high"] < 0 and r["share_days_a_better"] > 0.9
    same = ev.paired_block_bootstrap(good, good, n_boot=300)
    assert same["diff_a_minus_b"] == 0 and same["ci95_low"] == 0 and same["ci95_high"] == 0


def test_series_to_result_covers_all_local_hours():
    df = make_df()
    pred = df["load"] * 1.01
    res = ev.series_to_result(df["load"], pred, "2024-03-30", "2024-04-01")      # contains the 23-hour day
    assert len(res) == 24 + 23 + 24
    assert np.allclose(res["y_pred"], res["y_true"] * 1.01)


def test_neural_network_forecaster():
    pytest.importorskip("tensorflow")
    from edf.models.nn import NNForecaster
    df = make_df(noise=500.0)
    nn = NNForecaster(hidden=(32, 32), epochs=15, patience=3, n_models=1)
    res = {"naive": ev.backtest(df, SeasonalNaive(168), "2024-02-01", "2024-02-14"),
           "nn": ev.backtest(df, nn, "2024-02-01", "2024-02-14", refit_every_days=60)}
    assert res["nn"]["y_pred"].notna().all()
    tab = ev.summarise(res)
    assert tab.loc["nn", "MAE"] < tab.loc["naive", "MAE"]
