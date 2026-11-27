import numpy as np
import pandas as pd
import pytest

from edf import evaluation as ev
from edf.models.base import Forecaster
from edf.models.baselines import CalendarRidge, SeasonalMean, SeasonalNaive
from tests.conftest import make_df


def test_target_index_handles_dst():
    assert len(ev.target_index_for_day(pd.Timestamp("2024-03-31"))) == 23   # spring forward
    assert len(ev.target_index_for_day(pd.Timestamp("2024-10-27"))) == 25   # fall back
    assert len(ev.target_index_for_day(pd.Timestamp("2024-06-15"))) == 24


def test_cutoff_is_before_origin_and_before_targets():
    day = pd.Timestamp("2024-06-15")
    cutoff = ev.cutoff_for_day(day, origin_hour=12)
    # 12:00 CEST on 14 June = 10:00 UTC; last complete hour starts at 09:00 UTC
    assert cutoff == pd.Timestamp("2024-06-14 09:00", tz="UTC")
    tgt = ev.target_index_for_day(day)
    assert cutoff < tgt.min()
    assert int(((tgt - cutoff) / pd.Timedelta(hours=1)).max()) <= 37


def test_no_future_data_reaches_the_model():
    df = make_df(noise=100.0)

    class Spy(Forecaster):
        name = "spy"
        def predict(self, hist, target_index, exog):
            assert hist.index.max() < target_index.min(), "history overlaps the target period"
            assert hist.index.max() == ev.cutoff_for_day(target_index.tz_convert("Europe/Berlin")[0].tz_localize(None).normalize())
            return pd.Series(0.0, index=target_index)

    ev.backtest(df, Spy(), "2023-01-01", "2023-01-20")


def test_lag_not_available_raises():
    df = make_df()
    with pytest.raises(ValueError, match="not available"):
        ev.backtest(df, SeasonalNaive(lag_hours=24), "2023-01-01", "2023-01-05")   # lag24 is unknown for afternoon hours


def test_final_test_is_locked():
    df = make_df()
    with pytest.raises(ValueError, match="locked"):
        ev.backtest(df, SeasonalNaive(168), "2024-12-01", "2025-01-10")


def test_naive_is_exact_on_weekly_periodic_series():
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    s = pd.Series(np.tile(np.arange(168, dtype=float), len(idx) // 168 + 1)[: len(idx)], index=idx)
    df = pd.DataFrame({"load": s})
    res = ev.backtest(df, SeasonalNaive(168), "2023-03-01", "2023-03-31")
    assert np.allclose(res["y_pred"], res["y_true"])


def test_baselines_ranking_on_structured_data():
    df = make_df(noise=500.0)
    out = {}
    for m in (SeasonalNaive(168), SeasonalMean(4), CalendarRidge()):
        out[m.name] = ev.backtest(df, m, "2024-01-01", "2024-06-30")
    tab = ev.summarise(out, scale=ev.mase_scale(df["load"].loc[:"2023-12-31"]))
    # a model that knows the structure should beat the noisy naive; the 4-week mean should beat single-lag naive
    assert tab.index[0] == "calendar_ridge"
    assert tab.loc["mean_4w", "MAE"] < tab.loc["naive_168h", "MAE"]
    assert {"MAE", "RMSE", "MAPE_%", "MASE", "bias"} <= set(tab.columns)


def test_summarise_uses_common_timestamps():
    df = make_df(noise=100.0)
    a = ev.backtest(df, SeasonalNaive(168), "2024-01-01", "2024-01-10")
    b = a.copy(); b.loc[:10, "y_pred"] = np.nan
    tab = ev.summarise({"a": a, "b": b})
    assert tab.loc["a", "n"] == tab.loc["b", "n"]


def test_breakdown_by_hour():
    df = make_df(noise=100.0)
    res = ev.backtest(df, SeasonalNaive(168), "2024-01-01", "2024-01-31")
    b = ev.breakdown(res, "hour")
    assert len(b) == 24
