import numpy as np
import pandas as pd
import pytest

from edf import evaluation as ev
from edf.features import ALL_GROUPS, FEATURE_GROUPS, holiday_share, make_features, solar_elevation
from edf.models.baselines import SeasonalNaive
from edf.models.gbm import LGBMForecaster
from edf.timeutils import cutoff_for_day, target_index_for_day
from tests.conftest import make_df


def test_holiday_share_national_regional_and_none():
    d = pd.DatetimeIndex(["2024-01-01", "2024-12-25", "2024-10-31", "2024-06-12"])
    hs = holiday_share(d)
    assert hs[0] == pytest.approx(1.0) and hs[1] == pytest.approx(1.0)
    assert 0.1 < hs[2] < 0.6          # Reformation Day: only some states
    assert hs[3] == 0.0


def test_solar_elevation_sanity():
    e = solar_elevation(pd.DatetimeIndex(["2024-06-21 10:30", "2024-12-21 10:30", "2024-06-21 22:30"], tz="UTC"))
    assert 60 < e[0] < 64 and 12 < e[1] < 19 and e[2] < 0


def test_bridge_day_flag():
    df = make_df(noise=0.0)
    day = pd.Timestamp("2024-05-10")          # Friday after Ascension Day (Thursday 9 May)
    tgt = target_index_for_day(day)
    X = make_features(df, tgt)
    assert (X["bridge"] == 1).all()
    assert (X["hol_share"] == 0).all() and np.allclose(X["hol_prev"], 1.0)


def test_features_do_not_use_data_after_the_cutoff():
    df = make_df(noise=300.0)
    day = pd.Timestamp("2024-03-20")
    tgt = target_index_for_day(day)
    cutoff = cutoff_for_day(day)
    X1 = make_features(df, tgt)
    broken = df.copy()
    broken.loc[broken.index > cutoff, "load"] = 1e9      # corrupt EVERYTHING after the cutoff
    X2 = make_features(broken, tgt)
    pd.testing.assert_frame_equal(X1, X2)
    assert (X1["horizon"].between(13, 36)).all()


def test_train_and_predict_features_agree():
    """The same row must get identical features when built from full history vs. from the live context."""
    df = make_df(noise=300.0)
    day = pd.Timestamp("2024-03-20")
    tgt = target_index_for_day(day)
    cutoff = cutoff_for_day(day)
    hist = df.loc[:cutoff]
    ctx = pd.concat([hist.tail(24 * 30), df.reindex(tgt).drop(columns="load").assign(load=np.nan)])
    pd.testing.assert_frame_equal(make_features(df, tgt), make_features(ctx, tgt))


def test_lgbm_beats_naive_and_has_small_bias():
    df = make_df(noise=500.0)
    kw = dict(n_estimators=150, learning_rate=0.1, num_leaves=31)
    res = {
        "naive": ev.backtest(df, SeasonalNaive(168), "2024-02-01", "2024-02-29"),
        "gbm": ev.backtest(df, LGBMForecaster(**kw), "2024-02-01", "2024-02-29", refit_every_days=60),
        "gbm_ratio": ev.backtest(df, LGBMForecaster(target="ratio", **kw), "2024-02-01", "2024-02-29", refit_every_days=60),
    }
    tab = ev.summarise(res)
    assert tab.loc["gbm", "MAE"] < tab.loc["naive", "MAE"]
    assert tab.loc["gbm_ratio", "MAE"] < tab.loc["naive", "MAE"]
    assert abs(tab.loc["gbm", "bias"]) < 0.02 * 55000


def test_exog_noise_hurts_a_weather_driven_model():
    df = make_df(noise=100.0)
    m = LGBMForecaster(groups=("calendar", "weather"), n_estimators=120, learning_rate=0.1, num_leaves=31)
    clean = ev.backtest(df, m, "2024-02-01", "2024-02-20", refit_every_days=60)
    noisy = ev.backtest(df, m, "2024-02-01", "2024-02-20", refit_every_days=60, exog_noise={"temp": 4.0})
    assert ev.metrics(noisy)["MAE"] > ev.metrics(clean)["MAE"]


def test_ratio_target_requires_lags():
    with pytest.raises(ValueError):
        LGBMForecaster(groups=("calendar",), target="ratio")
