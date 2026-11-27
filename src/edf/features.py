"""Calendar and weather features (extended in week 3)."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from edf.config import CITIES, ORIGIN_HOUR, TZ
from edf.timeutils import cutoff_for_day


def to_local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TZ)


def is_holiday(local_index: pd.DatetimeIndex) -> np.ndarray:
    """National German public holidays only. State-specific holidays are a week-3 refinement."""
    years = range(local_index.year.min(), local_index.year.max() + 1)
    cal = holidays.country_holidays("DE", years=years)
    dates = local_index.normalize().tz_localize(None)
    return np.array([d.date() in cal for d in dates])


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, holiday flag in local time. `eff_dow` treats holidays as Sundays."""
    loc = to_local(index)
    hol = is_holiday(loc)
    dow = np.asarray(loc.dayofweek)
    return pd.DataFrame({
        "hour": np.asarray(loc.hour),
        "dow": dow,
        "eff_dow": np.where(hol, 6, dow),
        "month": np.asarray(loc.month),
        "doy": np.asarray(loc.dayofyear),
        "holiday": hol,
    }, index=index)


WEATHER_COLUMNS = {"temperature_2m": "temp", "shortwave_radiation": "rad", "wind_speed_10m": "wind"}


def weight_weather_frame(w: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted mean over cities. `w` has columns city, ts_utc, temperature_2m, shortwave_radiation,
    wind_speed_10m; returns an hourly UTC frame with temp, rad, wind."""
    if w.empty:
        return pd.DataFrame()
    w = w.copy()
    w["ts_utc"] = pd.DatetimeIndex(w["ts_utc"]).tz_convert("UTC")
    w["wt"] = w["city"].map({c: v[2] for c, v in CITIES.items()})
    out = {}
    for src, dst in WEATHER_COLUMNS.items():
        ok = w[src].notna()
        num = (w.loc[ok, src] * w.loc[ok, "wt"]).groupby(w.loc[ok, "ts_utc"]).sum()
        den = w.loc[ok, "wt"].groupby(w.loc[ok, "ts_utc"]).sum()
        out[dst] = num / den
    return pd.DataFrame(out)


def weighted_weather(con) -> pd.DataFrame:
    """Population-weighted weather across the configured cities (observed ERA5 values)."""
    w = con.execute(
        "SELECT city, ts_utc, temperature_2m, shortwave_radiation, wind_speed_10m FROM weather_hourly"
    ).df()
    return weight_weather_frame(w)


def weighted_temperature(con) -> pd.Series:
    wx = weighted_weather(con)
    return wx["temp"] if "temp" in wx else pd.Series(dtype=float, name="temp")


def load_dataset(con) -> pd.DataFrame:
    """Hourly UTC frame with `load` and, if available, `temp`, `rad`, `wind`. Gaps stay NaN."""
    s = con.execute(
        "SELECT ts_utc, value FROM smard_series WHERE series='load' AND resolution='hour' ORDER BY ts_utc"
    ).df()
    if s.empty:
        raise SystemExit("No load data yet: run `python -m edf.ingest smard` first.")
    load = s.set_index(pd.DatetimeIndex(s["ts_utc"]).tz_convert("UTC"))["value"].rename("load")
    full = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC")
    df = load.reindex(full).to_frame()
    wx = weighted_weather(con)
    for col in wx.columns:
        df[col] = wx[col].reindex(full)
    return df


# ---- Week 3: richer features ------------------------------------------------

# Approximate state populations (millions); used to weight state-specific public holidays.
STATE_POP = {"BW": 11.3, "BY": 13.4, "BE": 3.7, "BB": 2.6, "HB": 0.68, "HH": 1.9, "HE": 6.4, "MV": 1.6,
             "NI": 8.1, "NW": 18.1, "RP": 4.2, "SL": 1.0, "SN": 4.1, "ST": 2.2, "SH": 2.9, "TH": 2.1}


@lru_cache(maxsize=None)
def _holiday_share_year(year: int) -> dict:
    total = sum(STATE_POP.values())
    share: dict = defaultdict(float)
    for state, pop in STATE_POP.items():
        for d in holidays.country_holidays("DE", subdiv=state, years=year):
            share[d] += pop / total
    return dict(share)


def holiday_share(local_dates: pd.DatetimeIndex) -> np.ndarray:
    """Share (0-1) of the population living in a state where the date is a public holiday."""
    return np.array([_holiday_share_year(d.year).get(d.date(), 0.0) for d in local_dates])


def solar_elevation(index_utc: pd.DatetimeIndex, lat: float = 51.0, lon: float = 10.0) -> np.ndarray:
    """Approximate solar elevation (degrees) at the centre of Germany for the middle of each hour.
    Simplified declination formula, equation of time ignored (error of a few minutes)."""
    t = index_utc + pd.Timedelta(minutes=30)
    doy = np.asarray(t.dayofyear)
    hour = np.asarray(t.hour + t.minute / 60.0)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    solar_time = hour + lon / 15.0
    ha = np.radians(15.0 * (solar_time - 12.0))
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)))


FEATURE_GROUPS = {
    "calendar": ["hour", "dow", "month", "doy", "hol_share", "hol_prev", "hol_next", "bridge", "xmas"],
    "solar": ["sun_elev"],
    "weather": ["temp", "temp_day_mean", "temp_roll72", "hdd", "cdd", "rad", "wind"],
    "lags": ["horizon", "lag_last_day", "lag_168", "lag_336", "level_168", "level_24"],
}
ALL_GROUPS = tuple(FEATURE_GROUPS)


def make_features(ctx: pd.DataFrame, target_index: pd.DatetimeIndex, origin_hour: int = ORIGIN_HOUR) -> pd.DataFrame:
    """Feature table for `target_index`, built exactly as it would be known at the day-ahead origin.

    `ctx` is an hourly UTC frame with `load` (+ `temp`, `rad`, `wind`). Load-derived features only use values at
    or before the cutoff of each target's day (lags >= 168 h, the same hour on the latest available day, and
    rolling levels ending AT the cutoff), so the same function serves training rows and live forecasts.
    Weather columns of the target hours are the exogenous inputs (observed ERA5 here: 'oracle weather').
    """
    loc = target_index.tz_convert(TZ)
    days = loc.normalize().tz_localize(None)
    cut_map = {d: cutoff_for_day(pd.Timestamp(d), origin_hour) for d in days.unique()}
    cutoff = pd.DatetimeIndex([cut_map[d] for d in days])
    h = np.asarray((target_index - cutoff) / pd.Timedelta(hours=1))

    X = pd.DataFrame(index=target_index)
    # calendar
    hs = holiday_share(days)
    hp = holiday_share(days - pd.Timedelta(days=1))
    hn = holiday_share(days + pd.Timedelta(days=1))
    dow = np.asarray(days.dayofweek)
    X["hour"] = np.asarray(loc.hour)
    X["dow"] = dow
    X["month"] = np.asarray(loc.month)
    X["doy"] = np.asarray(loc.dayofyear)
    X["hol_share"], X["hol_prev"], X["hol_next"] = hs, hp, hn
    X["bridge"] = (((dow == 4) & (hp >= 0.5) & (hs < 0.5)) | ((dow == 0) & (hn >= 0.5) & (hs < 0.5))).astype(int)
    X["xmas"] = (((np.asarray(loc.month) == 12) & (np.asarray(loc.day) >= 24))
                 | ((np.asarray(loc.month) == 1) & (np.asarray(loc.day) == 1))).astype(int)
    # solar
    X["sun_elev"] = solar_elevation(target_index)
    # weather (exogenous values of the target hours)
    ctx_days = ctx.index.tz_convert(TZ).normalize().tz_localize(None)
    for col in ("temp", "rad", "wind"):
        X[col] = ctx[col].reindex(target_index).to_numpy() if col in ctx else np.nan
    if "temp" in ctx:
        X["temp_day_mean"] = ctx["temp"].groupby(ctx_days).mean().reindex(days).to_numpy()
        X["temp_roll72"] = ctx["temp"].rolling(72, min_periods=48).mean().reindex(cutoff).to_numpy()
    else:
        X["temp_day_mean"] = X["temp_roll72"] = np.nan
    X["hdd"] = np.clip(15 - X["temp"], 0, None)
    X["cdd"] = np.clip(X["temp"] - 22, 0, None)
    # load-derived (all strictly <= cutoff)
    L = ctx["load"]
    k = np.ceil(h / 24.0)
    X["horizon"] = h
    X["lag_last_day"] = L.reindex(target_index - pd.to_timedelta(24 * k, unit="h")).to_numpy()
    X["lag_168"] = L.reindex(target_index - pd.Timedelta(hours=168)).to_numpy()
    X["lag_336"] = L.reindex(target_index - pd.Timedelta(hours=336)).to_numpy()
    X["level_168"] = L.rolling(168, min_periods=120).mean().reindex(cutoff).to_numpy()
    X["level_24"] = L.rolling(24, min_periods=18).mean().reindex(cutoff).to_numpy()
    return X


def load_official(con) -> pd.Series:
    """Official TSO day-ahead load forecast (hourly, UTC) or an empty series if it was not imported."""
    s = con.execute(
        "SELECT ts_utc, value FROM smard_series WHERE series='official_forecast' AND resolution='hour' ORDER BY ts_utc"
    ).df()
    if s.empty:
        return pd.Series(dtype=float, name="official")
    return s.set_index(pd.DatetimeIndex(s["ts_utc"]).tz_convert("UTC"))["value"].rename("official")
