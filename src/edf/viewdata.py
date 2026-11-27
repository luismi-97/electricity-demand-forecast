"""Data preparation for the dashboard (kept free of Streamlit so it can be unit-tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from edf.timeutils import cutoff_for_day

MIN_HISTORY_DAYS = 400      # the models need about a year of history before a day can be replayed


def selectable_days(df: pd.DataFrame, origin_hour: int = 12) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last local day that can be replayed: enough history, load known up to the cutoff, weather present."""
    load = df["load"].dropna()
    first = (load.index.min().tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=MIN_HISTORY_DAYS)).tz_localize(None)
    last = None
    if "temp" in df and df["temp"].notna().any():
        last_weather = df["temp"].dropna().index.max().tz_convert("Europe/Berlin").normalize().tz_localize(None) - pd.Timedelta(days=1)
        last_load = load.index.max().tz_convert("Europe/Berlin").normalize().tz_localize(None) - pd.Timedelta(days=1)
        last = min(last_weather, last_load)
        while last > first and cutoff_for_day(last, origin_hour) > load.index.max():
            last -= pd.Timedelta(days=1)
    if last is None or last < first:
        raise ValueError("Not enough data to replay any day (need ~400 days of history, load and weather)")
    return first, last


def day_view(forecast: pd.DataFrame, load: pd.Series, official: pd.Series | None = None) -> pd.DataFrame:
    """Join the forecast of one day with the actual load, the official forecast and the weekly naive."""
    out = forecast.copy()
    idx = pd.DatetimeIndex(out["ts_utc"])
    out["actual_mwh"] = load.reindex(idx).to_numpy()
    out["naive_mwh"] = load.reindex(idx - pd.Timedelta(hours=168)).to_numpy()
    out["official_mwh"] = official.reindex(idx).to_numpy() if official is not None and len(official) else np.nan
    return out


def day_metrics(view: pd.DataFrame) -> dict:
    """MAE (MWh) of each forecast on one day, plus how often the actual load fell inside the interval."""
    m = {}
    for name, col in (("blend", "point_mwh"), ("official", "official_mwh"), ("naive", "naive_mwh")):
        ok = view[["actual_mwh", col]].dropna()
        m[name] = float((ok[col] - ok["actual_mwh"]).abs().mean()) if len(ok) else float("nan")
    ok = view.dropna(subset=["actual_mwh", "lower_mwh", "upper_mwh"])
    m["coverage"] = (float(((ok["actual_mwh"] >= ok["lower_mwh"]) & (ok["actual_mwh"] <= ok["upper_mwh"])).mean())
                     if len(ok) else float("nan"))
    return m
