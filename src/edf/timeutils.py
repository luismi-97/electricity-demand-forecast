"""Local-time helpers shared by evaluation and feature engineering."""
from __future__ import annotations

import pandas as pd

from edf.config import ORIGIN_HOUR, TZ


def target_index_for_day(day: pd.Timestamp, tz: str = TZ) -> pd.DatetimeIndex:
    """All hours of local calendar day `day` as a UTC index (23 or 25 hours on DST changes)."""
    d0 = pd.Timestamp(day.date()).tz_localize(tz)
    d1 = pd.Timestamp((day + pd.Timedelta(days=1)).date()).tz_localize(tz)
    return pd.date_range(d0, d1, freq="h", inclusive="left").tz_convert("UTC")


def cutoff_for_day(day: pd.Timestamp, origin_hour: int = ORIGIN_HOUR, tz: str = TZ) -> pd.Timestamp:
    """Timestamp (UTC) of the last load value known when the forecast for `day` is issued."""
    prev = pd.Timestamp((day - pd.Timedelta(days=1)).date())
    origin = (prev + pd.Timedelta(hours=origin_hour)).tz_localize(tz)
    return origin.tz_convert("UTC") - pd.Timedelta(hours=1)
