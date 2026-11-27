"""Baselines for day-ahead load forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from edf.features import calendar_frame
from edf.models.base import Forecaster


class SeasonalNaive(Forecaster):
    """Value of the same hour one (or more) week(s) ago. lag_hours must be available at forecast time."""

    def __init__(self, lag_hours: int = 168):
        self.lag_hours = lag_hours
        self.name = f"naive_{lag_hours}h"

    def predict(self, hist, target_index, exog):
        src = target_index - pd.Timedelta(hours=self.lag_hours)
        if src.max() > hist.index.max():
            raise ValueError(
                f"{self.name}: lag {self.lag_hours}h is not available at forecast time "
                f"(needs data up to {src.max()}, history ends {hist.index.max()})"
            )
        return pd.Series(hist["load"].reindex(src).to_numpy(), index=target_index)


class SeasonalMean(Forecaster):
    """Average of the same hour-of-week over the last `weeks` weeks (robust to one-off anomalies)."""

    def __init__(self, weeks: int = 4):
        self.weeks = weeks
        self.name = f"mean_{weeks}w"

    def predict(self, hist, target_index, exog):
        vals = []
        for k in range(1, self.weeks + 1):
            src = target_index - pd.Timedelta(hours=168 * k)
            if src.max() > hist.index.max():
                raise ValueError(f"{self.name}: lag {168 * k}h not available at forecast time")
            vals.append(hist["load"].reindex(src).to_numpy())
        return pd.Series(np.nanmean(np.vstack(vals), axis=0), index=target_index)


class CalendarRidge(Forecaster):
    """Statistical benchmark: ridge regression on hour-of-week (holidays as Sundays), weather and season.

    Uses the observed temperature of the target hour (oracle weather) -> optimistic, see README.
    """

    def __init__(self, alpha: float = 1.0, train_window_days: int | None = 730):
        self.alpha = alpha
        self.train_window_days = train_window_days
        self.name = "calendar_ridge"
        self._model: Ridge | None = None

    @staticmethod
    def _design(index: pd.DatetimeIndex, temp: np.ndarray) -> np.ndarray:
        cal = calendar_frame(index)
        how = (cal["eff_dow"] * 24 + cal["hour"]).to_numpy()
        onehot = np.zeros((len(index), 168))
        onehot[np.arange(len(index)), how] = 1.0
        t = temp / 10.0
        hdd = np.clip(15 - temp, 0, None) / 10.0
        cdd = np.clip(temp - 22, 0, None) / 10.0
        w = 2 * np.pi * cal["doy"].to_numpy() / 365.25
        season = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
        return np.column_stack([onehot, t, t**2, hdd, cdd, season])

    def fit(self, hist):
        if "temp" not in hist:
            raise ValueError("CalendarRidge needs a `temp` column")
        h = hist
        if self.train_window_days:
            h = h.loc[h.index.max() - pd.Timedelta(days=self.train_window_days):]
        h = h.dropna(subset=["load", "temp"])
        X = self._design(h.index, h["temp"].to_numpy())
        self._model = Ridge(alpha=self.alpha).fit(X, h["load"].to_numpy())
        return self

    def predict(self, hist, target_index, exog):
        if self._model is None:
            raise RuntimeError("fit() first")
        temp = exog["temp"].to_numpy()
        ok = ~np.isnan(temp)
        out = np.full(len(target_index), np.nan)
        if ok.any():
            out[ok] = self._model.predict(self._design(target_index[ok], temp[ok]))
        return pd.Series(out, index=target_index)
