"""Forecaster interface used by the backtest."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    name: str = "forecaster"

    def fit(self, hist: pd.DataFrame) -> "Forecaster":
        """Learn from `hist` (hourly UTC frame, columns `load` [+ exogenous]). Stateless models may skip."""
        return self

    @abstractmethod
    def predict(self, hist: pd.DataFrame, target_index: pd.DatetimeIndex, exog: pd.DataFrame) -> pd.Series:
        """Forecast `target_index` using ONLY `hist` for the load history.

        `hist` is cut at the forecast cutoff by the backtest; `exog` holds the exogenous columns of the
        target hours (e.g. temperature). With observed weather this is an optimistic 'oracle weather'
        setting, to be documented wherever results are reported.
        """
