"""Prediction service logic (framework-independent, used by the FastAPI app).

For a target day D the service reproduces exactly what the backtest does: it uses the load known at the 12:00 origin
of D-1, the weather of D (observed archive data for replays of past days, or a weather FORECAST for live use) and
the frozen models. Output: hourly point forecast (blend), 90 % interval (conformalised quantiles) and the components.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edf import db, weather
from edf.config import DB_PATH, MODEL_DIR, TZ
from edf.features import load_dataset, weight_weather_frame
from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster
from edf.timeutils import cutoff_for_day, target_index_for_day

WEATHER_COLS = ["temp", "rad", "wind"]


class ForecastNotAvailable(Exception):
    """The requested day cannot be forecast with the data currently available."""


class Predictor:
    def __init__(self, model_dir=MODEL_DIR, db_path=DB_PATH, cache_seconds: float = 60.0):
        d = Path(model_dir)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No model artifacts in {d} (run `python -m edf.train_final`)")
        self.meta = json.loads((d / "meta.json").read_text())
        self.lgbm = LGBMForecaster.load(d / "lgbm.joblib")
        self.quantile = LGBMQuantileForecaster.load(d / "quantile.joblib")
        self.nn = None
        if self.meta.get("with_nn") and (d / "nn").exists():
            from edf.models.nn import NNForecaster
            self.nn = NNForecaster.load(d / "nn")
        self.db_path, self.cache_seconds = db_path, cache_seconds
        self._cache: tuple[float, pd.DataFrame] | None = None
        self._lock = threading.Lock()

    def _dataset(self) -> pd.DataFrame:
        with self._lock:
            if self._cache and time.time() - self._cache[0] < self.cache_seconds:
                return self._cache[1]
            con = db.connect(self.db_path, read_only=True)
            try:
                df = load_dataset(con)
            finally:
                con.close()
            self._cache = (time.time(), df)
            return df

    def predict_day(self, day, weather_source: str = "archive", session=None) -> pd.DataFrame:
        if weather_source not in ("archive", "forecast"):
            raise ValueError("weather_source must be 'archive' or 'forecast'")
        day = pd.Timestamp(day).normalize()
        df = self._dataset()
        cutoff = cutoff_for_day(day, self.meta["origin_hour"])
        last = df["load"].dropna().index.max()
        if last < cutoff:
            raise ForecastNotAvailable(
                f"Load data ends at {last}; forecasting {day.date()} needs data up to {cutoff} "
                f"(day-ahead origin {self.meta['origin_hour']}:00 {TZ} on the previous day).")
        hist = df.loc[:cutoff]
        tgt = target_index_for_day(day)

        if weather_source == "forecast":
            frame = weather.fetch_forecast_frame(tgt.min().strftime("%Y-%m-%d"),
                                                 tgt.max().strftime("%Y-%m-%d"), session=session)
            exog = weight_weather_frame(frame).reindex(tgt)
        else:
            exog = df.reindex(tgt)
        exog = exog.reindex(columns=WEATHER_COLS)
        if exog["temp"].isna().mean() > 0.5:
            raise ForecastNotAvailable("No observed weather for that day in the warehouse; use weather=forecast "
                                       "(live weather forecast) or ingest more weather data.")

        lg = self.lgbm.predict(hist, tgt, exog).to_numpy()
        if self.nn is not None:
            w = self.meta["blend_weights"]
            nn = self.nn.predict(hist, tgt, exog).to_numpy()
            point = w["lgbm"] * lg + w["nn"] * nn
        else:
            nn, point = np.full(len(tgt), np.nan), lg
        q = self.quantile.predict(hist, tgt, exog)
        shift = self.meta["conformal_shift_mwh"]
        lower = np.minimum(q["q05"].to_numpy() - shift, point)
        upper = np.maximum(q["q95"].to_numpy() + shift, point)
        return pd.DataFrame({
            "ts_utc": tgt, "ts_local": tgt.tz_convert(TZ), "point_mwh": point, "lower_mwh": lower,
            "upper_mwh": upper, "lgbm_mwh": lg, "nn_mwh": nn,
        })
