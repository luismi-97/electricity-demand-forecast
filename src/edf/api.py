"""FastAPI service.   uvicorn edf.api:app --port 8000

Environment: EDF_MODEL_DIR (default models/v1), EDF_DB_PATH (default data/processed/warehouse.duckdb).
Endpoints: GET /health, GET /model, GET /forecast?date=YYYY-MM-DD[&weather=archive|forecast]
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from edf.serving import ForecastNotAvailable, Predictor


class HourForecast(BaseModel):
    ts_utc: str
    ts_local: str
    point_mwh: float
    lower_mwh: float
    upper_mwh: float
    lgbm_mwh: float
    nn_mwh: float | None = None


class ForecastResponse(BaseModel):
    date: dt.date
    weather_source: str
    config_hash: str
    interval: str
    hours: list[HourForecast]
    note: str


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="German day-ahead electricity demand forecast", version="1.0")
    state = {"predictor": predictor, "error": None}
    lock = threading.Lock()

    def get_predictor() -> Predictor:
        with lock:
            if state["predictor"] is None:
                try:
                    state["predictor"] = Predictor()
                except FileNotFoundError as exc:
                    state["error"] = str(exc)
        if state["predictor"] is None:
            raise HTTPException(503, state["error"] or "Model not loaded")
        return state["predictor"]

    @app.get("/health")
    def health():
        try:
            get_predictor()
            return {"status": "ok", "model_loaded": True}
        except HTTPException:
            return {"status": "degraded", "model_loaded": False}

    @app.get("/model")
    def model_info():
        m = dict(get_predictor().meta)
        return {k: m[k] for k in ("version", "config_hash", "created_at", "trained_through_utc", "with_nn",
                                  "blend_weights", "conformal_alpha", "conformal_shift_mwh", "origin_hour")}

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(date: dt.date = Query(..., description="Local target day, YYYY-MM-DD"),
                 weather: Literal["archive", "forecast"] = Query("archive")):
        pred = get_predictor()
        try:
            df = pred.predict_day(date, weather)
        except ForecastNotAvailable as exc:
            raise HTTPException(409, str(exc))
        alpha = pred.meta["conformal_alpha"]
        hours = [HourForecast(ts_utc=r.ts_utc.isoformat(), ts_local=r.ts_local.isoformat(),
                              point_mwh=round(float(r.point_mwh), 1), lower_mwh=round(float(r.lower_mwh), 1),
                              upper_mwh=round(float(r.upper_mwh), 1), lgbm_mwh=round(float(r.lgbm_mwh), 1),
                              nn_mwh=None if np.isnan(r.nn_mwh) else round(float(r.nn_mwh), 1))
                 for r in df.itertuples()]
        note = ("Replay with OBSERVED weather (optimistic: a real day-ahead forecast only has weather forecasts)."
                if weather == "archive" else "Uses a live weather forecast (not covered by the validation).")
        return ForecastResponse(date=date, weather_source=weather, config_hash=pred.meta["config_hash"],
                                interval=f"{round((1 - alpha) * 100)}% conformalised", hours=hours, note=note)

    return app


app = create_app()
