"""Hourly ERA5 reanalysis weather from the Open-Meteo archive API.

Data by Open-Meteo.com (CC BY 4.0, free for non-commercial use).
Attribution is required wherever results are published.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from edf.config import CITIES, OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL, WEATHER_VARIABLES


def parse_response(payload: dict, city: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["ts_utc"] = pd.to_datetime(df.pop("time"), utc=True)   # requested with timezone=UTC
    df["city"] = city
    cols = ["city", "ts_utc"] + [v for v in WEATHER_VARIABLES if v in df.columns]
    return df[cols]


def fetch_city(city: str, lat: float, lon: float, start: str, end: str, session=None, retries: int = 3,
               url: str = OPEN_METEO_ARCHIVE_URL) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    sess = session or requests
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return parse_response(resp.json(), city)
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch weather for {city}") from last_exc


def fetch_forecast_frame(start: str, end: str, session=None) -> pd.DataFrame:
    """Weather FORECAST (not reanalysis) for all configured cities, as a long city-level frame.

    Used by the API in live mode. The models were trained on observed (ERA5) weather, so forecast errors are an
    additional source of error that the validation did not include (see the sensitivity analysis in the README).
    """
    parts = [fetch_city(c, lat, lon, start, end, session=session, url=OPEN_METEO_FORECAST_URL)
             for c, (lat, lon, _w) in CITIES.items()]
    return pd.concat(parts, ignore_index=True)
