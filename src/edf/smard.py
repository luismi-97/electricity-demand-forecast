"""Client for the public SMARD chart-data endpoints (Bundesnetzagentur).

URL scheme (no authentication required):
    index : {BASE}/{filter}/{region}/index_{resolution}.json
            -> {"timestamps": [ms, ...]}          start of each data chunk
    chunk : {BASE}/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
            -> {"series": [[ts_ms, value|null], ...]}

NOTE: check the first live run against smard.de; if the format changed,
only `fetch_index` / `fetch_chunk` need to be adapted.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from edf.config import SMARD_BASE_URL, SMARD_REGION, SMARD_SERIES

log = logging.getLogger(__name__)
RESOLUTIONS = {"quarterhour", "hour", "day", "week", "month", "year"}


def _get_json(url: str, session: requests.Session | None = None, retries: int = 3, pause: float = 1.0) -> dict:
    sess = session or requests
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            log.warning("GET %s failed (%s), attempt %d/%d", url, exc, attempt, retries)
            time.sleep(pause * attempt)
    raise RuntimeError(f"Could not fetch {url}") from last_exc


def fetch_index(filter_id: int, region: str, resolution: str, session=None) -> list[int]:
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    return sorted(int(t) for t in _get_json(url, session)["timestamps"])


def fetch_chunk(filter_id: int, region: str, resolution: str, timestamp: int, session=None) -> list[list]:
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    return _get_json(url, session).get("series", [])


def fetch_series(
    series: str = "load",
    region: str = SMARD_REGION,
    resolution: str = "hour",
    start: str | None = None,
    end: str | None = None,
    session=None,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Download one SMARD series between start and end (inclusive, UTC).

    Returns a long DataFrame with columns: series, resolution, ts_utc, value.
    Values are in SMARD's native unit (MWh per time step); missing values are NaN
    and are dropped.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
    filter_id = SMARD_SERIES[series]
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1) if end else None

    stamps = fetch_index(filter_id, region, resolution, session)
    if start_ts is not None:  # keep the chunk that CONTAINS start, plus all later ones
        s_ms = int(start_ts.timestamp() * 1000)
        before = [t for t in stamps if t < s_ms]
        stamps = ([max(before)] if before else []) + [t for t in stamps if t >= s_ms]
    if end_ts is not None:
        e_ms = int(end_ts.timestamp() * 1000)
        stamps = [t for t in stamps if t <= e_ms]

    rows: list[list] = []
    for i, t in enumerate(stamps, 1):
        rows.extend(fetch_chunk(filter_id, region, resolution, t, session))
        if i % 50 == 0:
            log.info("%s: %d/%d chunks", series, i, len(stamps))
        time.sleep(pause)

    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["ts_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    if start_ts is not None:
        df = df[df["ts_utc"] >= start_ts]
    if end_ts is not None:
        df = df[df["ts_utc"] <= end_ts]
    df = df.drop_duplicates(subset="ts_utc").sort_values("ts_utc")
    df["series"] = series
    df["resolution"] = resolution
    return df[["series", "resolution", "ts_utc", "value"]].reset_index(drop=True)
