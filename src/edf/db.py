"""DuckDB warehouse: a single local file, plain SQL, idempotent upserts."""
from __future__ import annotations

import duckdb
import pandas as pd

from edf.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS smard_series (
    series      VARCHAR   NOT NULL,
    resolution  VARCHAR   NOT NULL,
    ts_utc      TIMESTAMPTZ NOT NULL,
    value       DOUBLE,
    PRIMARY KEY (series, resolution, ts_utc)
);
CREATE TABLE IF NOT EXISTS weather_hourly (
    city                 VARCHAR   NOT NULL,
    ts_utc               TIMESTAMPTZ NOT NULL,
    temperature_2m       DOUBLE,
    shortwave_radiation  DOUBLE,
    wind_speed_10m       DOUBLE,
    PRIMARY KEY (city, ts_utc)
);
"""

COLUMNS = {
    "smard_series": ["series", "resolution", "ts_utc", "value"],
    "weather_hourly": ["city", "ts_utc", "temperature_2m", "shortwave_radiation", "wind_speed_10m"],
}


def connect(path=DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse. `read_only=True` (used by the API) neither creates the file nor the schema."""
    if str(path) != ":memory:" and not read_only:
        import os
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE rows keyed on the table's primary key. Safe to re-run."""
    if df.empty:
        return 0
    cols = COLUMNS[table]
    for c in cols:
        if c not in df.columns:
            df = df.assign(**{c: None})
    data = df[cols]  # noqa: F841  (referenced by DuckDB's replacement scan)
    con.register("_incoming", data)
    con.execute(f"INSERT OR REPLACE INTO {table} SELECT {', '.join(cols)} FROM _incoming")
    con.unregister("_incoming")
    return len(df)
