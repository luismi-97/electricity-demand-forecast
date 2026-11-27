"""Command-line ingestion.

    python -m edf.ingest smard                  # resumes from the last stored hour
    python -m edf.ingest smard --start 2015-01-01 --series load residual_load
    python -m edf.ingest weather --start 2015-01-01
    python -m edf.ingest official --csv data/raw/smard_forecast.csv [--csv-resolution quarterhour]
    python -m edf.ingest all
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from edf import db, official, smard, weather
from edf.config import CITIES, DEFAULT_START, SMARD_SERIES, WEATHER_LAG_DAYS

log = logging.getLogger("edf.ingest")


def _resume_start(con, table: str, where: str, default: str, overlap_days: int = 7) -> str:
    row = con.execute(f"SELECT max(ts_utc) FROM {table} WHERE {where}").fetchall()[0][0]
    if row is None:
        return default
    return (pd.Timestamp(row).tz_convert("UTC") - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")


def ingest_smard(con, series=("load",), start=None, end=None, resolution="hour") -> None:
    for s in series:
        s_start = start or _resume_start(con, "smard_series", f"series='{s}' AND resolution='{resolution}'", DEFAULT_START)
        log.info("SMARD %s [%s] from %s to %s", s, resolution, s_start, end or "latest")
        df = smard.fetch_series(s, resolution=resolution, start=s_start, end=end)
        n = db.upsert(con, "smard_series", df)
        log.info("SMARD %s: %d rows upserted", s, n)


def ingest_weather(con, start=None, end=None) -> None:
    end = end or (date.today() - timedelta(days=WEATHER_LAG_DAYS)).isoformat()
    for city, (lat, lon, _w) in CITIES.items():
        c_start = start or _resume_start(con, "weather_hourly", f"city='{city}'", DEFAULT_START)
        log.info("Weather %s from %s to %s", city, c_start, end)
        df = weather.fetch_city(city, lat, lon, c_start, end)
        df = df.dropna(subset=["temperature_2m"])
        n = db.upsert(con, "weather_hourly", df)
        log.info("Weather %s: %d rows upserted", city, n)


def ingest_official(con, csv_path, resolution="hour", value_col=None) -> None:
    df = official.load_official_csv(csv_path, resolution=resolution, value_col=value_col)
    log.info("Official forecast: %s", official.describe_import(df))
    n = db.upsert(con, "smard_series", df)
    log.info("Official forecast: %d rows upserted", n)
    ld = con.execute("SELECT ts_utc, value FROM smard_series WHERE series='load' AND resolution='hour'").df()
    if len(ld):
        load = ld.set_index(pd.DatetimeIndex(ld["ts_utc"]).tz_convert("UTC"))["value"]
        off = df.set_index(pd.DatetimeIndex(df["ts_utc"]).tz_convert("UTC"))["value"]
        chk = official.alignment_check(off, load)
        if chk:
            best = min(chk, key=chk.get)
            log.info("Alignment check (MAE vs realised load by hour shift): %s",
                     {k: round(v) for k, v in chk.items()})
            if best != 0:
                log.warning("The forecast lines up best with the realised load after moving its timestamps by %+d h "
                            "(not 0): the CSV may use a different time zone than assumed (Europe/Berlin). "
                            "Check before using it.", best)
            else:
                log.info("Alignment OK: minimum error at shift 0.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Ingest open data into the local DuckDB warehouse")
    p.add_argument("target", choices=["smard", "weather", "official", "all"])
    p.add_argument("--csv", help="SMARD CSV export of the official forecast (target=official)")
    p.add_argument("--csv-resolution", default="hour", choices=["hour", "quarterhour"])
    p.add_argument("--value-col", default=None, help="name of the value column in the CSV (default: first numeric)")
    p.add_argument("--start", help="YYYY-MM-DD (default: resume from last stored value)")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--series", nargs="+", default=["load"], choices=sorted(SMARD_SERIES))
    p.add_argument("--db", default=None, help="path to the DuckDB file")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    con = db.connect(a.db) if a.db else db.connect()
    if a.target in ("smard", "all"):
        ingest_smard(con, a.series, a.start, a.end)
    if a.target in ("weather", "all"):
        ingest_weather(con, a.start, a.end)
    if a.target == "official":
        if not a.csv:
            p.error("target 'official' needs --csv path/to/file.csv")
        ingest_official(con, a.csv, a.csv_resolution, a.value_col)
    log.info("Done.")


if __name__ == "__main__":
    main()