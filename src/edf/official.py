"""Official day-ahead load forecast of the German TSOs, as published on SMARD / ENTSO-E.

The SMARD chart-data API does not (to our knowledge) expose this series, so it is imported from the CSV export of
SMARD's download centre: category "Stromverbrauch" -> "Prognostizierter Stromverbrauch: Gesamt", region Germany,
resolution hour (or quarter-hour), CSV. Typical format: ';' separated, local timestamps in the first two columns ("from" / "to"), then one
value column per series. Both locales are recognised: German ("01.01.2023 00:00", 41.792,50) and English
("Jan 1, 2023 12:00 AM", 41,792.50). With several value columns the first numeric one is used (override with
`value_col`); in the English export this is "grid load", the second column being the residual load.
The loader is tolerant, but CHECK the result against the file the first time (see `describe_import`).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from edf.config import TZ

SERIES = "official_forecast"


_EN_DECIMAL = re.compile(r"\.\d{1,2}$")
_DE_DECIMAL = re.compile(r",\d{1,2}$")


def _detect_decimal(values: pd.Series) -> str:
    """Return '.' (English: 41,792.50) or ',' (German: 41.792,50) from the values of one column."""
    v = values.dropna().astype(str).str.strip().head(2000)
    votes = {".": 0, ",": 0}
    for x in v:
        if "." in x and "," in x:
            votes["." if x.rfind(".") > x.rfind(",") else ","] += 1
        elif _EN_DECIMAL.search(x):
            votes["."] += 1
        elif _DE_DECIMAL.search(x):
            votes[","] += 1
    return "." if votes["."] > votes[","] else ","      # ties / thousands-only: assume German SMARD default


def _to_number(s: pd.Series, decimal: str | None = None) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype(float)
    decimal = decimal or _detect_decimal(s)
    thousands = "," if decimal == "." else "."
    cleaned = (s.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
               .str.replace(thousands, "", regex=False).str.replace(decimal, ".", regex=False)
               .replace({"-": np.nan, "": np.nan, "nan": np.nan}))
    return pd.to_numeric(cleaned, errors="coerce")


DATE_FORMATS = ("%d.%m.%Y %H:%M", "%b %d, %Y %I:%M %p", "%m/%d/%Y %I:%M %p", "%Y-%m-%d %H:%M")


def _parse_dates(col: pd.Series) -> pd.Series:
    """Try the known SMARD export date formats (German and English locale); use the one that parses best."""
    best, best_ok = None, -1
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(col, format=fmt, errors="coerce")
        ok = parsed.notna().sum()
        if ok > best_ok:
            best, best_ok = parsed, ok
    if best_ok < 0.9 * col.notna().sum():
        raise ValueError(f"Could not parse the date column (examples: {col.dropna().head(3).tolist()})")
    return best


def load_official_csv(path_or_buffer, resolution: str = "hour", value_col: str | None = None,
                      tz: str = TZ) -> pd.DataFrame:
    """Parse a SMARD CSV into the warehouse format: series, resolution, ts_utc, value (MWh per hour).

    resolution: 'hour' (values are MWh per hour) or 'quarterhour' (MWh per 15 min; four values are summed per hour,
    hours with fewer than four values are dropped).
    """
    if resolution not in ("hour", "quarterhour"):
        raise ValueError("resolution must be 'hour' or 'quarterhour'")
    raw = pd.read_csv(path_or_buffer, sep=";", dtype=str, encoding="utf-8-sig")
    if raw.shape[1] < 3:
        raise ValueError("Expected a ';'-separated file with at least 3 columns (from, to, value)")
    start_col = raw.columns[0]
    ts_local = _parse_dates(raw[start_col])
    if value_col is None:
        candidates = [c for c in raw.columns[2:] if _to_number(raw[c]).notna().any()]
        if not candidates:
            raise ValueError("No numeric value column found")
        value_col = candidates[0]
    vals = _to_number(raw[value_col])
    ok = ts_local.notna()
    df = pd.DataFrame({"ts_local": ts_local[ok], "value": vals[ok]}).sort_values("ts_local", kind="stable")
    # local -> UTC; repeated autumn hour is disambiguated by order, missing spring hour does not occur in data
    idx = pd.DatetimeIndex(df["ts_local"]).tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    s = pd.Series(df["value"].to_numpy(), index=idx.tz_convert("UTC"))
    s = s[~s.index.duplicated(keep="first")].dropna()
    if resolution == "quarterhour":
        g = s.groupby(s.index.floor("h"))
        s = g.sum()[g.count() == 4]
    out = s.rename_axis("ts_utc").reset_index(name="value")
    out.insert(0, "resolution", "hour")
    out.insert(0, "series", SERIES)
    return out[["series", "resolution", "ts_utc", "value"]]


def describe_import(df: pd.DataFrame) -> str:
    full = pd.date_range(df["ts_utc"].min(), df["ts_utc"].max(), freq="h", tz="UTC")
    return (f"{len(df):,} hourly values from {df['ts_utc'].min()} to {df['ts_utc'].max()} | "
            f"missing hours: {len(full) - len(df):,} | mean {df['value'].mean():,.0f} MWh")


def alignment_check(official: pd.Series, load: pd.Series, max_shift: int = 3) -> dict:
    """Sanity check of the time zone handling: MAE between the realised load and the official forecast whose
    timestamps are moved by -max_shift..+max_shift hours. A correctly aligned import has its minimum at shift 0
    (the forecast follows the daily profile of the realised load). A minimum at shift s means the forecast timestamps
    should be moved by s hours to line up, i.e. a time zone / DST problem in the import."""
    j = pd.concat([load.rename("load"), official.rename("off")], axis=1, join="inner").dropna()
    if len(j) < 24 * 30:
        return {}
    out = {}
    for sh in range(-max_shift, max_shift + 1):
        shifted = official.shift(sh, freq="h").rename("off")
        k = pd.concat([load.rename("load"), shifted], axis=1, join="inner").dropna()
        out[sh] = float((k["load"] - k["off"]).abs().mean())
    return out
