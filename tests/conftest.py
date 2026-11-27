import copy

import numpy as np
import pandas as pd
import pytest

from edf import db
from edf.config import CITIES
from edf.features import calendar_frame, load_dataset
from edf.frozen import FROZEN
from edf.train_final import train_final


def make_df(start="2021-01-01", end="2024-12-31 23:00", noise=0.0, seed=0, with_temp=True):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    cal = calendar_frame(idx)
    rng = np.random.default_rng(seed)
    temp = 10 - 12 * np.cos(2 * np.pi * (cal["doy"].to_numpy() - 15) / 365.25) + rng.normal(0, 2, len(idx))
    base = np.where(cal["eff_dow"] < 5, 8000, 3000) * np.sin(2 * np.pi * (cal["hour"].to_numpy() - 6) / 24)
    load = 55000 + base + 300 * np.clip(15 - temp, 0, None) + rng.normal(0, noise, len(idx)) if noise else \
        55000 + base + 300 * np.clip(15 - temp, 0, None)
    sun = np.clip(np.sin(2 * np.pi * (cal["hour"].to_numpy() - 6) / 24), 0, None) * (1 + 0.5 * np.sin(2 * np.pi * (cal["doy"].to_numpy() - 80) / 365.25))
    rad = 600 * sun * rng.uniform(0.3, 1.0, len(idx))
    load = load - 4.0 * rad
    df = pd.DataFrame({"load": load}, index=idx)
    if with_temp:
        df["temp"] = temp
        df["rad"] = rad
        df["wind"] = rng.uniform(0, 12, len(idx))
    return df


@pytest.fixture(scope="session")
def synth():
    return make_df(noise=500.0)


def small_frozen():
    f = copy.deepcopy(FROZEN)
    for k in ("lgbm", "quantile"):
        f[k].update(n_estimators=60, num_leaves=15, learning_rate=0.1)
    f["nn"].update(hidden=[16], epochs=4, patience=2, n_models=1)
    return f


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    """Synthetic DuckDB warehouse file + the dataframe it contains."""
    path = tmp_path_factory.mktemp("wh") / "w.duckdb"
    df = make_df("2020-01-01", "2024-12-31 23:00", noise=500.0)
    con = db.connect(path)
    db.upsert(con, "smard_series", pd.DataFrame({"series": "load", "resolution": "hour",
                                                 "ts_utc": df.index, "value": df["load"].to_numpy()}))
    for c in CITIES:
        db.upsert(con, "weather_hourly", pd.DataFrame({
            "city": c, "ts_utc": df.index, "temperature_2m": df["temp"].to_numpy(),
            "shortwave_radiation": df["rad"].to_numpy(), "wind_speed_10m": df["wind"].to_numpy()}))
    con.close()
    return path, load_dataset(db.connect(path, read_only=True))


@pytest.fixture(scope="session")
def artifacts(warehouse, tmp_path_factory):
    path, df = warehouse
    out = tmp_path_factory.mktemp("models")
    meta = train_final(df, out, frozen=small_frozen(), with_nn=False, calibration_days=45)
    return out, meta


