# %% [markdown]
# # 01 - Exploratory data analysis: German grid load
# Goal of week 1: understand the data BEFORE modelling. Run cell by cell
# (VS Code / Jupytext) or as a script: `python notebooks/01_eda.py`.

# %%
import matplotlib
matplotlib.use("Agg")  # remove this line to show figures interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edf import db
from edf.config import FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
con = db.connect(read_only=True)   # read-only: does not block other readers
load = (con.execute("SELECT ts_utc, value FROM smard_series WHERE series='load' AND resolution='hour' ORDER BY ts_utc")
        .df().set_index("ts_utc")["value"])
if load.empty:
    raise SystemExit("No load data yet: run `python -m edf.ingest smard` first.")
load.index = pd.DatetimeIndex(load.index).tz_convert("UTC")

# %% [markdown]
# ## 1. Data quality: coverage, gaps, duplicates

# %%
full = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC")
missing = full.difference(load.index)
print(f"Period: {load.index.min()} -> {load.index.max()}  ({len(load):,} hourly values)")
print(f"Missing hours: {len(missing):,} ({len(missing) / len(full):.3%})")
if len(missing):
    gaps = pd.Series(missing).diff().ne(pd.Timedelta("1h")).cumsum()
    print("Longest gap (h):", pd.Series(missing).groupby(gaps.values).size().max())
print(load.describe().round(1))
load = load.reindex(full)  # explicit NaNs for gaps (do NOT silently interpolate yet)

# %% [markdown]
# ## 2. Load profiles: hour of day x day type, and yearly seasonality
# Timestamps are stored in UTC; convert to local time for calendar effects (handles DST).

# %%
local = load.tz_convert("Europe/Berlin")
df = pd.DataFrame({"load": local.values, "hour": local.index.hour, "dow": local.index.dayofweek,
                   "month": local.index.month, "year": local.index.year}, index=local.index)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for dow, label in [(0, "Mon"), (2, "Wed"), (5, "Sat"), (6, "Sun")]:
    df[df.dow == dow].groupby("hour")["load"].mean().plot(ax=axes[0], label=label)
axes[0].set(title="Mean daily profile by weekday", xlabel="hour (Europe/Berlin)", ylabel="load (MWh)")
axes[0].legend()
df.groupby("month")["load"].mean().plot(kind="bar", ax=axes[1], title="Mean load by month")
df.groupby("year")["load"].mean().plot(marker="o", ax=axes[2], title="Mean load by year")
plt.tight_layout(); plt.savefig(FIG_DIR / "01_profiles.png", dpi=150); plt.close()

# %% [markdown]
# ## 3. Structural breaks: COVID-19 (2020) and the 2022 energy crisis
# Decide later how to treat these periods (keep, down-weight, or flag with a dummy).

# %%
weekly = load.resample("W").mean()
fig, ax = plt.subplots(figsize=(14, 4))
weekly.plot(ax=ax, lw=0.8)
for d, lbl in [("2020-03-22", "COVID lockdown"), ("2022-02-24", "energy crisis")]:
    ax.axvline(pd.Timestamp(d, tz="UTC"), color="red", ls="--", alpha=0.6)
    ax.text(pd.Timestamp(d, tz="UTC"), weekly.max(), lbl, rotation=90, va="top", fontsize=8)
ax.set(title="Weekly mean load", ylabel="MWh")
plt.tight_layout(); plt.savefig(FIG_DIR / "01_weekly_breaks.png", dpi=150); plt.close()

# %% [markdown]
# ## 4. Autocorrelation: which lags matter? (expect peaks at 24 h and 168 h)

# %%
lags = range(1, 24 * 8 + 1)
s = load.interpolate(limit=6)
acf = [s.autocorr(lag=l) for l in lags]
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(list(lags), acf)
for l in (24, 48, 168):
    ax.axvline(l, color="grey", ls=":")
ax.set(title="Autocorrelation of hourly load", xlabel="lag (hours)", ylabel="ACF")
plt.tight_layout(); plt.savefig(FIG_DIR / "01_acf.png", dpi=150); plt.close()
print({f"lag {l}h": round(s.autocorr(lag=l), 3) for l in (1, 24, 168)})

# %% [markdown]
# ## 5. Weather relationship (population-weighted temperature)
# Load usually has a U-shaped relation with temperature (heating below ~15 C).

# %%
from edf.config import CITIES
w = con.execute("SELECT city, ts_utc, temperature_2m FROM weather_hourly").df()
if len(w):
    w["ts_utc"] = pd.DatetimeIndex(w["ts_utc"]).tz_convert("UTC")
    wt = {c: v[2] for c, v in CITIES.items()}
    w["wt"] = w["city"].map(wt)
    temp = (w.assign(x=w.temperature_2m * w.wt).groupby("ts_utc")["x"].sum()
            / w.groupby("ts_utc")["wt"].sum())
    j = pd.concat([load.rename("load"), temp.rename("temp")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(j.temp, j.load, s=2, alpha=0.15)
    ax.set(title="Load vs. weighted temperature", xlabel="temperature (C)", ylabel="load (MWh)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_load_vs_temp.png", dpi=150); plt.close()
    print("corr(load, temp) =", round(j.corr().iloc[0, 1], 3))
else:
    print("No weather data yet: run `python -m edf.ingest weather`.")