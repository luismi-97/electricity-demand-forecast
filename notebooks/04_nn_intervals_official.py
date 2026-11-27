# %% [markdown]
# # 04 - Neural network, prediction intervals and comparison with the official forecast
# Validation period only (2023-2024). The final test set stays locked until `05_final_test.py`.
# Needs the official forecast imported first:
#   python -m edf.ingest official --csv data/raw/smard_forecast.csv
# (SMARD download centre -> Stromverbrauch -> Prognostizierter Stromverbrauch: Gesamt, Germany, hourly, CSV)
# Usage: python notebooks/04_nn_intervals_official.py [--refit-days 30] [--trees 500] [--nn-ensemble 3]

# %%
import argparse

import matplotlib
matplotlib.use("Agg")  # remove to show figures interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from edf import db, evaluation as ev
from edf.config import FIG_DIR, ROOT, VALIDATION_PERIOD
from edf.features import ALL_GROUPS, load_dataset, load_official
from edf.models.baselines import SeasonalNaive
from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster
from edf.models.nn import NNForecaster

ap = argparse.ArgumentParser()
ap.add_argument("--refit-days", type=int, default=30)
ap.add_argument("--trees", type=int, default=500)
ap.add_argument("--nn-ensemble", type=int, default=3)
ap.add_argument("--nn-epochs", type=int, default=60)
args, _ = ap.parse_known_args()

FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)
con = db.connect(read_only=True)
df = load_dataset(con)
official = load_official(con)
con.close()
start, end = VALIDATION_PERIOD
print(f"Validation {start} -> {end} | official forecast available: {len(official) > 0}")

# %% [markdown]
# ## 1. Point forecasts: naive, LightGBM, neural network, blend, official forecast

# %%
kw = dict(n_estimators=args.trees)
models = {
    "naive_168h": SeasonalNaive(168),
    "lgbm_+lags": LGBMForecaster(groups=ALL_GROUPS, name="lgbm_+lags", **kw),
    "nn_mlp": NNForecaster(n_models=args.nn_ensemble, epochs=args.nn_epochs, name="nn_mlp"),
    "lgbm_quantile": LGBMQuantileForecaster(**kw),
}
results = {}
for name, m in models.items():
    print("backtesting", name, "...", flush=True)
    results[name] = ev.backtest(df, m, start, end, refit_every_days=args.refit_days)

blend = results["lgbm_+lags"].copy()
blend["y_pred"] = (results["lgbm_+lags"]["y_pred"].to_numpy() + results["nn_mlp"]["y_pred"].to_numpy()) / 2
results["blend_lgbm_nn"] = blend
if len(official):
    results["official_tso"] = ev.series_to_result(df["load"], official, start, end, "official_tso")
else:
    print("!! No official forecast imported: skipping that comparison (see header of this file).")

scale = ev.mase_scale(df["load"].loc[: pd.Timestamp(start, tz="UTC")])
table = ev.summarise(results, scale=scale)     # identical timestamps for every model
print(table.round(2))
table.round(3).to_csv(ROOT / "reports" / "week4_validation.csv")

# %% [markdown]
# ## 2. Is the difference real? Paired bootstrap over days (negative = first model better)

# %%
ours = [n for n in ("lgbm_+lags", "nn_mlp", "blend_lgbm_nn") if n in results]
refs = [n for n in ("naive_168h", "official_tso") if n in results]
rows = {}
for a in ours:
    for b in refs:
        rows[f"{a} vs {b}"] = ev.paired_block_bootstrap(results[a], results[b])
cmp_table = pd.DataFrame(rows).T
print(cmp_table.round(3))
cmp_table.round(3).to_csv(ROOT / "reports" / "week4_paired_tests.csv")

# %% [markdown]
# ## 3. Prediction intervals (90 %): raw quantile regression vs. rolling conformal calibration

# %%
q = results["lgbm_quantile"]
qc = ev.rolling_conformal(q, window_days=90, lag_days=2, alpha=0.10)
iv = pd.DataFrame({
    "raw quantile": ev.interval_metrics(q),
    "conformalised": ev.interval_metrics(qc, suffix="_c"),
}).T
print(iv.round(3))
print("pinball loss:", {k: round(v, 1) for k, v in ev.pinball_loss(q).items()})
iv.round(3).to_csv(ROOT / "reports" / "week4_intervals.csv")

# coverage by local hour and by month (is the interval honest everywhere?)
r = qc.dropna(subset=["q05_c", "q95_c", "y_true"]).copy()
loc = pd.DatetimeIndex(r["ts_utc"]).tz_convert("Europe/Berlin")
r["inside"] = ((r["y_true"] >= r["q05_c"]) & (r["y_true"] <= r["q95_c"])).astype(float)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
r.groupby(loc.hour)["inside"].mean().plot(ax=axes[0], title="Coverage by hour (conformalised, target 0.90)")
r.groupby(loc.month)["inside"].mean().plot(ax=axes[1], title="Coverage by month")
for ax in axes:
    ax.axhline(0.9, color="red", ls="--", lw=0.8); ax.set_ylim(0.6, 1.0); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "04_interval_coverage.png", dpi=150); plt.close()

# example weeks
weeks = [pd.Timestamp("2024-01-08", tz="UTC"), pd.Timestamp("2024-07-08", tz="UTC")]
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
for ax, w0 in zip(axes, weeks):
    sub = qc[(qc["ts_utc"] >= w0) & (qc["ts_utc"] < w0 + pd.Timedelta(days=7))]
    ax.fill_between(sub["ts_utc"], sub["q05_c"], sub["q95_c"], alpha=0.3, label="90% interval")
    ax.plot(sub["ts_utc"], sub["y_true"], color="black", lw=1, label="actual")
    ax.plot(sub["ts_utc"], sub["q50"], color="tab:red", lw=1, label="forecast")
    ax.set_title(f"Week of {w0.date()}"); ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "04_interval_examples.png", dpi=150); plt.close()

# %% [markdown]
# ## 4. Where does each model win? MAE by hour and by (national) holiday

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in [n for n in ("naive_168h", "lgbm_+lags", "nn_mlp", "official_tso") if n in results]:
    ev.breakdown(results[name], "hour")["MAE"].plot(ax=ax, label=name)
ax.set(title="MAE by local hour", xlabel="hour", ylabel="MAE (MWh)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "04_mae_by_hour.png", dpi=150); plt.close()
for name in [n for n in ("lgbm_+lags", "nn_mlp", "official_tso") if n in results]:
    b = ev.breakdown(results[name], "holiday")["MAE"]
    print(f"{name:14s} MAE holidays: {b.get(True, float('nan')):7.0f} | other days: {b.get(False, float('nan')):7.0f}")