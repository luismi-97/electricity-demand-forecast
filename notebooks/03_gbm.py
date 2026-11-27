# %% [markdown]
# # 03 - Feature engineering and gradient boosting (validation period only)
# Same protocol as week 2 (day-ahead at 12:00 local time, rolling origin, locked final test).
# Questions: (1) do recent-load features remove the level drift seen in the ridge baseline?
# (2) do holidays/solar features fix the holiday and 17:00 errors? (3) how much does the
# 'oracle weather' assumption flatter the results?
# Usage: python notebooks/03_gbm.py [--refit-days 30] [--trees 500] [--skip-sensitivity]

# %%
import argparse

import matplotlib
matplotlib.use("Agg")  # remove to show figures interactively
import matplotlib.pyplot as plt
import pandas as pd

from edf import db, evaluation as ev
from edf.config import FIG_DIR, ROOT, VALIDATION_PERIOD
from edf.features import ALL_GROUPS, load_dataset
from edf.models.baselines import CalendarRidge, SeasonalNaive
from edf.models.gbm import LGBMForecaster

ap = argparse.ArgumentParser()
ap.add_argument("--refit-days", type=int, default=30)
ap.add_argument("--trees", type=int, default=500)
ap.add_argument("--skip-sensitivity", action="store_true")
args, _ = ap.parse_known_args()

FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)
con = db.connect(read_only=True)
df = load_dataset(con)
con.close()
assert "temp" in df, "Run `python -m edf.ingest weather` first"
start, end = VALIDATION_PERIOD
kw = dict(n_estimators=args.trees)

# %% [markdown]
# ## 1. Ablation: which feature groups matter?

# %%
models = {
    "naive_168h": SeasonalNaive(168),
    "calendar_ridge": CalendarRidge(),
    "lgbm_cal+weather": LGBMForecaster(groups=("calendar", "weather"), name="lgbm_cal+weather", **kw),
    "lgbm_+solar": LGBMForecaster(groups=("calendar", "weather", "solar"), name="lgbm_+solar", **kw),
    "lgbm_+lags": LGBMForecaster(groups=ALL_GROUPS, name="lgbm_+lags", **kw),
    "lgbm_ratio": LGBMForecaster(groups=ALL_GROUPS, target="ratio", name="lgbm_ratio", **kw),
}
results = {}
for name, m in models.items():
    print("backtesting", name, "...", flush=True)
    results[name] = ev.backtest(df, m, start, end, refit_every_days=args.refit_days)

scale = ev.mase_scale(df["load"].loc[: pd.Timestamp(start, tz="UTC")])
table = ev.summarise(results, scale=scale)
print(table.round(2))
table.round(3).to_csv(ROOT / "reports" / "gbm_validation.csv")
best = table.index[table.index.str.startswith("lgbm")][0]
print("best LightGBM variant:", best)

# %% [markdown]
# ## 2. Did we fix the problems of week 2? (bias, holidays, hours, months)

# %%
r = results[best].dropna()
r = r.assign(err=r.y_pred - r.y_true)
loc = pd.DatetimeIndex(r.ts_utc).tz_convert("Europe/Berlin")
print("\nMean error (bias) by year/month, best model:")
print(r.groupby([loc.year, loc.month])["err"].mean().unstack(0).round(0))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for name in ["naive_168h", "calendar_ridge", best]:
    ev.breakdown(results[name], "hour")["MAE"].plot(ax=axes[0], label=name)
    ev.breakdown(results[name], "month")["MAE"].plot(ax=axes[1], label=name)
axes[0].set(title="MAE by local hour", xlabel="hour", ylabel="MAE (MWh)")
axes[1].set(title="MAE by month", xlabel="month")
axes[0].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "03_gbm_errors.png", dpi=150); plt.close()

for name in ["naive_168h", "calendar_ridge", best]:
    b = ev.breakdown(results[name], "holiday")["MAE"]
    print(f"{name:16s} MAE holidays (national): {b.get(True, float('nan')):8.0f} | other days: {b.get(False, float('nan')):8.0f}")

# %% [markdown]
# ## 3. Feature importance (gain) of the best model, last refit

# %%
imp = models[best].feature_importance()
print(imp.head(15).round(0))
ax = imp.head(15)[::-1].plot(kind="barh", figsize=(7, 5), title=f"Gain importance - {best}")
plt.tight_layout(); plt.savefig(FIG_DIR / "03_feature_importance.png", dpi=150); plt.close()

# %% [markdown]
# ## 4. How optimistic is 'oracle weather'? Add Gaussian noise to the temperature of the target hours.
# The noise levels are ASSUMPTIONS to show sensitivity, not measured forecast errors. The proper fix is to use
# archived weather FORECASTS (e.g. Open-Meteo historical forecast data) - to be evaluated in a later iteration.

# %%
if not args.skip_sensitivity:
    rows = {}
    for sd in (0.0, 1.0, 2.0, 3.0):
        res = ev.backtest(df, models[best], start, end, refit_every_days=args.refit_days,
                          exog_noise={"temp": sd} if sd else None)
        rows[f"temp noise sd={sd} C"] = ev.metrics(res, scale)
    sens = pd.DataFrame(rows).T
    print(sens.round(2))
    sens.round(3).to_csv(ROOT / "reports" / "gbm_weather_sensitivity.csv")