# %% [markdown]
# # 02 - Evaluation framework and baselines (validation period only)
# Protocol: day-ahead forecasts issued at 12:00 local time on D-1, rolling-origin backtest,
# models refitted every 30 days. The final test set (>= 2025-01-01) stays LOCKED.
# Weather is the OBSERVED (ERA5) temperature -> optimistic; quantified in week 3.

# %%
import matplotlib
matplotlib.use("Agg")  # remove to show figures interactively
import matplotlib.pyplot as plt
import pandas as pd

from edf import db, evaluation as ev
from edf.config import FIG_DIR, ROOT, VALIDATION_PERIOD
from edf.features import load_dataset
from edf.models.baselines import CalendarRidge, SeasonalMean, SeasonalNaive

FIG_DIR.mkdir(parents=True, exist_ok=True)
con = db.connect(read_only=True)
df = load_dataset(con)
con.close()                          # release the file lock right away
start, end = VALIDATION_PERIOD
print(f"Data: {df.index.min()} -> {df.index.max()} | validation: {start} -> {end}")

# %% [markdown]
# ## 1. Run the backtests

# %%
models = [SeasonalNaive(168), SeasonalMean(4)]
if "temp" in df:
    models.append(CalendarRidge())
else:
    print("No weather data: skipping CalendarRidge (run `python -m edf.ingest weather`).")

results = {}
for m in models:
    print("backtesting", m.name, "...")
    results[m.name] = ev.backtest(df, m, start, end, refit_every_days=30)

scale = ev.mase_scale(df["load"].loc[: pd.Timestamp(start, tz="UTC")])   # training data only
table = ev.summarise(results, scale=scale)
print(table.round(2))
out = ROOT / "reports"
out.mkdir(exist_ok=True)
table.round(3).to_csv(out / "baselines_validation.csv")

# %% [markdown]
# ## 2. Where do the errors come from?

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for name, res in results.items():
    ev.breakdown(res, "hour")["MAE"].plot(ax=axes[0], label=name)
    ev.breakdown(res, "month")["MAE"].plot(ax=axes[1], label=name)
axes[0].set(title="MAE by local hour of day", xlabel="hour", ylabel="MAE (MWh)")
axes[1].set(title="MAE by month", xlabel="month")
axes[0].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG_DIR / "02_baseline_errors.png", dpi=150); plt.close()

# %% [markdown]
# ## 3. Holidays and weekdays (do the baselines break on special days?)

# %%
for name, res in results.items():
    print(name, "| MAE on holidays:", round(ev.breakdown(res, "holiday")["MAE"].get(True, float("nan")), 1),
          "| other days:", round(ev.breakdown(res, "holiday")["MAE"].get(False, float("nan")), 1))