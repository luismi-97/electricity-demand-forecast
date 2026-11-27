# %% [markdown]
# # 05 - FINAL TEST (one shot)
# Evaluates the frozen model configuration on the locked period (>= 2025-01-01).
# RULES: choose the configuration on the validation period (notebooks 02-04) BEFORE running this; run it ONCE;
# report the numbers as they come out, whatever they are. Re-running after changing the models turns the test set
# into a second validation set and the result would no longer be an honest estimate.
#
# Usage: python notebooks/05_final_test.py --confirm-final-test   (all settings come from edf/frozen.py)

# %%
import argparse
import sys

import pandas as pd

from edf import db, evaluation as ev
from edf.config import FINAL_TEST_START, ROOT, VALIDATION_PERIOD
from edf.features import load_dataset, load_official
from edf.frozen import FROZEN, build_models, config_hash

ap = argparse.ArgumentParser()
ap.add_argument("--confirm-final-test", action="store_true")
args, _ = ap.parse_known_args()
if not args.confirm_final_test:
    sys.exit("Refusing to run: this evaluates the LOCKED test set. Re-run with --confirm-final-test "
             "once the model configuration is final (commit + tag the repo first).")

con = db.connect(read_only=True)
df = load_dataset(con)
official = load_official(con)
con.close()
last_full_day = (df["load"].dropna().index.max().tz_convert("Europe/Berlin").normalize()
                 - pd.Timedelta(days=1)).tz_localize(None)       # last day with all 24 hours
if len(official):
    last_full_day = min(last_full_day, official.index.max().tz_convert("Europe/Berlin").normalize().tz_localize(None)
                        - pd.Timedelta(days=1))
start, end = FINAL_TEST_START, last_full_day.strftime("%Y-%m-%d")
assert FROZEN["final_test_start"] == FINAL_TEST_START and tuple(FROZEN["validated_on"]) == tuple(VALIDATION_PERIOD)
print(f"FINAL TEST {start} -> {end} | frozen config {FROZEN['version']} (hash {config_hash()}) validated on {VALIDATION_PERIOD}")

# %%
models = build_models()
results = {n: ev.backtest(df, m, start, end, refit_every_days=FROZEN["refit_days"], allow_final_test=True)
           for n, m in models.items()}
w = FROZEN["blend_weights"]
blend = results["lgbm_+lags"].copy()
blend["y_pred"] = w["lgbm"] * results["lgbm_+lags"]["y_pred"].to_numpy() + w["nn"] * results["nn_mlp"]["y_pred"].to_numpy()
results["blend_lgbm_nn"] = blend
if len(official):
    results["official_tso"] = ev.series_to_result(df["load"], official, start, end, "official_tso")

scale = ev.mase_scale(df["load"].loc[: pd.Timestamp(VALIDATION_PERIOD[0], tz="UTC")])
table = ev.summarise(results, scale=scale)
print(table.round(2))
table.round(3).to_csv(ROOT / "reports" / "final_test.csv")

# %%
pairs = [("blend_lgbm_nn", "official_tso"), ("blend_lgbm_nn", "naive_168h"),      # primary
         ("lgbm_+lags", "official_tso"), ("lgbm_+lags", "naive_168h"),               # secondary
         ("blend_lgbm_nn", "lgbm_+lags"), ("blend_lgbm_nn", "nn_mlp")]               # is the blend worth it?
rows = {f"{a} vs {b}": ev.paired_block_bootstrap(results[a], results[b]) for a, b in pairs if a in results and b in results}
cmp_table = pd.DataFrame(rows).T
print(cmp_table.round(3))
cmp_table.round(3).to_csv(ROOT / "reports" / "final_test_paired.csv")

qc = ev.rolling_conformal(results["lgbm_quantile"], **FROZEN["conformal"])
iv = pd.DataFrame({"raw quantile": ev.interval_metrics(results["lgbm_quantile"]),
                   "conformalised": ev.interval_metrics(qc, suffix="_c")}).T
print(iv.round(3))
iv.round(3).to_csv(ROOT / "reports" / "final_test_intervals.csv")