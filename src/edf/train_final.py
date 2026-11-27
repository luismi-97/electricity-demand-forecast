"""Train the FROZEN configuration on all available data and save the artifacts used by the API.

    python -m edf.train_final                         # models/v1/  (LightGBM, quantile model, NN, meta.json)
    python -m edf.train_final --no-nn                 # without TensorFlow (point forecast = LightGBM only)

Run this AFTER the final test has been evaluated: the deployed models are trained on every observation, including
the period that was locked during development. The interval width of the API comes from a conformal calibration on a
hold-out of the most recent days (models fitted on the data before the hold-out), not from in-sample residuals.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from edf import db, evaluation as ev
from edf.config import MODEL_DIR
from edf.features import load_dataset
from edf.frozen import FROZEN, config_hash
from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster

log = logging.getLogger("edf.train_final")


def conformal_shift(df: pd.DataFrame, frozen: dict, calibration_days: int) -> float:
    """CQR offset from a hold-out: fit on the data before the last `calibration_days` days, score those days."""
    last_day = (df["load"].dropna().index.max().tz_convert("Europe/Berlin").normalize()
                - pd.Timedelta(days=1)).tz_localize(None)                       # last day with all 24 hours
    start = (last_day - pd.Timedelta(days=calibration_days - 1)).strftime("%Y-%m-%d")
    model = LGBMQuantileForecaster(**frozen["quantile"])
    res = ev.backtest(df, model, start, last_day.strftime("%Y-%m-%d"), refit_every_days=10**6,
                      allow_final_test=True)       # deployment calibration, not an evaluation
    r = res.dropna(subset=["y_true", "q05", "q95"])
    scores = np.maximum(r["q05"] - r["y_true"], r["y_true"] - r["q95"]).to_numpy()
    alpha = frozen["conformal"]["alpha"]
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    shift = float(np.quantile(scores, level))
    log.info("Conformal shift %.0f MWh from %d hold-out hours (%s -> %s)", shift, len(scores), start, last_day.date())
    return shift


def train_final(df: pd.DataFrame, out_dir, frozen: dict | None = None, with_nn: bool = True,
                calibration_days: int = 90) -> dict:
    frozen = copy.deepcopy(frozen or FROZEN)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist = df[df["load"].notna().cumsum() > 0]
    trained_through = df["load"].dropna().index.max()

    shift = conformal_shift(df, frozen, calibration_days)

    log.info("Fitting LightGBM ...")
    LGBMForecaster(**frozen["lgbm"], name="lgbm_+lags").fit(hist).save(out / "lgbm.joblib")
    log.info("Fitting quantile model ...")
    LGBMQuantileForecaster(**frozen["quantile"]).fit(hist).save(out / "quantile.joblib")
    if with_nn:
        from edf.models.nn import NNForecaster
        log.info("Fitting neural network ensemble ...")
        NNForecaster(**frozen["nn"], name="nn_mlp").fit(hist).save(out / "nn")

    import lightgbm
    meta = {
        "version": frozen["version"], "config_hash": config_hash() if frozen is FROZEN else "custom",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through_utc": trained_through.isoformat(), "data_start_utc": df.index.min().isoformat(),
        "n_rows": int(df["load"].notna().sum()), "with_nn": with_nn,
        "blend_weights": frozen["blend_weights"], "conformal_shift_mwh": shift,
        "conformal_alpha": frozen["conformal"]["alpha"], "calibration_days": calibration_days,
        "origin_hour": frozen["origin_hour"], "lightgbm_version": lightgbm.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Saved artifacts to %s", out)
    return meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train the frozen configuration on all data and save artifacts")
    p.add_argument("--out", default=str(MODEL_DIR))
    p.add_argument("--no-nn", action="store_true", help="skip the neural network (no TensorFlow needed to serve)")
    p.add_argument("--calibration-days", type=int, default=90)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = db.connect(read_only=True)
    df = load_dataset(con)
    con.close()
    train_final(df, a.out, with_nn=not a.no_nn, calibration_days=a.calibration_days)


if __name__ == "__main__":
    main()