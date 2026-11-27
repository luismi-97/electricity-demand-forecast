"""Rolling-origin evaluation for day-ahead forecasting.

Protocol
--------
For every target day D (local time, Europe/Berlin) the forecast is issued at ORIGIN_HOUR (12:00) on D-1,
the day-ahead gate closure. The model only sees load up to the last complete hour before the origin, so the
horizon of the 24 (23/25 on DST days) target hours ranges from about 13 to 36 hours. Models are refitted every
`refit_every_days` days on data available at that time (expanding history, optional window inside the model).

Guards
------
* `hist` handed to a model is cut at the cutoff (no future load can leak in).
* Lags that would not be available at forecast time raise an error (see baselines).
* Evaluating on or after FINAL_TEST_START is blocked unless `allow_final_test=True`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edf.config import FINAL_TEST_START, ORIGIN_HOUR, TZ
from edf.features import calendar_frame
from edf.timeutils import cutoff_for_day, target_index_for_day  # noqa: F401  (re-exported)


def backtest(
    df: pd.DataFrame,
    model,
    start: str,
    end: str,
    origin_hour: int = ORIGIN_HOUR,
    refit_every_days: int = 30,
    min_train_days: int = 365,
    allow_final_test: bool = False,
    exog_noise: dict | None = None,
    noise_seed: int = 0,
) -> pd.DataFrame:
    """Run the rolling-origin backtest. Returns columns: ts_utc, y_true, y_pred, horizon_h.

    exog_noise: optional {column: std} of Gaussian noise added to the exogenous values of the TARGET hours only,
    to mimic imperfect weather forecasts (sensitivity analysis; the std values are assumptions, not measurements).
    """
    if pd.Timestamp(end) >= pd.Timestamp(FINAL_TEST_START) and not allow_final_test:
        raise ValueError(
            f"Evaluation period reaches the locked final test set (>= {FINAL_TEST_START}). "
            "Use allow_final_test=True only for the final, one-off evaluation."
        )
    days = pd.date_range(start, end, freq="D")
    rows, last_fit = [], None
    exog_cols = [c for c in df.columns if c != "load"]

    for day in days:
        cutoff = cutoff_for_day(day, origin_hour)
        hist = df.loc[:cutoff]
        if (hist["load"].dropna().index.max() - hist.index.min()) < pd.Timedelta(days=min_train_days):
            raise ValueError(f"Less than {min_train_days} days of history before {day.date()}")
        if last_fit is None or (day - last_fit).days >= refit_every_days:
            model.fit(hist)
            last_fit = day
        tgt = target_index_for_day(day)
        exog = df.reindex(tgt)[exog_cols].copy()
        if exog_noise:
            rng = np.random.default_rng([noise_seed, day.toordinal()])
            for col, sd in exog_noise.items():
                if col in exog:
                    exog[col] = exog[col] + rng.normal(0.0, sd, len(exog))
        pred = model.predict(hist, tgt, exog)
        extra = {}
        if isinstance(pred, pd.DataFrame):          # quantile forecasts: 'q50' is the point forecast
            extra = {c: pred[c].to_numpy(dtype=float) for c in pred.columns}
            point = extra["q50"]
        else:
            point = np.asarray(pred, dtype=float)
        out = pd.DataFrame({
            "ts_utc": tgt,
            "y_true": df["load"].reindex(tgt).to_numpy(),
            "y_pred": point,
            "horizon_h": ((tgt - cutoff) / pd.Timedelta(hours=1)).astype(int),
            **extra,
        })
        rows.append(out)

    res = pd.concat(rows, ignore_index=True)
    res.attrs["model"] = model.name
    return res


# ---- Metrics ---------------------------------------------------------------

def mase_scale(load: pd.Series, lag: int = 168) -> float:
    """In-sample MAE of the seasonal-naive forecast, used to scale MASE (computed on TRAINING data only)."""
    return float(np.nanmean(np.abs(load.values[lag:] - load.values[:-lag])))


def metrics(res: pd.DataFrame, scale: float | None = None) -> dict:
    r = res.dropna(subset=["y_true", "y_pred"])
    e = r["y_pred"] - r["y_true"]
    out = {
        "n": len(r),
        "MAE": e.abs().mean(),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MAPE_%": float((e.abs() / r["y_true"]).mean() * 100),
        "bias": e.mean(),
    }
    if scale:
        out["MASE"] = out["MAE"] / scale
    return out


def summarise(results: dict[str, pd.DataFrame], scale: float | None = None) -> pd.DataFrame:
    """One row per model, sorted by MAE. All models must be evaluated on the same timestamps."""
    common = None
    for res in results.values():
        ok = set(res.dropna(subset=["y_true", "y_pred"])["ts_utc"])
        common = ok if common is None else common & ok
    rows = {}
    for name, res in results.items():
        rows[name] = metrics(res[res["ts_utc"].isin(common)], scale)
    return pd.DataFrame(rows).T.sort_values("MAE")


def breakdown(res: pd.DataFrame, by: str = "hour") -> pd.DataFrame:
    """MAE by local hour of day, weekday, month or holiday flag."""
    r = res.dropna(subset=["y_true", "y_pred"]).copy()
    cal = calendar_frame(pd.DatetimeIndex(r["ts_utc"]))
    r["abs_err"] = (r["y_pred"] - r["y_true"]).abs().to_numpy()
    key = cal[by].to_numpy()
    return r.groupby(key)["abs_err"].mean().rename("MAE").to_frame()


# ---- Comparison with an external forecast (e.g. the official TSO forecast) ----------------------------------

def series_to_result(load: pd.Series, pred: pd.Series, start: str, end: str, name: str = "external") -> pd.DataFrame:
    """Turn an external hourly forecast into the same result format as `backtest` for the local days start..end."""
    idx = pd.DatetimeIndex([t for d in pd.date_range(start, end, freq="D") for t in target_index_for_day(d)])
    res = pd.DataFrame({"ts_utc": idx, "y_true": load.reindex(idx).to_numpy(),
                        "y_pred": pred.reindex(idx).to_numpy(dtype=float)})
    res.attrs["model"] = name
    return res


def paired_block_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0, tz: str = TZ) -> dict:
    """Paired comparison of the MAE of forecasts `a` and `b` on identical timestamps.

    Days are resampled as blocks (hourly errors within a day are strongly dependent). Returns the MAE difference
    a - b (negative = `a` better), its 95% bootstrap interval and the share of days on which `a` has the lower MAE.
    """
    m = a[["ts_utc", "y_true", "y_pred"]].merge(b[["ts_utc", "y_pred"]], on="ts_utc", suffixes=("_a", "_b"))
    m = m.dropna()
    day = pd.DatetimeIndex(m["ts_utc"]).tz_convert(tz).normalize()
    ea = (m["y_pred_a"] - m["y_true"]).abs().groupby(day).agg(["sum", "count"])
    eb = (m["y_pred_b"] - m["y_true"]).abs().groupby(day).agg(["sum", "count"])
    sa, sb, n = ea["sum"].to_numpy(), eb["sum"].to_numpy(), ea["count"].to_numpy()
    rng = np.random.default_rng(seed)
    D = len(n)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, D, D)
        diffs[i] = (sa[k].sum() - sb[k].sum()) / n[k].sum()
    return {
        "mae_a": sa.sum() / n.sum(), "mae_b": sb.sum() / n.sum(),
        "diff_a_minus_b": (sa.sum() - sb.sum()) / n.sum(),
        "ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5)),
        "share_days_a_better": float((sa / n < sb / n).mean()), "n_days": int(D),
    }


# ---- Prediction intervals ------------------------------------------------------------------------------------

def _q_level(col: str) -> float:
    return float("0." + col[1:]) if col.startswith("q") else float("nan")


def interval_metrics(res: pd.DataFrame, lo: str = "q05", hi: str = "q95", suffix: str = "") -> dict:
    """Empirical coverage and sharpness of the interval [lo, hi] (nominal coverage = hi level - lo level)."""
    lo_c, hi_c = lo + suffix, hi + suffix
    r = res.dropna(subset=["y_true", lo_c, hi_c])
    nominal = _q_level(hi) - _q_level(lo)
    inside = ((r["y_true"] >= r[lo_c]) & (r["y_true"] <= r[hi_c]))
    return {"n": len(r), "nominal": nominal, "coverage": float(inside.mean()),
            "mean_width": float((r[hi_c] - r[lo_c]).mean()),
            "width_%_of_load": float(((r[hi_c] - r[lo_c]) / r["y_true"]).mean() * 100)}


def pinball_loss(res: pd.DataFrame) -> dict:
    """Average pinball loss of every quantile column (names like q05, q50, q95)."""
    out = {}
    for c in [c for c in res.columns if c.startswith("q") and c[1:].isdigit()]:
        tau = _q_level(c)
        r = res.dropna(subset=["y_true", c])
        e = r["y_true"] - r[c]
        out[c] = float(np.maximum(tau * e, (tau - 1) * e).mean())
    return out


def rolling_conformal(res: pd.DataFrame, window_days: int = 90, lag_days: int = 2, alpha: float = 0.10,
                      lo: str = "q05", hi: str = "q95", tz: str = TZ) -> pd.DataFrame:
    """Conformalised quantile regression with a rolling calibration window.

    For each local day D the interval is widened by the (1-alpha) quantile of the non-conformity scores
    max(lo - y, y - hi) observed on days D-lag_days-window_days+1 .. D-lag_days. `lag_days=2` guarantees that only
    outcomes known at the 12:00 origin of D-1 are used (the actuals of D-1 are not complete yet).
    Adds the columns `{lo}_c` and `{hi}_c`; days without enough calibration history are left as NaN.
    """
    r = res.copy()
    day = pd.DatetimeIndex(r["ts_utc"]).tz_convert(tz).normalize().tz_localize(None)
    r["_day"] = day
    r["_score"] = np.maximum(r[lo] - r["y_true"], r["y_true"] - r[hi])
    days = sorted(r["_day"].unique())
    by_day = {d: g["_score"].dropna().to_numpy() for d, g in r.groupby("_day")}
    adj = {}
    for d in days:
        first = d - pd.Timedelta(days=lag_days + window_days - 1)
        last = d - pd.Timedelta(days=lag_days)
        scores = np.concatenate([by_day[k] for k in by_day if first <= k <= last] or [np.array([])])
        if len(scores) >= 24 * 30:
            level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
            adj[d] = float(np.quantile(scores, level))
    shift = r["_day"].map(adj)
    r[lo + "_c"] = r[lo] - shift
    r[hi + "_c"] = r[hi] + shift
    return r.drop(columns=["_day", "_score"])
