"""LightGBM day-ahead forecaster."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from edf.features import ALL_GROUPS, FEATURE_GROUPS, make_features
from edf.models.base import Forecaster

TAIL_ROWS = 24 * 30   # history needed at prediction time (rolling windows + 336 h lag)


class LGBMForecaster(Forecaster):
    """Gradient boosting on calendar / solar / weather / recent-load features.

    target='raw'    : predict the load.
    target='ratio'  : predict load / load(t-168h) and multiply back, which absorbs slow level drift.
    train_window_days / halflife_days : train on recent history with exponentially decaying sample weights.
    """

    def __init__(self, groups=ALL_GROUPS, target: str = "raw", n_estimators: int = 500, learning_rate: float = 0.05,
                 num_leaves: int = 63, min_child_samples: int = 50, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, objective: str = "regression", train_window_days: int | None = 1095,
                 halflife_days: float | None = 365, seed: int = 0, name: str | None = None):
        if target not in ("raw", "ratio"):
            raise ValueError("target must be 'raw' or 'ratio'")
        if target == "ratio" and "lags" not in groups:
            raise ValueError("target='ratio' needs the 'lags' group (lag_168)")
        self.groups, self.target = tuple(groups), target
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
                           colsample_bytree=colsample_bytree, objective=objective, random_state=seed,
                           verbosity=-1, n_jobs=-1)
        self.train_window_days, self.halflife_days = train_window_days, halflife_days
        self.name = name or f"lgbm_{'+'.join(self.groups)}_{target}"
        self.columns = [c for g in self.groups for c in FEATURE_GROUPS[g]]
        self.model_: lgb.LGBMRegressor | None = None

    def _training_table(self, hist):
        h = hist
        if self.train_window_days:
            h = h.loc[h.index.max() - pd.Timedelta(days=self.train_window_days):]
        idx = h["load"].dropna().index
        X = make_features(h, idx)
        y = h["load"].reindex(idx).to_numpy()
        if self.target == "ratio":
            base = X["lag_168"].to_numpy()
            ok = np.isfinite(base) & (base > 0)
            X, y, idx = X[ok], y[ok] / base[ok], idx[ok]
        age_days = np.asarray((idx.max() - idx) / pd.Timedelta(days=1))
        w = np.exp(-np.log(2) * age_days / self.halflife_days) if self.halflife_days else None
        return X, y, w

    def fit(self, hist):
        X, y, w = self._training_table(hist)
        self.model_ = lgb.LGBMRegressor(**self.params).fit(X[self.columns], y, sample_weight=w)
        return self

    def predict(self, hist, target_index, exog):
        if self.model_ is None:
            raise RuntimeError("fit() first")
        ctx = pd.concat([hist.tail(TAIL_ROWS), exog.reindex(target_index).assign(load=np.nan)])
        ctx = ctx[~ctx.index.duplicated(keep="first")]
        X = make_features(ctx, target_index)
        pred = self.model_.predict(X[self.columns])
        if self.target == "ratio":
            pred = pred * X["lag_168"].to_numpy()
        return pd.Series(pred, index=target_index)

    def feature_importance(self) -> pd.Series:
        imp = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.columns).sort_values(ascending=False)

    # ---- persistence (joblib = pickle: only load files you created yourself) ----
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"groups": self.groups, "target": self.target, "name": self.name, "model": self.model_}, path)

    @classmethod
    def load(cls, path) -> "LGBMForecaster":
        d = joblib.load(path)
        obj = cls(groups=d["groups"], target=d["target"], name=d["name"])
        obj.model_ = d["model"]
        return obj


class LGBMQuantileForecaster(LGBMForecaster):
    """Quantile regression with LightGBM: one model per quantile -> prediction intervals.

    `predict` returns a DataFrame with columns q05, q50, q95 (for the default quantiles); `q50` is the point
    forecast used by the backtest. Quantiles are sorted row-wise to avoid crossing.
    """

    def __init__(self, quantiles=(0.05, 0.5, 0.95), **kw):
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include 0.5 (point forecast)")
        kw.setdefault("name", "lgbm_quantile")
        super().__init__(target="raw", **kw)
        self.quantiles = tuple(sorted(quantiles))
        self.models_: dict[float, lgb.LGBMRegressor] = {}

    @staticmethod
    def col(q: float) -> str:
        return f"q{int(round(q * 100)):02d}"

    def fit(self, hist):
        X, y, w = self._training_table(hist)
        self.models_ = {}
        for q in self.quantiles:
            params = {**self.params, "objective": "quantile", "alpha": q}
            self.models_[q] = lgb.LGBMRegressor(**params).fit(X[self.columns], y, sample_weight=w)
        self.model_ = self.models_[0.5]
        return self

    def predict(self, hist, target_index, exog):
        if not self.models_:
            raise RuntimeError("fit() first")
        ctx = pd.concat([hist.tail(TAIL_ROWS), exog.reindex(target_index).assign(load=np.nan)])
        ctx = ctx[~ctx.index.duplicated(keep="first")]
        X = make_features(ctx, target_index)[self.columns]
        preds = np.column_stack([self.models_[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=1)
        return pd.DataFrame(preds, index=target_index, columns=[self.col(q) for q in self.quantiles])

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"groups": self.groups, "name": self.name, "quantiles": self.quantiles,
                     "models": self.models_}, path)

    @classmethod
    def load(cls, path) -> "LGBMQuantileForecaster":
        d = joblib.load(path)
        obj = cls(quantiles=d["quantiles"], groups=d["groups"], name=d["name"])
        obj.models_ = d["models"]
        obj.model_ = obj.models_[0.5]
        return obj
