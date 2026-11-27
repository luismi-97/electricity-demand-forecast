"""Feed-forward neural network (TensorFlow / Keras) on the same features as the gradient-boosting model.

Design choices
--------------
* Cyclical encoding (sin/cos) of hour, weekday, month and day of year; other features standardised with
  statistics from the training window; missing values filled with training medians.
* The network predicts the CHANGE with respect to the same hour last week (`lag_168`, falling back to the 168 h
  rolling level), standardised. This residual formulation removes the level drift that hurt models without
  recent-load information.
* Early stopping on the last `val_days` of the training window (chronological split, never random).
* Optional small ensemble (`n_models`) averaging nets with different seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from edf.features import ALL_GROUPS, FEATURE_GROUPS, make_features
from edf.models.base import Forecaster
from edf.models.gbm import TAIL_ROWS

CYCLIC = {"hour": 24, "dow": 7, "month": 12, "doy": 366}


class NNForecaster(Forecaster):
    def __init__(self, groups=ALL_GROUPS, hidden=(128, 128), dropout: float = 0.1, lr: float = 1e-3,
                 epochs: int = 60, patience: int = 6, batch_size: int = 256, val_days: int = 60,
                 n_models: int = 1, train_window_days: int | None = 1095, seed: int = 0, name: str = "nn_mlp"):
        self.groups = tuple(groups)
        self.hidden, self.dropout, self.lr = tuple(hidden), dropout, lr
        self.epochs, self.patience, self.batch_size, self.val_days = epochs, patience, batch_size, val_days
        self.n_models, self.train_window_days, self.seed, self.name = n_models, train_window_days, seed, name
        self.columns = [c for g in self.groups for c in FEATURE_GROUPS[g]]
        self.nets_: list = []

    # ---- feature matrix ------------------------------------------------------
    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in self.columns:
            if c in CYCLIC:
                ang = 2 * np.pi * X[c].to_numpy(float) / CYCLIC[c]
                parts += [np.sin(ang), np.cos(ang)]
            else:
                parts.append(X[c].to_numpy(float))
        return np.column_stack(parts)

    @staticmethod
    def _base(X: pd.DataFrame, fallback: float) -> np.ndarray:
        b = X["lag_168"].to_numpy(float)
        b = np.where(np.isfinite(b), b, X["level_168"].to_numpy(float))
        return np.where(np.isfinite(b), b, fallback)

    def _build(self, n_in: int):
        import tensorflow as tf
        layers = [tf.keras.layers.Input(shape=(n_in,))]
        for u in self.hidden:
            layers += [tf.keras.layers.Dense(u, activation="relu"), tf.keras.layers.Dropout(self.dropout)]
        layers.append(tf.keras.layers.Dense(1))
        net = tf.keras.Sequential(layers)
        net.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="mae")
        return net

    # ---- Forecaster API ------------------------------------------------------
    def fit(self, hist):
        import tensorflow as tf
        if "lags" not in self.groups:
            raise ValueError("NNForecaster predicts the change vs. lag_168 and needs the 'lags' group")
        h = hist
        if self.train_window_days:
            h = h.loc[h.index.max() - pd.Timedelta(days=self.train_window_days):]
        idx = h["load"].dropna().index
        X = make_features(h, idx)
        y = h["load"].reindex(idx).to_numpy(float)
        self.fallback_ = float(np.nanmean(y))
        resid = y - self._base(X, self.fallback_)
        M = self._matrix(X)
        self.med_ = np.nanmedian(M, axis=0)
        M = np.where(np.isfinite(M), M, self.med_)
        self.mu_, self.sd_ = M.mean(axis=0), M.std(axis=0) + 1e-6
        self.y_mu_, self.y_sd_ = resid.mean(), resid.std() + 1e-6
        Z, t = (M - self.mu_) / self.sd_, (resid - self.y_mu_) / self.y_sd_

        cut = idx.max() - pd.Timedelta(days=self.val_days)
        tr, va = np.asarray(idx <= cut), np.asarray(idx > cut)
        self.nets_ = []
        for k in range(self.n_models):
            tf.keras.utils.set_random_seed(self.seed + k)
            net = self._build(Z.shape[1])
            cb = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
            net.fit(Z[tr], t[tr], validation_data=(Z[va], t[va]), epochs=self.epochs,
                    batch_size=self.batch_size, callbacks=cb, verbose=0, shuffle=True)
            self.nets_.append(net)
        return self

    def predict(self, hist, target_index, exog):
        if not self.nets_:
            raise RuntimeError("fit() first")
        ctx = pd.concat([hist.tail(TAIL_ROWS), exog.reindex(target_index).assign(load=np.nan)])
        ctx = ctx[~ctx.index.duplicated(keep="first")]
        X = make_features(ctx, target_index)
        M = self._matrix(X)
        M = np.where(np.isfinite(M), M, self.med_)
        Z = ((M - self.mu_) / self.sd_).astype("float32")
        out = np.mean([np.asarray(n(Z, training=False)).ravel() for n in self.nets_], axis=0)
        pred = self._base(X, self.fallback_) + out * self.y_sd_ + self.y_mu_
        return pd.Series(pred, index=target_index)

    # ---- persistence ----
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, net in enumerate(self.nets_):
            net.save(d / f"net_{k}.keras")
        np.savez(d / "stats.npz", med=self.med_, mu=self.mu_, sd=self.sd_,
                 y=np.array([self.y_mu_, self.y_sd_, self.fallback_]))
        (d / "config.json").write_text(json.dumps({
            "groups": list(self.groups), "n_models": len(self.nets_), "name": self.name}))

    @classmethod
    def load(cls, directory) -> "NNForecaster":
        import tensorflow as tf
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        obj = cls(groups=cfg["groups"], n_models=cfg["n_models"], name=cfg["name"])
        st = np.load(d / "stats.npz")
        obj.med_, obj.mu_, obj.sd_ = st["med"], st["mu"], st["sd"]
        obj.y_mu_, obj.y_sd_, obj.fallback_ = (float(x) for x in st["y"])
        obj.nets_ = [tf.keras.models.load_model(d / f"net_{k}.keras") for k in range(cfg["n_models"])]
        return obj
