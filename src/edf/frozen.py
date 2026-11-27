"""FROZEN configuration (v1) - the exact setup evaluated on the validation period 2023-2024.

This file is the single source of truth for the final test (`notebooks/05_final_test.py`). Do not edit it after the
final test has been run: any change creates a new version (v2) and the old test result stays what it is.
`config_hash()` identifies the configuration; commit and tag the repository (e.g. `git tag v1-frozen`) BEFORE
running the final test and quote the hash next to the results.
"""
from __future__ import annotations

import hashlib
import json

from edf.features import ALL_GROUPS

FROZEN = {
    "version": "v1",
    "validated_on": ["2023-01-01", "2024-12-31"],
    "final_test_start": "2025-01-01",
    "origin_hour": 12,
    "refit_days": 30,
    "lgbm": {
        "groups": list(ALL_GROUPS), "target": "raw", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "regression",
        "train_window_days": 1095, "halflife_days": 365, "seed": 0,
    },
    "nn": {
        "groups": list(ALL_GROUPS), "hidden": [128, 128], "dropout": 0.1, "lr": 0.001, "epochs": 60, "patience": 6,
        "batch_size": 256, "val_days": 60, "n_models": 3, "train_window_days": 1095, "seed": 0,
    },
    "quantile": {
        "quantiles": [0.05, 0.5, 0.95], "groups": list(ALL_GROUPS), "n_estimators": 500, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "train_window_days": 1095, "halflife_days": 365, "seed": 0,
    },
    "blend_weights": {"lgbm": 0.5, "nn": 0.5},
    "conformal": {"window_days": 90, "lag_days": 2, "alpha": 0.10},
}


def config_hash() -> str:
    return hashlib.sha256(json.dumps(FROZEN, sort_keys=True).encode()).hexdigest()[:12]


def build_models() -> dict:
    """Models of the final test, built only from FROZEN."""
    from edf.models.baselines import SeasonalNaive
    from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster
    from edf.models.nn import NNForecaster

    return {
        "naive_168h": SeasonalNaive(168),
        "lgbm_+lags": LGBMForecaster(**FROZEN["lgbm"], name="lgbm_+lags"),
        "nn_mlp": NNForecaster(**FROZEN["nn"], name="nn_mlp"),
        "lgbm_quantile": LGBMQuantileForecaster(**FROZEN["quantile"]),
    }