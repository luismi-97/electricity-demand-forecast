import inspect

from edf.frozen import FROZEN, build_models, config_hash
from edf.models.gbm import LGBMForecaster, LGBMQuantileForecaster
from edf.models.nn import NNForecaster


def _defaults(cls):
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items() if v.default is not inspect._empty}


def test_frozen_lgbm_matches_the_defaults_used_in_validation():
    d = _defaults(LGBMForecaster)
    for k, v in FROZEN["lgbm"].items():
        if k == "groups":
            assert tuple(v) == tuple(d[k])
        else:
            assert d[k] == v, k


def test_frozen_nn_matches_defaults_except_ensemble_size():
    d = _defaults(NNForecaster)
    for k, v in FROZEN["nn"].items():
        if k == "n_models":
            assert v == 3                       # notebook 04 default (--nn-ensemble 3)
        elif k in ("groups", "hidden"):
            assert tuple(v) == tuple(d[k]), k
        else:
            assert d[k] == v, k


def test_frozen_quantile_matches_lgbm_hyperparameters():
    for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
              "train_window_days", "halflife_days", "seed"):
        assert FROZEN["quantile"][k] == FROZEN["lgbm"][k], k
    assert tuple(FROZEN["quantile"]["quantiles"]) == _defaults(LGBMQuantileForecaster)["quantiles"]


def test_build_models_and_hash_are_stable():
    m = build_models()
    assert set(m) == {"naive_168h", "lgbm_+lags", "nn_mlp", "lgbm_quantile"}
    assert m["nn_mlp"].n_models == 3 and m["lgbm_+lags"].params["n_estimators"] == 500
    assert config_hash() == config_hash() and len(config_hash()) == 12
    assert abs(sum(FROZEN["blend_weights"].values()) - 1.0) < 1e-12