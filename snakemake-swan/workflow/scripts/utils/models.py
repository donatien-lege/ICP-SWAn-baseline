from dataclasses import dataclass
from typing import Iterable
from utils import mlp
import sklearn_quantile
import sklearn.ensemble
import sklearn.neighbors
import xgboost


@dataclass
class Model():
    model: object
    default_params: dict


class DefaultModel():
    algorithms = {
        "XGB": {
            "clf" : Model(
                        model=xgboost.XGBClassifier,
                        default_params={
                            "tree_method": "hist",
                            "subsample": 0.5,
                        },
                ),
            "reg" : Model(
                        model=xgboost.XGBRegressor,
                        default_params={
                            "objective": "reg:quantileerror",
                            "tree_method": "hist",
                            "subsample": 0.5,
                        },
                ),
            "arg_quantile": "quantile_alpha"
        },

        "RF": {
            "clf" : Model(sklearn.ensemble.RandomForestClassifier, {}),
            "reg" : Model(sklearn_quantile.SampleRandomForestQuantileRegressor, {}),
            "arg_quantile": "q"
        },

        "KNN": {
            "clf" : Model(sklearn.neighbors.KNeighborsClassifier, {}),
            "reg" : Model(sklearn_quantile.KNeighborsQuantileRegressor, {}),
            "arg_quantile": "q"
        },

        "MLP": {
            "clf" : Model(mlp.DenseClassifier, {}),
            "reg" : Model(mlp.DenseRegressor,  {}),
            "arg_quantile": "quantile"
        },
    }

    @classmethod
    def instanciate(
            cls, 
            algo: str, 
            task: str, 
            kwargs: dict,
            quantiles: Iterable = None
        ):
        args = cls.algorithms[algo][task].default_params | kwargs
        if quantiles is not None:
            args[cls.algorithms[algo]["arg_quantile"]] = quantiles
        return cls.algorithms[algo][task].model(**args)
