import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from utils import models
from typing import Iterable
from sklearn.metrics import root_mean_squared_error
from snakemake.script import snakemake as sk

def cv_reg(
    in_pulses: Iterable,
    in_labels: Iterable,
    out_score: str,
    wc_fold: str,
    wc_model: str,
    config_models: dict,
    config_quantiles: Iterable
):
    fpulses_val = in_pulses.pop(int(wc_fold) -1)
    flabels_val = in_labels.pop(int(wc_fold) -1)
    pulses_train = pd.concat(map(pd.read_csv, in_pulses)).to_numpy()
    labels_train = pd.concat(map(pd.read_csv, in_labels)).to_numpy()[:, -3:]
    pulses_val = pd.read_csv(fpulses_val).to_numpy()
    labels_val = pd.read_csv(flabels_val).to_numpy()[:, -3:]

    #select pulses with a calculable ratio
    idx_train = labels_train[:, 0] == 0
    pulses_train = pulses_train[idx_train, :]
    labels_train = labels_train[idx_train, 1:]

    #select pulses with a calculable ratio
    idx_val = labels_val[:, 0] == 0
    pulses_val = pulses_val[idx_val, :]
    labels_val = labels_val[idx_val, 1:]

    #cross-validation
    gen = models.DefaultModel()
    scores = {}
    preds = np.zeros((len(pulses_val), 2))

    for params in ParameterGrid(config_models[wc_model]):
        for i in [-2, -1]:
            reg = gen.instanciate(wc_model, "reg", params, config_quantiles)
            reg.fit(pulses_train, labels_train[:, i])
            pred = reg.predict(pulses_val)
            preds[:, i] = pred.mean(axis=np.argmin(pred.shape))
        scores[tuple(params.items())] = root_mean_squared_error(
            labels_val, labels_val
        )

    np.save(out_score, scores, allow_pickle=True)


if __name__ == "__main__":
    cv_reg(
        in_pulses=sk.input["pulses"],
        in_labels=sk.input["labels"],
        out_score=sk.output["score"],
        wc_fold=sk.wildcards["fold"],
        wc_model=sk.wildcards["model"],
        config_models=sk.config["models"],
        config_quantiles=sk.config["quantiles"]
    )