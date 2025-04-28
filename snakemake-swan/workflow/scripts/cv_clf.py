import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from utils import models
from typing import Iterable
from sklearn.metrics import precision_score
from snakemake.script import snakemake as sk

def cv_clf(
    in_pulses: Iterable,
    in_labels: Iterable,
    out_prec: str,
    wc_fold: str,
    wc_model: str,
    config_models: dict,
    config_quantiles: Iterable
):
    fpulses_val = in_pulses.pop(int(wc_fold) -1)
    flabels_val = in_labels.pop(int(wc_fold) -1)
    pulses_train = pd.concat(map(pd.read_csv, in_pulses)).to_numpy()
    labels_train = pd.concat(map(pd.read_csv, in_labels)).to_numpy()[:, -3]
    pulses_val = pd.read_csv(fpulses_val).to_numpy()
    labels_val = pd.read_csv(flabels_val).to_numpy()[:, -2]
    gen = models.DefaultModel()
    scores = {}

    for params in ParameterGrid(config_models[wc_model]):
        clf = gen.instanciate(wc_model, "clf", params)
        clf.fit(pulses_train, labels_train)
        preds = clf.predict(pulses_val)
        scores[tuple(params.items())] = precision_score(
            labels_val, 
            preds, 
            average='weighted',
            zero_division=0
        )
    np.save(out_prec, scores, allow_pickle=True)


if __name__ == "__main__":
    cv_clf(
        in_pulses=sk.input["pulses"],
        in_labels=sk.input["labels"],
        out_prec=sk.output["prec"],
        wc_fold=sk.wildcards["fold"],
        wc_model=sk.wildcards["model"],
        config_models=sk.config["models"],
        config_quantiles=sk.config["quantiles"]
    )