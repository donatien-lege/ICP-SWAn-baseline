import pandas as pd
import numpy as np
from utils import models
from typing import Iterable
import joblib
from utils import models
from snakemake.script import snakemake as sk

def train_reg(
    in_pulses: Iterable,
    in_labels: Iterable,
    in_scores: Iterable,
    out_reg_P1: str,
    out_reg_P2: str,
    out_params_reg: str,
    wc_model: str,
    config_quantiles: Iterable
):
    #Retrieve the best parameters
    pulses = pd.concat(map(pd.read_csv, in_pulses)).to_numpy()
    labels = pd.concat(map(pd.read_csv, in_labels)).to_numpy()[:, -3:]
    select = labels == 0
    pulses, labels = pulses[select], labels[select]
    labels = labels[:, 1:]
    metrics = [np.load(p, allow_pickle=True).item() for p in in_scores]
    vmean = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    best_params = dict(min(vmean, key=vmean.get))
    pd.DataFrame(best_params, index=[0]).to_csv(out_params_reg, index=False) 

    #Each peak has its own regression model
    gen = models.DefaultModel()
    for i, out_file in enumerate([out_reg_P1, out_reg_P2]):
        reg = gen.instanciate(wc_model, "reg", best_params, config_quantiles)
        reg.fit(pulses, labels[:, i])
        joblib.dump(reg, out_file, compress=('gzip', 3)) 

if __name__ == "__main__":
    train_reg(
        in_pulses=sk.input["pulses"],
        in_labels=sk.input["labels"],
        in_scores=sk.input["scores"],
        out_reg_P1=sk.output["reg_P1"],
        out_reg_P2=sk.output["reg_P2"],
        out_params_reg=sk.output["params_reg"],
        wc_model=sk.wildcards["model"],
        config_quantiles=sk.config["quantiles"],
    )