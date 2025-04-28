import pandas as pd
import numpy as np
from utils import models
from typing import Iterable
import joblib
from utils import models
from snakemake.script import snakemake as sk

def train_clf(
    in_pulses: Iterable,
    in_labels: Iterable,
    in_precs: Iterable,
    out_clf: str,
    out_params_clf: str,
    wc_model: str
):
    #Retrieve the best parameters
    pulses = pd.concat(map(pd.read_csv, in_pulses)).to_numpy()
    labels = pd.concat(map(pd.read_csv, in_labels)).to_numpy()[:, -3]
    metrics = [np.load(p, allow_pickle=True).item() for p in in_precs]
    vmean = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    best_params = dict(max(vmean, key=vmean.get))
    gen = models.DefaultModel()

    #Fit the classifier with the full training dataset
    clf = gen.instanciate(wc_model, "clf", best_params)
    clf.fit(pulses, labels)
    pd.DataFrame(best_params, index=[0]).to_csv(out_params_clf, index=False) 
    joblib.dump(clf, out_clf, compress=('gzip', 3)) 

if __name__ == "__main__":
    train_clf(
        in_pulses=sk.input["pulses"],
        in_labels=sk.input["labels"],
        in_precs=sk.input["precs"],
        out_clf=sk.output["clf"],
        out_params_clf=sk.output["params_clf"],   
        wc_model=sk.wildcards["model"]
    )