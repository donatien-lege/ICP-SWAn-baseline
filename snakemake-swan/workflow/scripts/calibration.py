import pandas as pd
import numpy as np
from utils import models
from typing import Iterable
import joblib
from utils.compute_ratio import compute_ratio
from snakemake.script import snakemake as sk
import utils.calibration_tools as ct
from functools import partial
from itertools import product
import pandas as pd


def pvalue(
    c: pd.DataFrame, preds: pd.Series, l0: float, l1: float, l2: float,
    alpha_0: float, alpha_1: float
) -> tuple[float]:

    valids = c["category"] == 0
    n_valids = sum(valids)

    #R0
    rej = (c["ood"] > l0) | (c["width_P1"] > l1) | (c["width_P2"] > l2)
    r0 = sum(valids & rej) / n_valids

    #R1
    preds[rej] = -1
    r1 = np.abs((c["ratio_pred"] - c["ratio_label"]) / c["ratio_label"]) 
    r1 = (r1 > 0.1).mean()
    r1 -= alpha_1 * sum(rej / len(c))
    r1 = r1.clip(0, 1)

    return max(
        ct.binom_p_value(r0, n_valids, alpha_0), ct.hb_p_value(r1, len(c), alpha_1)
    )


def calibration(
    in_spl: str,
    in_clf: str,
    in_reg_P1: str, 
    in_reg_P2: str,
    out_thresholds: str,
    config_set: Iterable
):
    #load files
    alpha_0, alpha_1, delta = config_set
    files = pd.read_csv(in_spl).query("fold == 0")
    pulses = pd.concat(map(pd.read_csv, files["pulses"])).to_numpy()
    labels = pd.concat(map(pd.read_csv, files["labels"]))
    labels = labels.loc[:, ["category", "final_P1", "final_P2"]]
    labels = labels.to_numpy()

    #load models
    models = []
    for file in [in_clf, in_reg_P1, in_reg_P2]:
        models.append(joblib.load(file))
    df = compute_ratio(*models, pulses, labels)
    
    #random split into two equal subsets
    df = df.sample(frac=1, random_state=1)
    c1 = df.iloc[:len(df) // 2, :]
    c2 = df.iloc[len(df) // 2:, :]

    #research space definition
    thresh_clf = ct.research_space(df["ood"], alpha_0, 100)
    thresh_p1 = ct.research_space(df["width_P1"], alpha_0, 25)
    thresh_p2 = ct.research_space(df["width_P2"], alpha_0, 25)

    #search for a reasonable testing path
    pval = partial(pvalue, alpha_0=alpha_0, alpha_1=alpha_1)
    valids = c1["category"] == 1
    pvalues = {}

    for threshs in product(thresh_clf, thresh_p1, thresh_p2):
        pvalues[threshs] = pval(c1, c1["ratio_pred"].copy(), *threshs)
    D = np.linspace(0.001, 1, 100)
    seq = [min(pvalues, key=lambda x: abs(pvalues[x] - d)) for d in D]
    triplets = list(dict.fromkeys(seq))
    #final evaluation
    pvalues = {}
    for t in triplets:
        pvalues[t] = pval(c2, c2["ratio_pred"].copy(), *t)
    triplets = sorted(pvalues, key=pvalues.get)
    final_pvalues = np.array(sorted(pvalues.values()))

    #we choose the last triplet of the path before a p-value >= delta
    #if all the p-values >= delta, we pick the first triplet of the path
    try:
        valid = np.where(final_pvalues < delta)[0]
        idx = valid[-1]
    except IndexError:
        idx = 0
    df = pd.DataFrame(dict(zip(("clf", "p1", "p2"), triplets[idx])), index=[0])
    df.to_csv(out_thresholds, index=False)
    
if __name__ == "__main__":
    calibration(
        in_spl=sk.input["spl"],
        in_clf=sk.input["clf"],
        in_reg_P1=sk.input["reg_P1"],
        in_reg_P2=sk.input["reg_P2"],
        out_thresholds=sk.output["thresholds"],
        config_set=sk.config["sets"][sk.wildcards["s"]]
    )