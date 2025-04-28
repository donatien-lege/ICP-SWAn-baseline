import pandas as pd
import numpy as np
import joblib
from utils.compute_ratio import compute_ratio
from snakemake.script import snakemake as sk
from utils.wroclaw_classification import psi

def metrics(df: pd.DataFrame):
    
    dico = {}
    full = {}

    #R0
    dico["R0"] = sum((~df["ok"]) & (df["category"] == 1)) / sum(df["category"] == 1) * 100

    #R1
    idx = (df["ok"] > 0) 
    R1 = np.abs((df["ratio_pred"][idx] - df["ratio_label"][idx]) / df["ratio_label"])
    dico["R1"] = sum(R1 > 0.1) / sum(idx) * 100

    #dP1
    idx = df["category"] & df["ok"]
    dico["dP1"] = (df["delta_P1"][idx] > 10).mean() * 100

    #dP2
    idx = df["category"] & df["ok"]
    dico["dP2"] = (df["delta_P2"][idx] > 10).mean() * 100

    #MAE
    dico["MAE"] = np.mean(np.abs(df["ratio_pred"][idx] - df["ratio_label"][idx]))
    return dico


def eval_model(
    param_folder: str,
    param_sampling_rate: int,
    in_thresholds: str, 
    in_clf: str,  
    in_reg_P1: str,  
    in_reg_P2: str,  
    out_metrics: str
):
    pulses = pd.read_csv(f"{param_folder}/pulses.csv").to_numpy()
    labels = pd.read_csv(f"{param_folder}/labels.csv")
    labels = labels.loc[:, ["category", "final_P1", "final_P2"]]
    labels = labels.to_numpy()
    lengths = pd.read_csv(f"{param_folder}/lengths.csv")
    thresholds = pd.read_csv(in_thresholds)
    
    #load models
    models = []
    for file in [in_clf, in_reg_P1, in_reg_P2]:
        models.append(joblib.load(file))
    df = compute_ratio(*models, pulses, labels)

    #delays in ms + add wroclaw classification
    factors = ((lengths / pulses.shape[1]) * (1000 / param_sampling_rate)).squeeze()
    df["delta_P1"] = df["delta_P1"] * factors
    df["delta_P2"] = df["delta_P2"] * factors
    df["shape"] = psi(pulses)
    cdt1 = df["ood"] <= thresholds.loc[0, "clf"]
    cdt2 = df["width_P1"] <= thresholds.loc[0, "p1"]
    cdt3 = df["width_P2"] <= thresholds.loc[0, "p2"]
    df["ok"] = cdt1 & cdt2 & cdt3

    #compute metrics grouped by shape
    d = {}
    for i in range(5):
        idx = df["shape"] == i
        d[i] = metrics(df.loc[idx, :])
    d["all"] = metrics(df)
    res = pd.DataFrame(d).T.round(3)
    res.to_csv(out_metrics, index_label="PSI")

if __name__ == "__main__":
    eval_model(
        param_folder=sk.params["folder"],
        param_sampling_rate=sk.params["sampling_rate"],
        in_thresholds=sk.input["thresholds"],
        in_clf=sk.input["clf"], 
        in_reg_P1=sk.input["reg_P1"], 
        in_reg_P2=sk.input["reg_P2"], 
        out_metrics=sk.output[0]
    )


