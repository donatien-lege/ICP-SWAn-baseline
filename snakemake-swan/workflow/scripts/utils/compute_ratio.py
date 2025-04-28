import pandas as pd
from typing import Iterable
from utils.candidates import positions
import numpy as np
import matplotlib.pyplot as plt

def compute_ratio(
    clf: object,
    reg_P1: object,
    reg_P2: object,
    pulses: Iterable,
    labels: pd.DataFrame
) -> pd.DataFrame:

    df = pd.DataFrame()

    #classification
    out_clf = clf.predict_proba(pulses)
    df["category"] = labels[:, 0] == 0
    df["ood"] = 1 - out_clf[:, 0]

    #interval widths
    reg_P1 = reg_P1.predict(pulses)
    reg_P2 = reg_P2.predict(pulses)
    out_reg = np.concatenate(
        (reg_P1, reg_P2), axis=np.argmin(reg_P1.shape)
    )

    if out_reg.shape[0] < out_reg.shape[1]:
        out_reg = out_reg.T
    df["width_P1"] = out_reg[:, 1] - out_reg[:, 0]
    df["width_P2"] = out_reg[:, 3] - out_reg[:, 2]
    df["lP1"] = out_reg[:, 0]
    df["uP1"] = out_reg[:, 1]
    df["lP2"] = out_reg[:, 2]
    df["uP2"] = out_reg[:, 3]

    #ratio preds
    pos_preds = np.stack([positions(p, o) for p, o in zip(pulses, out_reg)])
    pos_preds = pos_preds.astype(int)
    df.loc[pos_preds[:, 0] == 1, "ood"] = 1
    all = np.arange(len(pos_preds))
    pred_P1 = pulses[all, pos_preds[:, 0]] 
    pred_P2 = pulses[all, pos_preds[:, 1]]
    df["ratio_pred"] = pred_P2 / pred_P1

    #ratio labels
    all = np.arange(len(pos_preds))
    true_P1 = pulses[all, labels[:, 1].astype(int)] 
    true_P2 = pulses[all, labels[:, 2].astype(int)]
    df["ratio_label"] = true_P2 / true_P1

    #detection delays
    df["delta_P1"] = np.abs(pos_preds[:, 0] - labels[:, 1])
    df["delta_P2"] = np.abs(pos_preds[:, 1] - labels[:, 2])

    return df