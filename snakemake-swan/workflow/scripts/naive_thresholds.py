import pandas as pd 
import numpy as np
from sklearn.metrics import roc_curve
import joblib
from snakemake.script import snakemake as sk

def naive_thresholds(
    in_spl: str,
    in_clf: str,
    out_thresholds: str
):
    #load files
    files = pd.read_csv(in_spl).query("fold == 0")
    pulses = pd.concat(map(pd.read_csv, files["pulses"])).to_numpy()
    labels = pd.concat(map(pd.read_csv, files["labels"]))
    labels = labels["category"] == 0
    clf = joblib.load(in_clf)

    #evaluate the trained classifier
    in_d = clf.predict_proba(pulses)[:, 0]
    fpr, tpr, thresh = roc_curve(labels, in_d)
    threshold = thresh[np.argmax(tpr - fpr)]

    #save
    df = pd.DataFrame(
        {"clf": threshold, "p1": pulses.shape[-1], "p2": pulses.shape[-1]},
        index=[0]
    )
    df.to_csv(out_thresholds, index=False)

if __name__ == "__main__":
    naive_thresholds(
        in_spl=sk.input["spl"],
        in_clf=sk.input["clf"],
        out_thresholds=sk.output["thresholds"],
    )
