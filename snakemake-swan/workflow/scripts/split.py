import glob
import numpy as np 
import pandas as pd
from snakemake.script import snakemake as sk

def split(
    out_spl: str,
    param_folder: str,
    config_n_folds: int,
):

    pulses = sorted(glob.glob(f"{param_folder}/pulses/*"))
    labels = sorted(glob.glob(f"{param_folder}/labels/*"))
    indices = np.random.RandomState(17).permutation(len(pulses))
    folds = np.zeros_like(indices)

    for i, idx in enumerate(np.array_split(indices, config_n_folds + 1)):
        #the 0-th fold is kept for calibration
        folds[idx] = i

    df = pd.DataFrame({"pulses": pulses, "labels": labels, "fold": folds})
    df.to_csv(sk.output["spl"], index=False)

if __name__ == "__main__":
    split(
        out_spl=sk.output["spl"],
        param_folder=sk.params["folder"],
        config_n_folds=sk.config["n_folds"]
    )