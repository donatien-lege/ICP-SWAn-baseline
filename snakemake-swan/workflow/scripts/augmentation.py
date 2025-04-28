import sklearn  
import pandas as pd 
import numpy as np
from utils.wroclaw_classification import psi
from utils.synthetize_data import SynthData
from snakemake.script import snakemake as sk

def generate(
    pulses: np.array, 
    shapes: np.array,
    labels: np.array,
    n_components: int = 5,
    n_neighbors: int = 10,
    var: int = 10, 
    n_gen: int = 1,
) -> tuple[np.array, np.array]:

    sd = SynthData()
    sd.fit(pulses)
    chunks_synth_pulses = []
    chunks_synth_labels = []

    for i in range(n_gen):
        synth_pulses = sd.generate(n_components, n_neighbors, var)
        synth_labels = sd.adjust_labels(labels, synth_pulses)
        synth_psi = psi(synth_pulses)
        corrupted = (synth_psi == 4) & (shapes != 4)
        synth_labels[corrupted, -3] = 2
        chunks_synth_pulses.append(synth_pulses)
        chunks_synth_labels.append(synth_labels)

    return np.concatenate(chunks_synth_pulses), np.concatenate(chunks_synth_labels)


def augmentation(
    param_folder: str,
    in_spl: str,
    out_pulses: str,
    out_labels: str,
    wc_fold: int,
    config_iterations: dict,
):
    #load dataset
    wc_fold = int(wc_fold)
    df = pd.read_csv(in_spl).query("fold == @wc_fold")
    labels = pd.concat(map(pd.read_csv, df["labels"]))
    labels = labels.loc[:, ["category", "final_P1", "final_P2"]].to_numpy()
    pulses = pd.concat(map(pd.read_csv, df["pulses"])).to_numpy()
    shapes = psi(pulses)

    #augmentation
    tot_pulses = [pulses]
    tot_labels = [labels]
    for key, args in config_iterations.items():
        if key == -1:
            p, l = generate(pulses, shapes, labels, **args)
        else:
            idx = labels[:, -3] == key
            p, l = generate(pulses[idx], shapes[idx], labels[idx], **args)
        tot_pulses.append(p)
        tot_labels.append(l)

    pulses = np.concatenate(tot_pulses)
    labels = np.concatenate(tot_labels)

    pd.DataFrame(pulses).to_csv(out_pulses, index=False)
    pd.DataFrame(labels).to_csv(out_labels, index=False)

if __name__ == "__main__":
    augmentation(
        param_folder=sk.params["folder"],
        in_spl=sk.input["spl"],
        out_pulses=sk.output["pulses"],
        out_labels=sk.output["labels"],
        wc_fold=sk.wildcards["fold"],
        config_iterations=sk.config["iterations"],
    )