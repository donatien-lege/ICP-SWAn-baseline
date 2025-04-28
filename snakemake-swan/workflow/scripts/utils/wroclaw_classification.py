from .ProcessingTools.pulse_classifier import Classifier
from .ProcessingTools import model
import torch
import numpy as np
from scipy.signal import argrelmax, argrelmin
from importlib import resources as impresources
from os.path import join
import pandas as pd

def designate_peak(candidates, v1, v2):
    scores = (candidates - v1)**2 + np.abs(candidates - v2)**2
    return candidates[np.argmin(scores)]

def candidates(pulse):
    diff1 = np.gradient(pulse * 100)
    diff2 = np.gradient(diff1)
    curve = diff2 / (1 + diff1**2) ** (3 / 2)
    cds = argrelmin(curve)[0]
    return cds[curve[cds] < 0]

def positions(pulse, output_nn):
    local_max = candidates(pulse)
    try:
        vmin = int(output_nn[0])
        vmax = int(output_nn[1]) + 1
        p1 = designate_peak(local_max, vmin, vmax)
        local_max = local_max[local_max > p1]
        vmin = max(int(output_nn[2]), p1)
        vmax = int(output_nn[3]) + 1
        p2 = designate_peak(local_max, vmin, vmax)
    except ValueError:
        return np.empty(2)
    return np.array([p1, p2])

def classify(chunks, device):
    clf = GRUNet()
    weights = join(impresources.files(P1P2), "weights_classifier.pth")
    clf.load_state_dict(torch.load(weights))
    clf.to(device).eval()
    output = torch.concatenate([clf(c).squeeze() for c in chunks])
    preds = torch.argmax(output, axis=1)
    conformal = torch.max(output, axis=1).values > 0.6
    decisions = (preds == 0) & conformal
    return decisions.cpu().detach().numpy().squeeze()

def detect(chunks, device):
    clf = GRUQuantile()
    weights = join(impresources.files(P1P2), "weights_detector.pth")
    clf.load_state_dict(torch.load(weights, weights_only=True))
    clf.to(device).eval()
    preds = torch.concatenate([clf(c).squeeze() for c in chunks])
    preds = preds.cpu().detach().numpy().squeeze()
    margins = (preds[:, 3] - preds[:, 2] + preds[:, 1] - preds[:, 0]) / 2
    conformal = margins < 30
    preds[~conformal] = np.empty(preds.shape[-1])
    return preds

def ratio(input, digitize=False, smooth=True, min=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = np.array_split(input, max(1, len(input) // 256))
    chunks = [torch.Tensor(s).to(device).float().unsqueeze(0) for s in split]
    
    #processing
    have_ratio = classify(chunks, device)
    output = detect(chunks, device)
    conformal = (have_ratio) & (~np.isnan(output[:, 0]))
    
    #detection
    pos = np.zeros((len(input), 2))
    tuples = zip(input[conformal], output[conformal])
    try:
        pos[conformal] = [positions(i, o) for i, o in tuples]
    except ValueError:
        pass
    idx = (pos[:, 0] > 1) & (pos[:, 1] > 1)
    
    #ratio
    ratios = np.empty(len(input))
    tups = zip(input[idx], pos[idx, 1].astype(int), pos[idx, 0].astype(int))
    ratios[idx] = [pulse[p2] / pulse[p1] for pulse, p2, p1 in tups]
    ratios[np.logical_or(ratios < 0.1, ratios > 4)] = np.nan

    #post processing
    if digitize:
        ratios = np.digitize(ratios, digitize)
    if smooth:
        ratios = pd.Series(ratios).rolling(window=100, min_periods=min).apply(np.nanmean)
        ratios = ratios.values.squeeze() 
    return ratios

def peak_positions(input):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = np.array_split(input, max(1, len(input) // 256))
    chunks = [torch.Tensor(s).to(device).float().unsqueeze(0) for s in split]
    
    #processing
    have_ratio = classify(chunks, device)
    output = detect(chunks, device)
    conformal = (have_ratio) & (~np.isnan(output[:, 0]))
    
    #detection
    pos = np.zeros((len(input), 2))
    tuples = zip(input[conformal], output[conformal])
    try:
        pos[conformal] = [positions(i, o) for i, o in tuples]
    except ValueError:
        pass
    idx = (pos[:, 0] > 1) & (pos[:, 1] > 1)
    #ratio
    final_pos = np.empty((len(input), 2))
    final_pos[idx] = pos[idx, :2]
    return final_pos.astype(int).clip(0, 180)

###PSI### 

def window(a, w=4, o=10):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view

def psi(input: np.array, smooth=False, window_shape=300, gap=10): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #inputs
    split = np.array_split(input, max(1, len(input) // 256))
    chunks = [torch.Tensor(s).to(device).float() for s in split]
    #NN
    weights = join(str(impresources.files(model)),'model_weights.pth')
    params = {"classification":
                {"model_weights": weights,
                "batch_size": 256,
                "gpu": True,
                "resampling":True
                }
            }
    clf = Classifier(params)
    rnn = clf.model
    rnn.eval()
    #evals
    preds = [torch.argmax(rnn(c.unsqueeze(1)), axis=1) for c in chunks]
    preds = torch.concatenate(preds)
    preds = preds.cpu().detach().numpy().squeeze().astype(float)
    if smooth:
        preds = [np.nanmean(w) for w in window(preds, w=window_shape, o=gap)]
        preds = np.array(preds)
    return preds
        
    
    
