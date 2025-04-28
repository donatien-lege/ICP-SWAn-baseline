import numpy as np 
from scipy.signal import argrelmin  
from scipy.stats import linregress 
from typing import Iterable

#to straighten up a pulse
def fitline(x0, x1, y0, y1, length):
    final_x = np.arange(length)
    slope, intercept, r, p, se = linregress((x0, x1), (y0, y1))
    return intercept + final_x * slope

def drawline(pulse):
    idxmax = np.argmax(pulse)
    if (idxmax == len(pulse) - 1) or (idxmax == 0):
        x0, x1 = 0, len(pulse) -1
    else:
        x0 = np.argmin(pulse[:idxmax])
        x1 = np.argmin(pulse[idxmax:]) + idxmax
    y0, y1 = pulse[x0], pulse[x1]
    line = fitline(x0, x1, y0, y1, len(pulse))
    return line - min(line)

def straightenup(pulses: Iterable):
    lines = np.apply_along_axis(drawline, 1, pulses)
    return pulses - lines


#Peak designation
def designate_peak(cds, v1, v2):
    scores = (cds - v1)**2 + np.abs(cds - v2)**2
    return cds[np.argmin(scores)]

def candidates(pulse):
    diff1 = np.gradient(pulse * 100)
    diff2 = np.gradient(diff1)
    curve = diff2 / (1 + diff1**2) ** (3 / 2)
    cds = argrelmin(curve)[0]
    return cds[curve[cds] < 0]

def positions(pulse, output_nn):
    local_max = candidates(pulse)
    #P1
    try:
        vmin = output_nn[0]
        vmax = output_nn[1] + 1
        p1 = designate_peak(local_max, vmin, vmax)
    except ValueError:
        return np.array([-1, -1])
    #P2
    try:
        local_max = local_max[local_max > p1]
        vmin = max(output_nn[2], p1)
        vmax = output_nn[3] + 1
        p2 = designate_peak(local_max, vmin, vmax)
    except ValueError:
        return np.array([-1, -1])
    return np.array([p1, p2])