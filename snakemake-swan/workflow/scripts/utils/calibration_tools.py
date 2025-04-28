from itertools import product
from scipy.stats import binom
import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt

def research_space(series: Iterable, quantile: float, length: int) -> np.array:
    
    max_threshold = np.quantile(series, 1- 0.1 * quantile)
    min_threshold = np.quantile(series, 1-quantile)
    return np.linspace(min_threshold, max_threshold, length)

#p-values from  https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py
def binom_p_value(r_hat, n, alpha):
    return binom.cdf(np.ceil(n*r_hat),n,alpha)

def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat),n,alpha)
    def h1(y,mu):
        with np.errstate(divide='ignore'):
            return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(min(r_hat,alpha),alpha))
    return min(bentkus_p_value,hoeffding_p_value)
