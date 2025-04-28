from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
import numpy as np
from numpy.random import multivariate_normal as mnorm
from utils.candidates import straightenup, candidates


class SynthData:
    def __init__(self,  **kwargs) -> None:
        self.pca = PCA(whiten=False)
        self.nn = NearestNeighbors(**kwargs)
        self.dataset = None

    def fit(self, arr: np.array) -> None:
        self.dataset = arr
        self.pca.fit(arr)


    @staticmethod
    def randomize_weights(x: np.array, var: int, n_neighbors: int) -> np.array:
        weights = x + mnorm(np.zeros(n_neighbors), var * np.identity(n_neighbors))
        vmin, vmax = min(weights), max(weights)
        return (weights - vmin) / (vmax - vmin)


    def generate(self, n_dim: int, n_neigh: int, std: float = 0) -> np.array:

        cds = self.pca.transform(self.dataset)
        synth_cds = np.copy(cds)
        self.nn.fit(cds[:, :n_dim])
        cds = np.expand_dims(cds, 1)

        # KNN on n_dim first components
        for i, ex in enumerate(cds):
            knn = self.nn.kneighbors(ex[:, :n_dim], n_neigh, return_distance=False)
            cov = np.cov(cds.squeeze()[knn.squeeze(), :n_dim].squeeze().T)
            noise = mnorm(np.zeros(n_dim), cov = std * cov)
            synth_cds[i, :n_dim] = ex[:, :n_dim] + noise

        # preprocessing
        synth_pulses = self.pca.inverse_transform(synth_cds)
        return np.apply_along_axis(minmax_scale, 1, straightenup(synth_pulses))


    @staticmethod
    def adjust_labels(labels: np.array, synth: np.array) -> np.array:
        for i, ((_, p1, p2), pulse) in enumerate(zip(labels, synth)):
            cds = candidates(pulse)
            vmax = np.argmax(pulse)
            p1 = cds[np.argmin(np.abs(cds - p1))]
            p2 = cds[np.argmin(np.abs(cds - p2))]
            labels[i, -2] = p1
            labels[i, -1] = p2

        return labels
