import numpy as np
from sklearn.cluster import KMeans

from glob_config import SEED


class Clusterer:
    """
    K-Means based candidate selector for active learning.

    Fits K-Means on extracted feature embeddings and selects the
    samples closest to each cluster centroid as annotation candidates.
    Distributes candidates uniformly across clusters to ensure diversity.

    Arguments
    ---------
    n_clusters : int
        Number of K-Means clusters. Should match the number of
        classes in the dataset. Default: 100.
    seed : int
        Random seed for K-Means reproducibility. Default: SEED.

    Example
    -------
    >>> clusterer = Clusterer(n_clusters=100)
    >>> clusterer.fit(features, indices)
    >>> candidates = clusterer.select_candidates(n=500)
    """

    def __init__(self, n_clusters: int = 100, seed: int = SEED) -> None:
        self.n_clusters = n_clusters
        self._kmeans = KMeans(
            n_clusters=n_clusters, random_state=seed, n_init="auto"
        )
        self._features: np.ndarray | None = None
        self._indices: np.ndarray | None = None

    def fit(self, features: np.ndarray, indices: np.ndarray) -> "Clusterer":
        """
        Fit K-Means on the provided feature embeddings.

        Arguments
        ---------
        features : np.ndarray
            Feature embeddings of shape (N, D).
        indices : np.ndarray
            Dataset indices corresponding to each embedding, shape (N,).

        Returns
        -------
        Clusterer
            The fitted clusterer (self), for method chaining.

        Example
        -------
        >>> clusterer.fit(features, indices)
        """
        self._kmeans.fit(features)
        self._features = features
        self._indices = indices
        return self

    def select_candidates(self, n: int) -> np.ndarray:
        """
        Select the n samples closest to their cluster centroids.

        Distributes n selections as evenly as possible across all clusters,
        with remainder samples assigned to the first clusters. Within each
        cluster, selects the k samples with smallest Euclidean distance
        to the centroid.

        Arguments
        ---------
        n : int
            Total number of candidates to select.

        Returns
        -------
        np.ndarray
            Dataset indices of selected candidates, shape (n,).

        Example
        -------
        >>> candidates = clusterer.select_candidates(n=500)
        >>> candidates.shape
        (500,)
        """
        if self._features is None or self._indices is None:
            raise RuntimeError("Call fit() first")

        labels: np.ndarray = self._kmeans.labels_
        centroids: np.ndarray = self._kmeans.cluster_centers_
        n_per_cluster, remainder = divmod(n, self.n_clusters)

        selected: list[int] = []
        for c in range(self.n_clusters):
            k = n_per_cluster + (1 if c < remainder else 0)
            if k == 0:
                continue

            mask = labels == c
            cluster_features = self._features[mask]
            cluster_indices = self._indices[mask]

            if len(cluster_indices) == 0:
                continue

            dists = np.linalg.norm(cluster_features - centroids[c], axis=1)
            top_k = np.argsort(dists)[:k]
            selected.extend(cluster_indices[top_k].tolist())

        return np.array(selected, dtype=np.int64)
