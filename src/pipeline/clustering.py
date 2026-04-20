from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from glob_config import HAS_CUML, SEED

if HAS_CUML:
    from cuml.cluster import HDBSCAN  # type: ignore
else:
    from sklearn.cluster import HDBSCAN

# Minimum samples in a cluster before it is eligible for splitting.
_MIN_SPLIT_SIZE = 8


@dataclass
class ClusterResult:
    """
    Output of a clustering operation.

    Provides a sinle data type passed into all QueryStrategy implementations.
    Cluster labels are always remapped to 0..K-1 so that centroids[i] is the
    centroid for cluster i. Noise points (from density-based clustering)
    carry label -1.

    Attributes
    ----------
    features : np.ndarray, shape (N, D)
        Feature embeddings of the clustered samples.
    indices : np.ndarray, shape (N,)
        Dataset indices corresponding to each embedding.
    labels : np.ndarray, shape (N,)
        Cluster assignment per sample; -1 denotes noise.
    centroids : np.ndarray, shape (K, D)
        One centroid per cluster, indexed by label value.
    """

    features: np.ndarray
    indices: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray

    def filter(self, mask: np.ndarray) -> "ClusterResult":
        """
        Return a view of this result restricted to the given boolean mask.

        Centroid array is shared unchanged; only features, indices, and
        labels are masked.

        Arguments
        ---------
        mask : np.ndarray, shape (N,), dtype bool
            True for samples to keep.

        Returns
        -------
        ClusterResult
            A new ClusterResult with masked rows and the same centroids.

        Example
        -------
        >>> non_noise = result.filter(result.labels != -1)
        """
        return ClusterResult(
            features=self.features[mask],
            indices=self.indices[mask],
            labels=self.labels[mask],
            centroids=self.centroids,
        )


class Clusterer(ABC):
    """
    Abstract base class for clustering-based candidate selectors.

    Subclasses fit a clustering model on feature embeddings and return a
    ClusterResult that query strategies can operate on.

    Example
    -------
    >>> clusterer = KMeansClusterer(n_clusters=100)
    >>> result = clusterer.fit(features, indices)
    >>> candidates = query_strategy.select(result, n=500)
    """

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        labeled_targets: dict[int, int] | None = None,
    ) -> ClusterResult:
        """
        Fit the clustering model and return a ClusterResult.

        Arguments
        ---------
        features : np.ndarray, shape (N, D)
            Feature embeddings to cluster.
        indices : np.ndarray, shape (N,)
            Dataset indices corresponding to each embedding.
        labeled_targets : dict[int, int] | None
            Mapping from dataset index to ground-truth class for every
            currently labeled sample. Used by purity-aware clusterers to
            skip splitting clusters that are already pure. Ignored by
            clusterers that do not use purity guidance.

        Returns
        -------
        ClusterResult
            Cluster assignments, centroids, and raw features.
        """


class KMeansClusterer(Clusterer):
    """
    K-Means clusterer.

    Cluster labels are 0..n_clusters-1, matching the row order of
    cluster_centers_. No noise points are produced.

    Arguments
    ---------
    n_clusters : int
        Number of clusters. Should match the number of classes in the
        dataset for best results. Default: 100.
    seed : int
        Random seed for reproducibility. Default: SEED.

    Example
    -------
    >>> clusterer = KMeansClusterer(n_clusters=100)
    >>> result = clusterer.fit(features, indices)
    """

    def __init__(self, n_clusters: int = 100, seed: int = SEED) -> None:
        self.n_clusters = n_clusters
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10 if HAS_CUML else "auto",
        )

    def fit(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        _labeled_targets: dict[int, int] | None = None,
    ) -> ClusterResult:
        """
        Fit K-Means and return a ClusterResult with real centroids.

        Example
        -------
        >>> result = clusterer.fit(features, indices)
        >>> result.centroids.shape
        (100, 64)
        """
        self._kmeans.fit(features)
        return ClusterResult(
            features=features,
            indices=indices,
            labels=self._kmeans.labels_,
            centroids=self._kmeans.cluster_centers_,
        )


class DensityClusterer(Clusterer):
    """
    HDBSCAN-based clusterer.

    The number of clusters is inferred from the data. Noise points are
    assigned label -1 and excluded from candidate selection by query
    strategies. Cluster IDs are remapped to 0..K-1 and centroids are
    computed as cluster means.

    Arguments
    ---------
    min_cluster_size : int
        Minimum samples for a group to form a cluster. Smaller values
        yield more, finer-grained clusters. Default: 5.
    seed : int
        Stored for API consistency; HDBSCAN is deterministic. Default: SEED.

    Example
    -------
    >>> clusterer = DensityClusterer(min_cluster_size=10)
    >>> result = clusterer.fit(features, indices)
    """

    def __init__(self, min_cluster_size: int = 5, seed: int = SEED) -> None:
        self._hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
        self._seed = seed

    def fit(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        _labeled_targets: dict[int, int] | None = None,
    ) -> ClusterResult:
        """
        Fit HDBSCAN and return a ClusterResult with cluster-mean centroids.

        Cluster IDs are remapped to 0..K-1 so centroids[i] always
        corresponds to label i, matching the KMeansClusterer interface.

        Example
        -------
        >>> result = clusterer.fit(features, indices)
        >>> (result.labels == -1).sum()  # number of noise points
        """
        self._hdbscan.fit(features)
        original_labels = self._hdbscan.labels_

        unique_clusters = sorted(
            c for c in np.unique(original_labels) if c != -1
        )
        label_map = {orig: new for new, orig in enumerate(unique_clusters)}
        labels = np.array(
            [label_map.get(int(label), -1) for label in original_labels],
            dtype=np.int64,
        )

        if unique_clusters:
            centroids = np.stack(
                [
                    features[labels == i].mean(axis=0)
                    for i in range(len(unique_clusters))
                ]
            )
            # Since DBSCAN labels all outliers as -1 instead of
            # assigning them to a cluster, we just force assign
            # them to their nearest cluster so that it doesn't
            # vanish
            noise_mask = labels == -1
            if noise_mask.any():
                dists = np.linalg.norm(
                    features[noise_mask, None] - centroids[None], axis=2
                )
                labels[noise_mask] = dists.argmin(axis=1)
        else:
            centroids = np.empty((0, features.shape[1]), dtype=features.dtype)

        return ClusterResult(
            features=features,
            indices=indices,
            labels=labels,
            centroids=centroids,
        )


class HierarchicalKMeansClusterer(Clusterer):
    """
    Hierarchical clusterer that bisects each cluster along its principal axis.

    Runs an initial K-Means, then for each cluster finds the direction of
    maximum variance via SVD and splits the cluster into two along that axis
    using K-Means(k=2). This process repeats for "split_depth" levels,
    progressively increasing cluster purity without requiring ground-truth
    labels.

    When "labeled_targets" is supplied to fit(), clusters whose labeled
    members already reach "purity_threshold" are left unsplit, matching the
    POC's purity-guided stopping criterion.

    Reference
    ---------
    - https://medium.com/@atharv4study/hierarchical-k-means-clustering-algorithm-f61f3a2bec93

    Arguments
    ---------
    n_clusters : int
        Number of initial K-Means clusters. Default: 100.
    split_depth : int
        Number of bisection passes to apply after the initial clustering.
        Each pass at most doubles the cluster count. Default: 1.
    min_split_size : int
        Minimum samples a cluster must contain to be eligible for splitting.
        Smaller clusters are kept as-is. Default: _MIN_SPLIT_SIZE.
    purity_threshold : float
        Fraction of labeled members that must share the majority class for a
        cluster to be considered pure and skipped during splitting.
        Only used when labeled_targets is provided. Default: 0.85.
    min_labeled_for_purity : int
        Minimum number of labeled members a cluster must contain before
        purity-guided stopping is applied. Clusters with fewer labeled
        members are always split as usual. Default: 3.
    seed : int
        Random seed for K-Means reproducibility. Default: SEED.

    Example
    -------
    >>> clusterer = HierarchicalKMeansClusterer(n_clusters=100, split_depth=1)
    >>> result = clusterer.fit(features, indices)
    >>> result.centroids.shape[0]  # up to 200 clusters after one split pass
    """

    def __init__(
        self,
        n_clusters: int = 100,
        split_depth: int = 1,
        min_split_size: int = _MIN_SPLIT_SIZE,
        purity_threshold: float = 0.85,
        min_labeled_for_purity: int = 3,
        seed: int = SEED,
    ) -> None:
        self._n_clusters = n_clusters
        self._split_depth = split_depth
        self._min_split_size = min_split_size
        self._purity_threshold = purity_threshold
        self._min_labeled_for_purity = min_labeled_for_purity
        self._seed = seed

    def fit(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        labeled_targets: dict[int, int] | None = None,
    ) -> ClusterResult:
        """
        Fit initial K-Means then iteratively bisect each cluster.

        When labeled_targets is provided, clusters whose labeled members are
        already pure (purity >= purity_threshold) are skipped each pass.

        Example
        -------
        >>> result = clusterer.fit(
        ...     features,
        ...     indices,
        ...     labeled_targets={0: 3, 1: 3}
        ... )
        """
        result = KMeansClusterer(self._n_clusters, self._seed).fit(
            features, indices
        )
        for _ in range(self._split_depth):
            result = self._split_pass(result, labeled_targets)
        return result

    def _split_pass(
        self,
        result: ClusterResult,
        labeled_targets: dict[int, int] | None,
    ) -> ClusterResult:
        """
        One bisection pass: split every eligible cluster along its principal
        axis, skipping clusters that are already pure.

        Clusters smaller than min_split_size, with degenerate variance, or
        whose labeled members satisfy the purity threshold are kept intact.
        The returned ClusterResult has remapped contiguous labels and freshly
        computed centroids.
        """
        new_labels = np.full_like(result.labels, fill_value=-1)
        next_id = 0

        for c in sorted(set(result.labels.tolist())):
            if c == -1:
                continue
            mask = result.labels == c
            cluster_indices = result.indices[mask]
            cluster_features = result.features[mask]

            if labeled_targets is not None and self._is_pure(
                cluster_indices, labeled_targets
            ):
                new_labels[mask] = next_id
                next_id += 1
                continue

            sub_labels = self._bisect(cluster_features)
            if sub_labels is None:
                new_labels[mask] = next_id
                next_id += 1
            else:
                original_positions = np.where(mask)[0]
                for sub in (0, 1):
                    new_labels[original_positions[sub_labels == sub]] = next_id
                    next_id += 1

        centroids = np.stack(
            [
                result.features[new_labels == i].mean(axis=0)
                for i in range(next_id)
            ]
        )
        return ClusterResult(
            features=result.features,
            indices=result.indices,
            labels=new_labels,
            centroids=centroids,
        )

    def _is_pure(
        self, cluster_indices: np.ndarray, labeled_targets: dict[int, int]
    ) -> bool:
        """
        Return True if the labeled members of this cluster are pure enough
        to skip splitting.

        A cluster is considered pure when at least min_labeled_for_purity of
        its members are labeled and the majority class accounts for at least
        purity_threshold of those labeled members.
        """
        labels_in_cluster = [
            labeled_targets[int(idx)]
            for idx in cluster_indices
            if int(idx) in labeled_targets
        ]
        if len(labels_in_cluster) < self._min_labeled_for_purity:
            return False
        counts = np.bincount(labels_in_cluster)
        return (
            float(counts.max() / len(labels_in_cluster))
            >= self._purity_threshold
        )

    def _bisect(self, features: np.ndarray) -> np.ndarray | None:
        """
        Split features into two groups along the first principal axis.

        Returns None if the cluster is too small or has near-zero variance
        along its principal direction (degenerate cluster, cannot be split).

        Arguments
        ---------
        features : np.ndarray, shape (N, D)
            Feature embeddings of a single cluster.

        Returns
        -------
        np.ndarray, shape (N,) with values in {0, 1}, or None.
        """
        if len(features) < self._min_split_size:
            return None

        centered = features - features.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = (centered @ Vt[0]).reshape(-1, 1)

        if projected.std() < 1e-8:
            return None

        return KMeans(
            n_clusters=2,
            random_state=self._seed,
            n_init=10 if HAS_CUML else "auto",
        ).fit_predict(projected)
