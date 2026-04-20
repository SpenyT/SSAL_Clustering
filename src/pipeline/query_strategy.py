from abc import ABC, abstractmethod

import numpy as np

from glob_config import HAS_CUML, SEED
from pipeline.clustering import ClusterResult

if HAS_CUML:
    from cuml.cluster import KMeans as _KMeans  # type: ignore
else:
    from sklearn.cluster import KMeans as _KMeans


class QueryStrategy(ABC):
    """
    Abstract base class for active learning query strategies.

    A query strategy selects which unlabeled samples to send to the oracle
    for annotation. Concrete strategies implement different notions of
    sample informativeness.

    Example
    -------
    >>> strategy = MixedQueryStrategy()
    >>> candidates = strategy.select(result, n=500)
    """

    @abstractmethod
    def select(
        self,
        result: ClusterResult,
        n: int,
        labeled_targets: dict[int, int] | None = None,
    ) -> np.ndarray:
        """
        Select n annotation candidates from a ClusterResult.

        Arguments
        ---------
        result : ClusterResult
            Fitted cluster result over the current pool. May contain both
            labeled and unlabeled samples when called from the SSALC pipeline;
            strategies that are purity-aware use labeled_targets to distinguish
            them and restrict selection to unlabeled indices only.
        n : int
            Number of candidates to select.
        labeled_targets : dict[int, int] | None
            Mapping from dataset index to ground-truth class for every
            currently labeled sample. Strategies that do not need purity
            information may ignore this argument.

        Returns
        -------
        np.ndarray, shape (≤n,)
            Dataset indices of selected candidates.
        """


class RandomStrategy(QueryStrategy):
    """
    Selects samples uniformly at random, excluding noise points.

    Maintains a stateful RNG so successive calls within the same AL loop
    produce different, non-overlapping random draws.

    Arguments
    ---------
    seed : int
        Seed for the internal RNG. Default: SEED.

    Example
    -------
    >>> strategy = RandomStrategy(seed=42)
    >>> candidates = strategy.select(result, n=500)
    """

    def __init__(self, seed: int = SEED) -> None:
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        result: ClusterResult,
        n: int,
        _labeled_targets: dict[int, int] | None = None,
    ) -> np.ndarray:
        indices = result.indices[result.labels != -1]
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        n = min(n, len(indices))
        return self._rng.choice(indices, size=n, replace=False).astype(
            np.int64
        )


class ClusterBoundaryStrategy(QueryStrategy):
    """
    Selects unlabeled samples near the class boundary inside impure clusters.

    For each cluster whose labeled members do not yet reach "purity_threshold",
    a local KMeans(k=2) split is performed on all cluster members to find the
    two sub-population centroids. The midpoint between those centroids is the
    estimated class boundary. Unlabeled samples nearest to that midpoint are
    the most ambiguous between the two sub-classes and are therefore the most
    informative to label next.

    Clusters with fewer than "min_labeled_for_purity" labeled members are
    treated as impure (unknown purity) and always included as candidates.
    Clusters that have already reached the purity threshold are skipped.

    Arguments
    ---------
    purity_threshold : float
        Clusters whose labeled members have majority-class fraction >= this
        value are skipped. Default: 0.85.
    min_labeled_for_purity : int
        Minimum labeled members required before the purity check is trusted.
        Clusters below this count are always treated as impure. Default: 3.
    seed : int
        Random seed for the local KMeans(k=2). Default: SEED.

    Example
    -------
    >>> strategy = ClusterBoundaryStrategy()
    >>> candidates = strategy.select(
    ...     result,
    ...     n=350,
    ...     labeled_targets={0: 3, 1: 7}
    ... )
    """

    def __init__(
        self,
        purity_threshold: float = 0.85,
        min_labeled_for_purity: int = 3,
        seed: int = SEED,
    ) -> None:
        self._purity_threshold = purity_threshold
        self._min_labeled_for_purity = min_labeled_for_purity
        self._seed = seed

    def select(
        self,
        result: ClusterResult,
        n: int,
        labeled_targets: dict[int, int] | None = None,
    ) -> np.ndarray:
        cluster_ids = [c for c in np.unique(result.labels) if c != -1]
        if not cluster_ids:
            return np.array([], dtype=np.int64)

        labeled_set = set(labeled_targets.keys()) if labeled_targets else set()
        candidates: list[tuple[int, float]] = []

        for c in cluster_ids:
            mask = result.labels == c
            cluster_indices = result.indices[mask]
            cluster_features = result.features[mask]

            if labeled_targets is not None and self._is_pure(
                cluster_indices, labeled_targets
            ):
                continue

            # Local KMeans(k=2) on all cluster members (labeled + unlabeled)
            # gives the two sub-population centroids; their midpoint is the
            # estimated class boundary within this cluster.
            if len(cluster_features) >= 4:
                km = _KMeans(n_clusters=2, n_init=3, random_state=self._seed)
                km.fit(cluster_features)
                midpoint = km.cluster_centers_.mean(axis=0)
            else:
                midpoint = cluster_features.mean(axis=0)

            for idx, feat in zip(cluster_indices, cluster_features):
                if int(idx) not in labeled_set:
                    d = float(np.linalg.norm(feat - midpoint))
                    candidates.append((int(idx), d))

        if not candidates:
            return np.array([], dtype=np.int64)

        candidates.sort(key=lambda x: x[1])

        seen: set[int] = set()
        selected: list[int] = []
        for idx, _ in candidates:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
            if len(selected) >= n:
                break

        return np.array(selected, dtype=np.int64)

    def _is_pure(
        self, cluster_indices: np.ndarray, labeled_targets: dict[int, int]
    ) -> bool:
        labeled = [
            labeled_targets[int(idx)]
            for idx in cluster_indices
            if int(idx) in labeled_targets
        ]
        if len(labeled) < self._min_labeled_for_purity:
            return False
        counts = np.bincount(labeled)
        return float(counts.max() / len(labeled)) >= self._purity_threshold


class MixedQueryStrategy(QueryStrategy):
    """
    Combines cluster-boundary sampling with random sampling.

    Spends "boundary_ratio" of the budget on unlabeled samples nearest to
    the class boundary inside impure clusters (via ClusterBoundaryStrategy),
    then fills the remainder with random samples drawn from what is left —
    avoiding duplicates and already-labeled indices.

    This matches my proposal: query hard boundary samples for the biggest
    gain in cluster purity, plus random samples to maintain diversity.

    Arguments
    ---------
    boundary_ratio : float
        Fraction of budget spent on cluster-boundary samples. Default: 0.7.
    seed : int
        Seed for the random component. Default: SEED.

    Example
    -------
    >>> strategy = MixedQueryStrategy()
    >>> candidates = strategy.select(result, n=500, labeled_targets={...})
    >>> len(candidates)
    500
    """

    def __init__(self, boundary_ratio: float = 0.7, seed: int = SEED) -> None:
        self._boundary = ClusterBoundaryStrategy(seed=seed)
        self._random = RandomStrategy(seed=seed)
        self._boundary_ratio = boundary_ratio

    def select(
        self,
        result: ClusterResult,
        n: int,
        labeled_targets: dict[int, int] | None = None,
    ) -> np.ndarray:
        n_boundary = round(n * self._boundary_ratio)
        n_random = n - n_boundary

        boundary = self._boundary.select(result, n_boundary, labeled_targets)

        # I exclude already-selected and already-labeled indices
        labeled_set = set(labeled_targets.keys()) if labeled_targets else set()
        exclude = set(boundary.tolist()) | labeled_set
        remaining = result.filter(~np.isin(result.indices, list(exclude)))
        random_part = self._random.select(remaining, n_random)

        return np.concatenate([boundary, random_part]).astype(np.int64)
