from dataclasses import dataclass
import numpy as np
from pipeline.clustering import ClusterResult


@dataclass
class ClusterPurityReport:
    """
    Summary of cluster purity over a ClusterResult.

    Purity of a single cluster is the fraction of its samples that belong
    to the dominant class. An atomic cluster has purity = 1.0 (all samples
    share one class).

    Attributes
    ----------
    mean_purity : float
        Unweighted average purity across all clusters.
    weighted_purity : float
        Purity averaged by cluster size — better reflects overall quality
        when cluster sizes are uneven.
    atomic_cluster_count : int
        Number of clusters whose purity equals 1.0.
    n_clusters : int
        Total number of non-noise clusters evaluated.
    per_cluster_purity : np.ndarray, shape (K,)
        Individual purity value for each cluster (index matches label).
    """

    mean_purity: float
    weighted_purity: float
    atomic_cluster_count: int
    n_clusters: int
    per_cluster_purity: np.ndarray

    def __str__(self) -> str:
        return (
            f"mean={self.mean_purity:.3f} | "
            f"weighted={self.weighted_purity:.3f} | "
            f"atomic={self.atomic_cluster_count}/{self.n_clusters}"
        )


def compute_purity(
    result: ClusterResult, all_targets: np.ndarray
) -> ClusterPurityReport:
    """
    Compute cluster purity using ground-truth labels.

    Each sample in result.indices is looked up in all_targets to get its
    true class. Purity of a cluster is the fraction of samples belonging
    to the majority class within that cluster.

    Arguments
    ---------
    result : ClusterResult
        Fitted cluster result over the pool to evaluate.
    all_targets : np.ndarray, shape (N_dataset,)
        Ground-truth class labels for every sample in the full dataset,
        indexed by dataset index.

    Returns
    -------
    ClusterPurityReport
        Aggregated and per-cluster purity statistics.

    Example
    -------
    >>> report = compute_purity(result, np.array(train_dataset.targets))
    >>> print(report)
    mean=0.412 | weighted=0.398 | atomic=3/100
    """
    cluster_ids = [c for c in np.unique(result.labels) if c != -1]

    if not cluster_ids:
        return ClusterPurityReport(
            mean_purity=0.0,
            weighted_purity=0.0,
            atomic_cluster_count=0,
            n_clusters=0,
            per_cluster_purity=np.array([]),
        )

    per_cluster_purity = np.zeros(len(cluster_ids))
    cluster_sizes = np.zeros(len(cluster_ids), dtype=np.int64)

    for i, c in enumerate(cluster_ids):
        mask = result.labels == c
        cluster_targets = all_targets[result.indices[mask]]
        counts = np.bincount(cluster_targets)
        per_cluster_purity[i] = counts.max() / len(cluster_targets)
        cluster_sizes[i] = len(cluster_targets)

    mean_purity = float(per_cluster_purity.mean())
    weighted_purity = float(
        (per_cluster_purity * cluster_sizes).sum() / cluster_sizes.sum()
    )
    atomic_count = int((per_cluster_purity == 1.0).sum())

    return ClusterPurityReport(
        mean_purity=mean_purity,
        weighted_purity=weighted_purity,
        atomic_cluster_count=atomic_count,
        n_clusters=len(cluster_ids),
        per_cluster_purity=per_cluster_purity,
    )
