import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import glob_config
from data.dataset import get_transforms
from data.dataset_type import (
    ActiveLearningPool,
    IndexedCIFAR100,
    PseudoLabeledView,
)
from data.data_utils import CIFAR_DIR, DATA_DIR, get_mean_std
from model.checkpoint import checkpoint_path
from model.feature_extractor import (
    FeatureExtractor,
    PCAExtractor,
    ResnetExtractor,
)
from model.model_utils import prepare_model, try_compile
from model.resnet import load_resnet18
from pipeline.cluster_purity import ClusterPurityReport, compute_purity
from pipeline.clustering import (
    ClusterResult,
    Clusterer,
    HierarchicalKMeansClusterer,
)
from pipeline.query_strategy import MixedQueryStrategy, QueryStrategy
from model.model_utils import ModelName
from pipeline.resnet18_baseline import training_loop

# NOTE: I used tune.py (GridSearch) for best startegies and hyperparams
_N_AL_ROUNDS = 5 # number of active learning query rounds
_INITIAL_BUDGET_FRACTION = 0.2 # fraction of total budget used as the initial labeled seed
_PCA_COMPONENTS = 128 # PCA output dimensionality after ResNet feature extraction
_N_CLUSTERS = 500 # initial K-Means cluster count for HierarchicalKMeansClusterer
_SPLIT_DEPTH = 4 # number of bisection passes applied after initial clustering
_PSEUDO_LABEL_THRESHOLD = 0.7 # min majority-class fraction for a cluster to receive pseudo-labels
_MIN_LABELED_FOR_PSEUDO = 3 # min labeled members a cluster needs before pseudo-labeling applies


def _build_pseudo_labels(
    result: ClusterResult,
    all_targets: np.ndarray,
    labeled_set: set[int],
) -> dict[int, int]:
    """
    Assign pseudo-labels to unlabeled samples in sufficiently pure clusters.

    For each cluster where labeled members share a majority class with
    fraction >= _PSEUDO_LABEL_THRESHOLD, every unlabeled member receives
    that majority class as its pseudo-label.

    Arguments
    ---------
    result : ClusterResult
        Final clustering result over the full training pool.
    all_targets : np.ndarray, shape (N,)
        Ground-truth class labels for every sample in the training dataset.
    labeled_set : set[int]
        Dataset indices of currently labeled samples.

    Returns
    -------
    dict[int, int]
        Mapping from dataset index to pseudo-class label for unlabeled
        samples whose cluster is sufficiently pure.
    """
    pseudo: dict[int, int] = {}
    for c in np.unique(result.labels):
        if c == -1:
            continue
        mask = result.labels == c
        cluster_indices = result.indices[mask]
        labeled_classes = [
            int(all_targets[idx])
            for idx in cluster_indices
            if int(idx) in labeled_set
        ]
        if len(labeled_classes) < _MIN_LABELED_FOR_PSEUDO:
            continue
        counts = np.bincount(labeled_classes)
        if (
            float(counts.max() / len(labeled_classes))
            >= _PSEUDO_LABEL_THRESHOLD
        ):
            majority = int(counts.argmax())
            for idx in cluster_indices:
                if int(idx) not in labeled_set:
                    pseudo[int(idx)] = majority
    return pseudo


def _make_train_loader(dataset, batch_size: int) -> DataLoader:
    """
    Wrap a dataset in a reproducible, shuffled DataLoader for training.

    Arguments
    ---------
    dataset : Dataset
        Training dataset to wrap (may be a ConcatDataset).
    batch_size : int
        Number of samples per batch.

    Returns
    -------
    DataLoader
        Shuffled DataLoader with pinned memory and persistent workers
        configured from glob_config.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=glob_config.NUM_WORKERS,
        pin_memory=glob_config.PIN_MEMORY,
        persistent_workers=glob_config.NUM_WORKERS > 0,
        prefetch_factor=2 if glob_config.NUM_WORKERS > 0 else None,
        worker_init_fn=glob_config.seed_worker,
        generator=torch.Generator().manual_seed(glob_config.SEED),
    )


def _al_round(
    pool: ActiveLearningPool,
    extractor: PCAExtractor,
    clusterer: Clusterer,
    query_strategy: QueryStrategy,
    n_query: int,
    batch_size: int,
    all_targets: np.ndarray,
) -> ClusterPurityReport:
    """
    Execute one active learning round.

    Fits the extractor on the unlabeled pool, then extracts features from
    both unlabeled and labeled samples. Clustering uses all features so the
    HierarchicalKMeansClusterer can apply purity-guided stopping based on
    known labels in each cluster. The query strategy only sees unlabeled
    samples to avoid re-querying already-labeled ones.

    Returns the purity report computed before labeling, so it reflects
    the quality of the current clustering before the oracle is queried.

    Arguments
    ---------
    pool : ActiveLearningPool
        Current labeled/unlabeled split; updated in-place via pool.label().
    extractor : PCAExtractor
        Feature extractor; re-fit on the unlabeled pool each round.
    clusterer : Clusterer
        Clustering algorithm applied to the full combined feature matrix.
    query_strategy : QueryStrategy
        Strategy used to select n_query candidates from the cluster result.
    n_query : int
        Number of samples to label this round.
    batch_size : int
        Batch size for feature extraction DataLoaders.
    all_targets : np.ndarray, shape (N,)
        Ground-truth labels for the full training dataset, used to compute
        purity and build the labeled_targets dict passed to the clusterer.

    Returns
    -------
    ClusterPurityReport
        Purity metrics computed after clustering but before labeling.
    """
    extractor.fit(pool.unlabeled_dataset, batch_size)

    unlabeled_features, unlabeled_indices, _ = extractor.extract(
        pool.unlabeled_dataset, batch_size
    )
    labeled_features, labeled_indices, _ = extractor.extract(
        pool.labeled_dataset, batch_size
    )

    all_features = np.concatenate([unlabeled_features, labeled_features])
    all_indices = np.concatenate([unlabeled_indices, labeled_indices])
    labeled_targets = {
        int(idx): int(all_targets[idx]) for idx in labeled_indices
    }

    result = clusterer.fit(
        all_features, all_indices, labeled_targets=labeled_targets
    )
    purity = compute_purity(result, all_targets)


    # My strategies filter to unlabeled candidates internally via
    # labeled_targets.
    selected = query_strategy.select(
        result, n_query, labeled_targets=labeled_targets
    )
    pool.label(selected)
    return purity


def run_ssalc(
    train_dataset: IndexedCIFAR100,
    test_loader: DataLoader,
    budget: float,
    model_name: ModelName = "SSALC",
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
    extractor: FeatureExtractor | None = None,
    clusterer: Clusterer | None = None,
    query_strategy: QueryStrategy | None = None,
) -> None:
    """
    Run the full SSALC pipeline: active learning → pseudo-labeling → training.

    Initializes the labeled/unlabeled pool, runs _N_AL_ROUNDS of cluster-guided
    active learning to select informative samples, performs a final clustering
    pass to assign pseudo-labels to high-confidence unlabeled samples, then
    trains a ResNet-18 on the combined labeled + pseudo-labeled set.

    Arguments
    ---------
    train_dataset : IndexedCIFAR100
        Full training dataset with augmentation transforms.
    test_loader : DataLoader
        Test DataLoader returning ((imgs, labels), idx) batches.
    budget : float
        Annotation budget as a fraction of the training set, in (0, 1].
    model_name : str
        Name used for checkpointing and results logging. Default: "SSALC".
    epochs : int
        Number of training epochs. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    batch_size : int
        Batch size for both feature extraction and training. Default: 128.
    extractor : FeatureExtractor | None
        Feature extractor to use. Defaults to PCAExtractor(ResnetExtractor(),
        n_components=_PCA_COMPONENTS) if None.
    clusterer : Clusterer | None
        Clustering algorithm. Defaults to HierarchicalKMeansClusterer with
        pipeline defaults if None.
    query_strategy : QueryStrategy | None
        Active learning query strategy. Defaults to MixedQueryStrategy if None.

    Returns
    -------
    None
        Results are written to the log via ResultsLogger and the trained
        model checkpoint is saved via checkpoint_path.
    """
    print(f"\n-- {model_name} --")

    if clusterer is None:
        clusterer = HierarchicalKMeansClusterer(
            n_clusters=_N_CLUSTERS,
            split_depth=_SPLIT_DEPTH,
            seed=glob_config.SEED,
        )
    if query_strategy is None:
        query_strategy = MixedQueryStrategy(seed=glob_config.SEED)

    all_targets = np.array(train_dataset.targets)

    mean, std = get_mean_std()
    _, test_transform = get_transforms(mean, std)
    extract_dataset = IndexedCIFAR100(
        root=DATA_DIR,
        train=True,
        download=not os.path.exists(CIFAR_DIR),
        transform=test_transform,
    )

    n_total = len(train_dataset)
    total_labeled = max(100, int(budget * n_total))
    n_initial = max(100, int(_INITIAL_BUDGET_FRACTION * total_labeled))

    pool = ActiveLearningPool(
        train_dataset=train_dataset,
        extract_dataset=extract_dataset,
        initial_budget=n_initial / n_total,
        seed=glob_config.SEED,
        uniform=True,
    )

    if extractor is None:
        extractor = PCAExtractor(
            ResnetExtractor(), n_components=_PCA_COMPONENTS
        ).to(glob_config.DEVICE)

    n_remaining = total_labeled - pool.n_labeled
    n_per_round, remainder = divmod(n_remaining, _N_AL_ROUNDS)

    print(f"  Initial labeled: {pool.n_labeled} | Target: {total_labeled}")
    t_al = time.perf_counter()

    for r in range(_N_AL_ROUNDS):
        if pool.n_unlabeled == 0:
            break
        n_query = min(
            n_per_round + (1 if r < remainder else 0), pool.n_unlabeled
        )
        if n_query <= 0:
            continue
        print(f"  AL round {
            r + 1}/{_N_AL_ROUNDS} | querying {n_query} samples")
        purity = _al_round(
            pool,
            extractor,
            clusterer,
            query_strategy,
            n_query,
            batch_size,
            all_targets,
        )
        print(f"  Purity: {purity} | labeled: {
            pool.n_labeled}/{total_labeled}")

    t_al = time.perf_counter() - t_al
    print(f"  AL done in {t_al:.1f}s | final labeled: {pool.n_labeled}")

    # I use my pseudo-lables to refine my low-confidence (impure) clusters her
    unlabeled_features, unlabeled_indices, _ = extractor.extract(
        pool.unlabeled_dataset, batch_size
    )
    labeled_features, labeled_indices, _ = extractor.extract(
        pool.labeled_dataset, batch_size
    )
    all_features = np.concatenate([unlabeled_features, labeled_features])
    all_indices = np.concatenate([unlabeled_indices, labeled_indices])
    labeled_targets_dict = {
        int(i): int(all_targets[i]) for i in labeled_indices
    }
    final_result = clusterer.fit(
        all_features, all_indices, labeled_targets=labeled_targets_dict
    )

    pseudo_labels = _build_pseudo_labels(
        final_result, all_targets, set(labeled_indices.tolist())
    )
    print(
        f"  Pseudo-labeled: {len(pseudo_labels)} unlabeled samples "
        f"({len(pseudo_labels) / pool.n_unlabeled:.1%} of unlabeled pool)"
    )

    pseudo_view = PseudoLabeledView(train_dataset, pseudo_labels)
    train_dataset_combined = ConcatDataset([pool.labeled_dataset, pseudo_view])

    model = try_compile(
        prepare_model(load_resnet18(with_pretrained_weights=False))
    )
    train_loader = _make_train_loader(train_dataset_combined, batch_size)

    training_loop(
        model,
        train_loader,
        test_loader,
        checkpoint_path(model_name, budget),
        model_name,
        budget,
        epochs,
        lr,
    )
