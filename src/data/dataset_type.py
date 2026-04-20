import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

from glob_config import SEED


def _sample_indices(
    targets: np.ndarray,
    classes: list,
    n_labeled: int,
    rng: np.random.Generator,
    uniform: bool,
) -> np.ndarray:
    """
    Sample dataset indices for a labeled subset.

    If uniform, samples an equal number of indices per class.
    Otherwise, samples randomly across the full dataset.

    Arguments
    ---------
    targets : np.ndarray
        Class label for each sample in the dataset, shape (N,).
    classes : list
        List of class names, used to determine the number of classes.
    n_labeled : int
        Total number of indices to sample.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    uniform : bool
        If True, samples floor(n_labeled / n_classes) indices per class.
        If False, samples n_labeled indices at random.

    Returns
    -------
    np.ndarray
        Sampled indices of shape (n_labeled,).

    Example
    -------
    >>> rng = np.random.default_rng(42)
    >>> indices = _sample_indices(
    ...     targets, classes, n_labeled=500, rng=rng, uniform=True
    ... )
    >>> indices.shape
    (500,)
    """
    if uniform:
        n_per_class = n_labeled // len(classes)
        indices: list[int] = []
        for c in range(len(classes)):
            class_indices = np.where(targets == c)[0]
            indices.extend(
                rng.choice(class_indices, size=n_per_class, replace=False)
            )
        return np.array(indices)
    return rng.choice(len(targets), size=n_labeled, replace=False)


class IndexedCIFAR100(Dataset):
    """
    CIFAR-100 dataset wrapper that also returns the sample index.

    Wraps torchvision's CIFAR100 so that each call to __getitem__
    returns the sample alongside its dataset index. This makes it
    easier to track individual samples during clustering and active
    learning.

    Arguments
    ---------
    root : str
        Path to the directory where the dataset is stored or downloaded.
    train : bool
        If True, loads the training split. If False, loads the test split.
    transform : transforms.Compose
        Transforms applied to each image before returning.
    download : bool
        If True, downloads the dataset if not already present. Default: True.

    Example
    -------
    >>> dataset = IndexedCIFAR100(
    ...     root="data/", train=True, transform=transform
    ... )
    >>> (img, label), idx = dataset[0]
    """

    def __init__(
        self,
        root: str,
        train: bool,
        transform: transforms.Compose,
        download: bool = True,
    ) -> None:
        self._dataset = datasets.CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.classes: list[str] = self._dataset.classes
        self.targets: list[int] = self._dataset.targets

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, int], int]:
        return self._dataset[idx], idx


class IndexedCIFARSubset(IndexedCIFAR100):
    """
    A labeled subset of IndexedCIFAR100 sampled by annotation budget.

    Selects a fixed subset of the training data at construction time,
    either uniformly across classes or randomly. Used to simulate
    annotation budgets in baseline experiments.

    Arguments
    ---------
    root : str
        Path to the directory where the dataset is stored or downloaded.
    train : bool
        If True, loads the training split. If False, loads the test split.
    transform : transforms.Compose
        Transforms applied to each image before returning.
    budget : float
        Fraction of the dataset to label, in range (0, 1].
    seed : int
        Random seed for reproducibility. Default: SEED.
    uniform : bool
        If True, samples floor(n_labeled / n_classes) per class.
        If False, samples randomly across the full dataset. Default: True.
    download : bool
        If True, downloads the dataset if not already present. Default: True.

    Example
    -------
    >>> subset = IndexedCIFARSubset(
    ...     root="data/", train=True, transform=transform, budget=0.1
    ... )
    >>> len(subset)  # 10% of 50000
    5000
    """

    def __init__(
        self,
        root: str,
        train: bool,
        transform: transforms.Compose,
        budget: float,
        seed: int = SEED,
        uniform: bool = True,
        download: bool = True,
    ) -> None:
        super().__init__(root, train, transform, download)
        n_labeled = int(super().__len__() * budget)
        rng = np.random.default_rng(seed)

        self._indices = _sample_indices(
            np.array(self.targets), self.classes, n_labeled, rng, uniform
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: IndexedCIFAR100,
        budget: float,
        seed: int = SEED,
        uniform: bool = True,
    ) -> "IndexedCIFARSubset":
        """
        Create an IndexedCIFARSubset from an existing IndexedCIFAR100 instance.

        Avoids re-loading the dataset from disk by reusing an already
        constructed IndexedCIFAR100. Preferred over __init__ when the
        full dataset is already in memory.

        Arguments
        ---------
        dataset : IndexedCIFAR100
            The full dataset to sample from.
        budget : float
            Fraction of the dataset to label, in range (0, 1].
        seed : int
            Random seed for reproducibility. Default: SEED.
        uniform : bool
            If True, samples floor(n_labeled / n_classes) per class.
            If False, samples randomly across the full dataset. Default: True.

        Returns
        -------
        IndexedCIFARSubset
            A new subset instance backed by the provided dataset.

        Example
        -------
        >>> full = IndexedCIFAR100(
        ...     root="data/", train=True, transform=transform
        ... )
        >>> subset = IndexedCIFARSubset.from_dataset(full, budget=0.1)
        >>> len(subset)
        5000
        """
        instance = cls.__new__(cls)
        instance._dataset = dataset._dataset
        instance.classes = dataset.classes
        instance.targets = dataset.targets
        n_labeled = int(len(dataset) * budget)
        rng = np.random.default_rng(seed)
        instance._indices = _sample_indices(
            np.array(instance.targets),
            instance.classes,
            n_labeled,
            rng,
            uniform,
        )
        return instance

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return super().__getitem__(self._indices[idx])


######################################
# Active learning section
######################################
class _PoolView(Dataset):
    """
    A view over a subset of an IndexedCIFAR100 dataset.

    A helper that presents a subset of a dataset defined by an index array, without
    copying any underlying data. Used internally by ActiveLearningPool
    to expose the labeled and unlabeled splits as standard Dataset objects.
    Not intended to be instantiated directly. Mostly for memory efficiency.

    Arguments
    ---------
    base : IndexedCIFAR100
        The full dataset to index into.
    indices : np.ndarray
        Array of indices defining which samples are visible in this view.

    Example
    -------
    >>> view = _PoolView(dataset, indices=np.array([0, 5, 12]))
    >>> len(view)
    3
    """

    def __init__(self, base: "IndexedCIFAR100", indices: np.ndarray) -> None:
        self._base = base
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, int], int]:
        return self._base[self._indices[idx]]


class ActiveLearningPool:
    """
    Manages the labeled/unlabeled split for an active learning loop.

    Maintains two dynamic index sets (labeled and unlabeled) over a
    training dataset. The split evolves as the active learning loop
    queries samples via label(). Exposes each split as a Dataset view
    for use with DataLoader.

    Reference
    ---------
    - Overview:
        https://odsc.medium.com/crash-course-pool-based-sampling-in-active-learning-cb40e30d49df
    - Active Learning:
        https://burrsettles.com/pub/settles.activelearning.pdf
    - Implementation:
        https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html

    Arguments
    ---------
    train_dataset : IndexedCIFAR100
        Dataset used for training (with augmentation transforms).
    extract_dataset : IndexedCIFAR100
        Dataset used for feature extraction (without augmentation).
        Must be the same size as train_dataset.
    initial_budget : float
        Fraction of the dataset to label initially, in range (0, 1].
    seed : int
        Random seed for reproducibility. Default: SEED.
    uniform : bool
        If True, samples floor(n_labeled / n_classes) per class for
        the initial labeled set. If False, samples randomly. Default: True.

    Example
    -------
    >>> pool = ActiveLearningPool(train_ds, extract_ds, initial_budget=0.05)
    >>> pool.label(np.array([42, 137, 256]))
    >>> pool.n_labeled
    2503
    """

    def __init__(
        self,
        train_dataset: IndexedCIFAR100,
        extract_dataset: IndexedCIFAR100,
        initial_budget: float,
        seed: int = SEED,
        uniform: bool = True,
    ) -> None:
        self._train_ds = train_dataset
        self._extract_ds = extract_dataset

        n_total = len(train_dataset)
        n_labeled = int(n_total * initial_budget)
        rng = np.random.default_rng(seed)

        self._labeled = _sample_indices(
            np.array(train_dataset.targets),
            train_dataset.classes,
            n_labeled,
            rng,
            uniform,
        ).astype(np.int64)

        labeled_set = set(self._labeled.tolist())
        self._unlabeled = np.array(
            [i for i in range(n_total) if i not in labeled_set], dtype=np.int64
        )

    @property
    def labeled_dataset(self) -> _PoolView:
        """
        View of the current labeled split, backed by train_dataset.

        Returns
        -------
        _PoolView
            A Dataset view of the labeled split, using train_dataset for
            data access (with augmentation).

        Example
        -------
        >>> pool = ActiveLearningPool(
        ...     train_ds, extract_ds, initial_budget=0.05
        ... )
        >>> len(pool.labeled_dataset)
        2500
        """
        return _PoolView(self._train_ds, self._labeled)

    @property
    def unlabeled_dataset(self) -> _PoolView:
        """
        View of the current unlabeled split, backed by extract_dataset.

        Returns
        -------
        _PoolView
            A Dataset view of the unlabeled split, using extract_dataset
            for data access (without augmentation).

        Example
        -------
        >>> pool = ActiveLearningPool(
        ...     train_ds, extract_ds, initial_budget=0.05
        ... )
        >>> len(pool.unlabeled_dataset)
        47500
        """
        return _PoolView(self._extract_ds, self._unlabeled)

    def label(self, indices: np.ndarray) -> None:
        """
        Move samples from the unlabeled pool into the labeled set.

        Indices already in the labeled set are silently ignored.

        Arguments
        ---------
        indices : np.ndarray
            Dataset indices to label. Values not in the unlabeled pool
            are ignored.

        Example
        -------
        >>> pool = ActiveLearningPool(
        ...     train_ds, extract_ds, initial_budget=0.05
        ... )
        >>> pool.label(np.array([42, 137, 256]))
        >>> pool.n_labeled
        2503
        """
        indices = np.asarray(indices, dtype=np.int64)
        to_add = indices[np.isin(indices, self._unlabeled)]
        if len(to_add) == 0:
            return
        self._labeled = np.concatenate([self._labeled, to_add])
        self._unlabeled = np.array(
            sorted(set(self._unlabeled.tolist()) - set(to_add.tolist())),
            dtype=np.int64,
        )

    @property
    def n_labeled(self) -> int:
        """
        Number of currently labeled samples.

        Returns
        -------
        int
            Number of samples currently in the labeled pool.

        Example
        -------
        >>> pool = ActiveLearningPool(
        ...     train_ds, extract_ds, initial_budget=0.05
        ... )
        >>> pool.n_labeled
        2500
        """
        return len(self._labeled)

    @property
    def n_unlabeled(self) -> int:
        """
        Number of currently unlabeled samples.

        Returns
        -------
        int
            Number of samples currently in the unlabeled pool.

        Example
        -------
        >>> pool = ActiveLearningPool(
        ...     train_ds, extract_ds, initial_budget=0.05
        ... )
        >>> pool.n_unlabeled
        47500
        """
        return len(self._unlabeled)


class PseudoLabeledView(Dataset):
    """
    A dataset view over unlabeled samples with cluster-assigned pseudo-labels.

    Wraps the training dataset (with augmentation) and overrides the label
    for each sample with its assigned pseudo-label. Used to expand the
    training set with high-confidence unlabeled samples after the AL loop.

    Reference
    ---------
    - Pseudo-Label pipeline:
        https://medium.com/@i2vsys/recursive-clustering-based-pseudo-label-correction-pipeline-ec700d5643fe

    Arguments
    ---------
    train_dataset : IndexedCIFAR100
        Full training dataset (with augmentation transforms).
    pseudo_labels : dict[int, int]
        Mapping from dataset index to pseudo-class label.

    Example
    -------
    >>> view = PseudoLabeledView(train_ds, {42: 7, 137: 3})
    >>> len(view)
    2
    """

    def __init__(
        self,
        train_dataset: "IndexedCIFAR100",
        pseudo_labels: dict[int, int],
    ) -> None:
        self._base = train_dataset
        self._pseudo_labels = pseudo_labels
        self._indices = np.array(list(pseudo_labels.keys()), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, pos: int) -> tuple[tuple[torch.Tensor, int], int]:
        dataset_idx = int(self._indices[pos])
        (image, _), orig_idx = self._base[dataset_idx]
        return (image, self._pseudo_labels[dataset_idx]), orig_idx
