import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

from glob_config import SEED


# wrapper for CIFAR100 that also returns the index of each sample. Easier
# for clustering
class IndexedCIFAR100(Dataset):
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self._dataset[idx], idx


class IndexedCIFARSubset(IndexedCIFAR100):
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

        if uniform:
            n_per_class = n_labeled // len(self.classes)
            targets = np.array(self.targets)
            indices = []
            for c in range(len(self.classes)):
                class_indices = np.where(targets == c)[0]
                chosen = rng.choice(
                    class_indices, size=n_per_class, replace=False
                )
                indices.extend(chosen)
            self._indices = np.array(indices)
        else:
            self._indices = rng.choice(
                len(self), size=n_labeled, replace=False
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: IndexedCIFAR100,
        budget: float,
        seed: int = SEED,
        uniform: bool = True,
    ) -> "IndexedCIFARSubset":
        instance = cls.__new__(cls)
        instance._dataset = dataset._dataset
        instance.classes = dataset.classes
        instance.targets = dataset.targets
        n_labeled = int(len(dataset) * budget)
        rng = np.random.default_rng(seed)
        if uniform:
            n_per_class = n_labeled // len(instance.classes)
            targets = np.array(instance.targets)
            indices = []
            for c in range(len(instance.classes)):
                class_indices = np.where(targets == c)[0]
                chosen = rng.choice(
                    class_indices, size=n_per_class, replace=False
                )
                indices.extend(chosen)
            instance._indices = np.array(indices)
        else:
            instance._indices = rng.choice(
                len(dataset), size=n_labeled, replace=False
            )
        return instance

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return super().__getitem__(self._indices[idx])


# Active learning section


class _PoolView(Dataset):
    def __init__(self, base: "IndexedCIFAR100", indices: np.ndarray) -> None:
        self._base = base
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, int], int]:
        return self._base[self._indices[idx]]


class ActiveLearningPool:
    def __init__(
        self,
        train_dataset: IndexedCIFAR100,
        extract_dataset: IndexedCIFAR100,
        initial_budget: float,
        seed: int = SEED,
        uniform: bool = True,
    ) -> None:
        assert len(train_dataset) == len(
            extract_dataset
        ), "N train samples != N extracted samples"
        self._train_ds = train_dataset
        self._extract_ds = extract_dataset

        n_total = len(train_dataset)
        n_labeled = int(n_total * initial_budget)
        rng = np.random.default_rng(seed)

        if uniform:
            targets = np.array(train_dataset.targets)
            n_per_class = n_labeled // len(train_dataset.classes)
            labeled: list[int] = []
            for c in range(len(train_dataset.classes)):
                class_indices = np.where(targets == c)[0]
                labeled.extend(
                    rng.choice(class_indices, size=n_per_class, replace=False)
                )
            self._labeled = np.array(labeled, dtype=np.int64)
        else:
            self._labeled = rng.choice(
                n_total, size=n_labeled, replace=False
            ).astype(np.int64)

        labeled_set = set(self._labeled.tolist())
        self._unlabeled = np.array(
            [i for i in range(n_total) if i not in labeled_set], dtype=np.int64
        )

    @property
    def labeled_dataset(self) -> _PoolView:
        return _PoolView(self._train_ds, self._labeled)

    @property
    def unlabeled_dataset(self) -> _PoolView:
        return _PoolView(self._extract_ds, self._unlabeled)

    def label(self, indices: np.ndarray) -> None:
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
        return len(self._labeled)

    @property
    def n_unlabeled(self) -> int:
        return len(self._unlabeled)
