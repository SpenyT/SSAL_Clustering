import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

from glob_config import SEED


# wrapper for CIFAR100 that also returns the index of each sample. Easier for clustering
class IndexedCIFAR100(Dataset):
    def __init__( 
        self,
        root: str,
        train: bool,
        transform: transforms.Compose,
        download: bool = True
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
        return self._dataset[idx], idx # returns (image, label, index)
    

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
        n_labeled = int(len(self) * budget)
        rng = np.random.default_rng(seed)

        if uniform:
            n_per_class = n_labeled // len(self.classes)
            targets = np.array(self.targets)
            indices = []
            for c in range(len(self.classes)):
                class_indices = np.where(targets == c)[0]
                chosen = rng.choice(class_indices, size=n_per_class, replace=False)
                indices.extend(chosen)
            self._indices = np.array(indices)
        else:
            self._indices = rng.choice(len(self), size=n_labeled, replace=False)

    def __len__(self) -> int: return len(self._indices)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]: return super().__getitem__(self._indices[idx])