from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
        self.classes: List[str] = self._dataset.classes
        self.targets: List[int] = self._dataset.targets

    def __len__(self) -> int: 
        return len(self._dataset)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self._dataset[idx], idx # returns (image, label, index)
