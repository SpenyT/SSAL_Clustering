import os
from typing import List
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset

from data.dataset_type import IndexedCIFAR100
from data.utils import DATA_DIR, CIFAR_DIR, get_mean_std
from glob_config import NUM_WORKERS, PIN_MEMORY, SEED, seed_worker


def get_transforms(
    mean: List[float], std: List[float]
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and test image transform pipelines.

    The train pipeline applies random augmentations (crop, flip,
    color jitter, rotation) followed by normalization. The test
    pipeline applies only normalization.

    Arguments
    ---------
    mean : List[float]
        Per-channel mean used for normalization, length 3.
    std : List[float]
        Per-channel standard deviation used for normalization, length 3.

    Returns
    -------
    tuple[transforms.Compose, transforms.Compose]
        A (train_transform, test_transform) pair.

    Example
    -------
    >>> mean, std = get_mean_std()
    >>> train_tf, test_tf = get_transforms(mean, std)
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


def get_datasets() -> tuple[CIFAR100, CIFAR100]:
    """
    Load the full CIFAR-100 train and test datasets.

    Computes or loads cached per-channel mean and std, builds the
    appropriate transforms, and returns both splits as standard
    torchvision CIFAR100 datasets.

    Returns
    -------
    tuple[CIFAR100, CIFAR100]
        A (train_dataset, test_dataset) pair.

    Example
    -------
    >>> train_dataset, test_dataset = get_datasets()
    >>> len(train_dataset)
    50000
    """
    mean, std = get_mean_std()
    train_transform, test_transform = get_transforms(mean, std)
    download = not os.path.exists(CIFAR_DIR)

    train_dataset = CIFAR100(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = CIFAR100(
        root=DATA_DIR,
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def create_loader(
    dataset: Dataset, batch_size: int, shuffle: bool
) -> DataLoader:
    """
    Create a DataLoader with the project-standard configuration.

    Arguments
    ---------
    dataset : Dataset
        The dataset to wrap.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle samples each epoch. Use True for training,
        False for evaluation.

    Returns
    -------
    DataLoader
        Configured DataLoader ready for training or evaluation.

    Example
    -------
    >>> loader = make_loader(train_dataset, batch_size=128, shuffle=True)
    >>> loader = make_loader(test_dataset, batch_size=128, shuffle=False)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )


def get_indexed_datasets() -> tuple[IndexedCIFAR100, IndexedCIFAR100]:
    """
    Load the full CIFAR-100 train and test datasets as IndexedCIFAR100.

    Same as get_datasets() but wraps both splits in IndexedCIFAR100 so
    that each sample returns its dataset index alongside the image and
    label. Preferred over get_datasets() for clustering and active
    learning pipelines.

    Returns
    -------
    tuple[IndexedCIFAR100, IndexedCIFAR100]
        A (train_dataset, test_dataset) pair.

    Example
    -------
    >>> train_dataset, test_dataset = get_indexed_datasets()
    >>> (img, label), idx = train_dataset[0]
    """
    mean, std = get_mean_std()
    train_transform, test_transform = get_transforms(mean, std)
    download = not os.path.exists(CIFAR_DIR)

    return (
        IndexedCIFAR100(
            root=DATA_DIR,
            train=True,
            download=download,
            transform=train_transform,
        ),
        IndexedCIFAR100(
            root=DATA_DIR,
            train=False,
            download=download,
            transform=test_transform,
        ),
    )
