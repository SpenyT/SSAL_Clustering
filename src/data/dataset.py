import os
from typing import List
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from data.dataset_type import IndexedCIFAR100
from data.utils import DATA_DIR, CIFAR_DIR, load_data


def get_transforms(mean: List[float], std: List[float]) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_datasets() -> tuple[CIFAR100, CIFAR100]:
    mean, std = load_data(["mean", "std"])
    if mean is None or std is None:
        raise RuntimeError("Mean/std not found — run calculate_save_mean_std() first.")
    else:
        print(f"Successfully loaded [mean, std].")

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

def get_indexed_datasets() -> tuple[IndexedCIFAR100, IndexedCIFAR100]:
    mean, std = load_data(["mean", "std"])
    if mean is None or std is None:
        raise RuntimeError("Mean/std not found — run calculate_save_mean_std() first.")
    else:
        print(f"Successfully loaded [mean, std].")

    train_transform, test_transform = get_transforms(mean, std)
    download = not os.path.exists(CIFAR_DIR)

    return (
        IndexedCIFAR100(root=DATA_DIR, train=True, download=download, transform=train_transform), 
        IndexedCIFAR100(root=DATA_DIR,train=False,download=download,transform=test_transform)
    )