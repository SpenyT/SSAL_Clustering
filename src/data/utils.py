import os
import pickle
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from glob_config import VARIABLES_PATH, DATA_DIR, CIFAR_DIR

# taken from https://forums.fast.ai/t/normalizing-your-dataset/49799
def calculate_mean_std(loader: DataLoader) -> tuple[list[float], list[float]]:
    n = 0
    s1 = torch.zeros(3)
    s2 = torch.zeros(3)
    for x, _ in loader:
        B, _, H, W = x.shape
        n += B * H * W
        s1 += x.sum(dim=(0,2,3))
        s2 += (x*x).sum(dim=(0,2,3))
    mean = s1 / n
    std = (s2 / n - mean**2).sqrt()
    return mean.tolist(), std.tolist()


def load_data(data_name: str | list[str] | None = None) -> Any:
    if not os.path.exists(VARIABLES_PATH):
        return {}
    try:
        with open(VARIABLES_PATH, 'rb') as f:
            all_data: dict = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return {}

    if data_name is None:
        return all_data
    if isinstance(data_name, str):
        return all_data.get(data_name)
    return [all_data.get(name) for name in data_name]


def save_data(new_data: dict[str, Any]) -> None:
    all_data = load_data()
    new_dict = all_data | new_data
    with open(VARIABLES_PATH, 'wb') as f:
        pickle.dump(new_dict, f)


def calculate_save_mean_std() -> None:
    stats_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    stats_dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=not os.path.exists(CIFAR_DIR),
        transform=stats_transform
    )

    loader = DataLoader(stats_dataset, batch_size=256, shuffle=False)
    mean, std = calculate_mean_std(loader)
    save_data({"mean": mean, "std": std})


def unnormalize(img: Tensor, mean: list[float] | None = None, std: list[float] | None = None) -> Tensor:
    if mean is None:
        mean = load_data("mean")
    if std is None:
        std = load_data("std")

    mean_t = torch.tensor(mean).view(3,1,1)
    std_t = torch.tensor(std).view(3,1,1)
    return img * std_t + mean_t