import os
import pickle
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from glob_config import VARIABLES_PATH, DATA_DIR, CIFAR_DIR


def calculate_mean_std(loader: DataLoader) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and standard deviation over a dataset.

    Uses a single-pass online algorithm to accumulate pixel sums and
    squared sums, avoiding the need to load the full dataset into memory.

    References
    ----------
    Taken from: https://forums.fast.ai/t/normalizing-your-dataset/49799

    Arguments
    ---------
    loader : DataLoader
        DataLoader over the dataset to compute statistics for.
        Images must be in (B, C, H, W) float format.

    Returns
    -------
    tuple[list[float], list[float]]
        A (mean, std) pair, each a list of 3 floats (one per channel).

    Example
    -------
    >>> mean, std = calculate_mean_std(loader)
    >>> mean
    [0.5071, 0.4867, 0.4408]
    """
    n = 0
    s1 = torch.zeros(3)
    s2 = torch.zeros(3)
    for x, _ in loader:
        B, _, H, W = x.shape
        n += B * H * W
        s1 += x.sum(dim=(0, 2, 3))
        s2 += (x * x).sum(dim=(0, 2, 3))
    mean = s1 / n
    std = (s2 / n - mean**2).sqrt()
    return mean.tolist(), std.tolist()


def load_data(data_name: str | list[str] | None = None) -> Any:
    """
    Load saved variables from the project pickle cache.

    If data_name is None, returns the full cache dict. If a string,
    returns the value for that key. If a list, returns a list of
    values in the same order.

    Arguments
    ---------
    data_name : str | list[str] | None
        Key(s) to retrieve. If None, returns the entire cache.

    Returns
    -------
    Any
        The cached value(s), or an empty dict if the cache does not exist
        or is corrupt.

    Example
    -------
    >>> mean = load_data("mean")
    >>> mean, std = load_data(["mean", "std"])
    """
    if not os.path.exists(VARIABLES_PATH):
        return {}
    try:
        with open(VARIABLES_PATH, "rb") as f:
            all_data: dict = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return {}

    if data_name is None:
        return all_data
    if isinstance(data_name, str):
        return all_data.get(data_name)
    return [all_data.get(name) for name in data_name]


def save_data(new_data: dict[str, Any]) -> None:
    """
    Persist new key-value pairs to the project pickle cache.

    Merges new_data into the existing cache, overwriting any keys
    that already exist.

    Arguments
    ---------
    new_data : dict[str, Any]
        Key-value pairs to save.

    Example
    -------
    >>> save_data({
    ...     "mean": [0.507, 0.487, 0.441],
    ...     "std": [0.267, 0.256, 0.276]
    ... })
    """
    all_data = load_data()
    new_dict = all_data | new_data
    with open(VARIABLES_PATH, "wb") as f:
        pickle.dump(new_dict, f)


def calculate_save_mean_std() -> tuple[list[float], list[float]]:
    """
    Compute CIFAR-100 per-channel statistics and save them to cache.

    Loads the CIFAR-100 training set with only a ToTensor transform,
    computes mean and std via calculate_mean_std, and persists the
    result to the pickle cache for future runs.

    Returns
    -------
    tuple[list[float], list[float]]
        A (mean, std) pair, each a list of 3 floats (one per channel).

    Example
    -------
    >>> mean, std = calculate_save_mean_std()
    >>> mean
    [0.5071, 0.4867, 0.4408]
    """
    print("Calculating CIFAR100 Dataset's Mean and STD...")
    stats_transform = transforms.Compose([transforms.ToTensor()])
    stats_dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=not os.path.exists(CIFAR_DIR),
        transform=stats_transform,
    )

    loader = DataLoader(stats_dataset, batch_size=256, shuffle=False)
    mean, std = calculate_mean_std(loader)
    save_data({"mean": mean, "std": std})
    print(f'Saved [mean:{mean}, std: {std}] to "{CIFAR_DIR}\\variables.pkl"')
    return (mean, std)


def get_mean_std() -> tuple[list[float], list[float]]:
    """
    Return CIFAR-100 per-channel mean and std, computing if not cached.

    Loads from the pickle cache if available, otherwise computes
    and saves via calculate_save_mean_std.

    Returns
    -------
    tuple[list[float], list[float]]
        A (mean, std) pair, each a list of 3 floats (one per channel).

    Example
    -------
    >>> mean, std = get_mean_std()
    >>> std
    [0.2675, 0.2565, 0.2761]
    """
    loaded_mean = load_data("mean")
    if not loaded_mean:
        return calculate_save_mean_std()
    else:
        loaded_std = load_data("std")
        print(
            f"Loaded [mean:{loaded_mean}, std: {loaded_std}]"
            f' from "{CIFAR_DIR}\\variables.pkl"'
        )
        return (loaded_mean, loaded_std)


def unnormalize(
    img: Tensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Tensor:
    """
    Reverse normalization on an image tensor.

    Arguments
    ---------
    img : Tensor
        Normalized image tensor of shape (C, H, W).
    mean : list[float], optional
        Per-channel mean used during normalization. Loaded from
        cache if None.
    std : list[float], optional
        Per-channel std used during normalization. Loaded from
        cache if None.

    Returns
    -------
    Tensor
        Unnormalized image tensor of shape (C, H, W).

    Example
    -------
    >>> img_display = unnormalize(img_tensor)
    """
    if mean is None:
        mean = load_data("mean")
    if std is None:
        std = load_data("std")

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return img * std_t + mean_t
