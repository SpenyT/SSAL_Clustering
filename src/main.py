import torch
import random
import numpy as np
from typing import Final

from data.dataset import get_indexed_datasets

SEED: Final[int] = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(SEED)

    train_dataset, test_dataset = get_indexed_datasets()

    (img, label), idx = train_dataset[0]
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Sample 0 — class: {train_dataset.classes[label]}, idx: {idx}, shape: {img.shape}")