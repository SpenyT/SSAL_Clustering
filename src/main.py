import torch
import random
import numpy as np
from typing import Final
from torch.utils.data import DataLoader

from data.dataset import get_indexed_datasets
from data.utils import load_data, calculate_save_mean_std
from pipeline.resnet18_baseline import run_budget_experiments
from glob_config import NUM_WORKERS, SEED, PIN_MEMORY, DEVICE

def set_seed(seed : int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(SEED)

    if not load_data("mean"):
        calculate_save_mean_std()

    train_dataset, test_dataset = get_indexed_datasets()

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
    run_budget_experiments(train_dataset, test_loader, epochs=10)
