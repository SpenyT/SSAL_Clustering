import torch
import random
import numpy as np
from typing import Final
from torch.utils.data import DataLoader

from data.dataset import get_indexed_datasets
from pipeline.resnet18_baseline import run_pretrained

SEED: Final[int] = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(SEED)

    train_dataset, test_dataset = get_indexed_datasets()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    run_pretrained(train_loader, test_loader, epochs=30)
