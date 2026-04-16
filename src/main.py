import torch
import random
import numpy as np
from typing import Final
from torch.utils.data import DataLoader

from data.dataset import get_indexed_datasets
from data.dataset_type import IndexedCIFARSubset
from data.utils import load_data, calculate_save_mean_std
from pipeline.resnet18_baseline import run_pretrained
from glob_config import SEED

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

    subset = IndexedCIFARSubset.from_dataset(train_dataset, budget=0.1) # 10% annot. budget
    (img, label), idx = subset[0]

    # TODO: when implementing find a way to set num_workers automatically without taking all cores
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    run_pretrained(train_loader, test_loader, epochs=10)
