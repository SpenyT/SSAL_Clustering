import torch
import random
import numpy as np
from typing import Final
from test_suite import run_budget_experiments
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
    
    N_EPOCHS: Final[int] = 30
    LR: Final[float] = 0.01
    BATCH_SIZE: Final[int] = 128

    run_budget_experiments(N_EPOCHS, LR, BATCH_SIZE)
