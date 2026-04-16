import os
import torch
from typing import Final

# rand config
SEED: Final[int] = 42

# data paths config
BASE_DIR:       Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR:       Final[str] = f"{BASE_DIR}/data"
CIFAR_DIR:      Final[str] = f"{DATA_DIR}/cifar-100-python"
VARIABLES_PATH: Final[str] = f"{DATA_DIR}/variables.pkl"

# data definitions config
ANNOTATION_BUDGETS: Final[list[float]] = [0.05, 0.1, 0.2, 0.5, 1.0]

# device config
NUM_WORKERS: Final[int]  = max(1, os.cpu_count() // 2) # change to whatever is best for you
N_GPUS:      Final[int]  = torch.cuda.device_count()
DEVICE:      Final[str]  = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PIN_MEMORY:  Final[bool] = DEVICE.type != "cpu"
