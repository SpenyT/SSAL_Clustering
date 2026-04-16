import os
import re
import torch
from typing import Final

# helper

def _get_next_file_num(results_dir: str) -> int:
    if not os.path.isdir(results_dir):
        return 1
    pattern = re.compile(r'results_(\d+)\.csv')
    nums = [int(m.group(1)) for f in os.listdir(results_dir) if (m := pattern.fullmatch(f))]
    return max(nums, default=0) + 1


# rand config
SEED: Final[int] = 42

# data paths config
BASE_DIR:       Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR:       Final[str] = f"{BASE_DIR}/data"
CIFAR_DIR:      Final[str] = f"{DATA_DIR}/cifar-100-python"
VARIABLES_PATH: Final[str] = f"{DATA_DIR}/variables.pkl"
RESULTS_DIR:    Final[str] = f"{DATA_DIR}/results"
RESULTS_PATH:   Final[str] = f"{RESULTS_DIR}/results_{_get_next_file_num(RESULTS_DIR)}.csv"

# data definitions config
ANNOTATION_BUDGETS: Final[list[float]] = [0.05, 0.1, 0.2, 0.5, 1.0]
MODELS : Final[list[str]] = ["ResNet18_scratch", "ResNet18_pretrained", "SSALC"]

# device config
NUM_WORKERS: Final[int]  = max(1, os.cpu_count() // 2) # change to whatever is best for you
N_GPUS:      Final[int]  = torch.cuda.device_count()
DEVICE:      Final[str]  = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PIN_MEMORY:  Final[bool] = DEVICE.type != "cpu"
