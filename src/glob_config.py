import os
import re
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Final

# helpers


def _get_current_file_number(results_dir: str) -> int:
    if not os.path.isdir(results_dir):
        return 0
    pattern = re.compile(r"results_(\d+)\.csv")
    nums = [
        int(m.group(1))
        for f in os.listdir(results_dir)
        if (m := pattern.fullmatch(f))
    ]
    return max(nums, default=0)


def _get_results_file_path(results_dir: str) -> str:
    current = _get_current_file_number(RESULTS_DIR)
    if current == 0:
        return f"{results_dir}/results_1.csv"
    return f"{results_dir}/results_{current + 1}.csv"


# checks if gpu can use mixed precision
# essentially automatically casts floats from 32bit to 16bit when possible
# (by autocast)
def _is_amp_supported() -> bool:
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_capability()[0] >= 7
    if DEVICE.type == "mps":
        return True
    return False


def _try_import_cuml() -> bool:
    try:
        import importlib

        importlib.import_module("cuml")
        return True
    except ImportError:
        return False


# rand config
SEED: Final[int] = 42

# data paths config
BASE_DIR: Final[str] = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR: Final[str] = f"{BASE_DIR}/data"
CIFAR_DIR: Final[str] = f"{DATA_DIR}/cifar-100-python"
VARIABLES_PATH: Final[str] = f"{DATA_DIR}/variables.pkl"
RESULTS_DIR: Final[str] = f"{DATA_DIR}/results"
CHECKPOINT_DIR: Final[str] = f"{DATA_DIR}/checkpoints"
RESULTS_PATH: str = f"{RESULTS_DIR}/results_{
    _get_current_file_number(RESULTS_DIR) + 1}.csv"


# data definitions config
ANNOTATION_BUDGETS: Final[list[float]] = [0.05, 0.1, 0.2, 0.5, 1.0]
MODELS: Final[list[str]] = ["ResNet18_scratch", "ResNet18_pretrained", "SSALC"]

# device config
N_GPUS: Final[int] = torch.cuda.device_count()
DEVICE: Final[str] = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
PIN_MEMORY: Final[bool] = DEVICE.type != "cpu"
USE_AMP: Final[bool] = _is_amp_supported()
HAS_CUML: Final[bool] = _try_import_cuml()

# runtime config
# change to whatever is best for you
NUM_WORKERS: int = max(2, os.cpu_count() // 2)
IS_RESUME: bool = False
APPEND_LOG: bool = False


if DEVICE.type == "cuda":
    cudnn.benchmark = True

# dataloader worker config


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(
    num_workers: int = None, is_resume: bool = None, append_log: bool = None
) -> None:
    global NUM_WORKERS, IS_RESUME, APPEND_LOG, RESULTS_PATH
    if num_workers is not None:
        NUM_WORKERS = num_workers
    if is_resume is not None:
        IS_RESUME = is_resume
    if append_log is not None:
        APPEND_LOG = append_log
    if IS_RESUME and APPEND_LOG:
        RESULTS_PATH = _get_results_file_path(RESULTS_DIR)
