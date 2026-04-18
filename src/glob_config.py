import os
import re
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Final


def _get_current_file_number(results_dir: str) -> int:
    """
    Return the highest results file number found in results_dir.

    Scans for files matching results_N.csv and returns the largest N.
    Returns 0 if the directory does not exist or contains no matches.

    Arguments
    ---------
    results_dir : str
        Path to the results directory to scan.

    Returns
    -------
    int
        Highest results file number found, or 0 if none exist.

    Example
    -------
    >>> _get_current_file_number("data/results")
    3
    """
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
    """
    Return the path for the next results CSV file.

    Arguments
    ---------
    results_dir : str
        Path to the results directory.

    Returns
    -------
    str
        File path of the form results_dir/results_N.csv, where N is
        one greater than the current highest file number.

    Example
    -------
    >>> _get_results_file_path("data/results")
    'data/results/results_4.csv'
    """
    current = _get_current_file_number(results_dir)
    if current == 0:
        return f"{results_dir}/results_1.csv"
    return f"{results_dir}/results_{current + 1}.csv"


def _is_amp_supported() -> bool:
    """
    Check whether the current device supports automatic mixed precision.

    AMP is supported on CUDA devices with compute capability >= 7.0
    (Volta and later) and on all MPS devices.

    Returns
    -------
    bool
        True if AMP is supported on the current device.

    Example
    -------
    >>> USE_AMP = _is_amp_supported()
    """
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_capability()[0] >= 7
    if DEVICE.type == "mps":
        return True
    return False


def _get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Returns CUDA if a GPU is available, MPS if on Apple Silicon,
    and CPU otherwise.

    Returns
    -------
    torch.device
        The selected compute device.

    Example
    -------
    >>> DEVICE = _get_device()
    >>> DEVICE.type
    'cuda'
    """
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return torch.device(device_type)


def _try_import_cuml() -> bool:
    """
    Check whether cuML is available in the current environment.

    Returns
    -------
    bool
        True if cuML can be imported, False otherwise.

    Example
    -------
    >>> HAS_CUML = _try_import_cuml()
    """
    try:
        import importlib

        importlib.import_module("cuml")
        return True
    except ImportError:
        return False


def seed_worker(_worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs for a DataLoader worker.

    Passed as worker_init_fn to DataLoader to ensure reproducible
    data loading across workers. Each worker derives its seed from
    PyTorch's initial seed.

    Arguments
    ---------
    _worker_id : int
        Worker index assigned by DataLoader. Not used directly:
        the seed is derived from torch.initial_seed() instead.

    Example
    -------
    >>> DataLoader(dataset, worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
DEVICE: Final[torch.device] = _get_device()
PIN_MEMORY: Final[bool] = DEVICE.type != "cpu"
USE_AMP: Final[bool] = _is_amp_supported()
HAS_CUML: Final[bool] = _try_import_cuml()

# runtime config
NUM_WORKERS: int = max(2, os.cpu_count() // 2)
IS_RESUME: bool = False
APPEND_LOG: bool = False


if DEVICE.type == "cuda":
    cudnn.benchmark = True


def load_config(
    num_workers: int = None, is_resume: bool = None, append_log: bool = None
) -> None:
    """
    Override runtime configuration globals at startup.

    Any argument left as None retains its current value. If both
    is_resume and append_log are True, RESULTS_PATH is updated to
    point to the next available results file.

    Arguments
    ---------
    num_workers : int, optional
        Number of DataLoader worker processes.
    is_resume : bool, optional
        If True, the pipeline resumes from the latest checkpoint.
    append_log : bool, optional
        If True, results are written to a new file rather than
        overwriting the existing one.

    Example
    -------
    >>> load_config(num_workers=4, is_resume=True, append_log=True)
    """
    global NUM_WORKERS, IS_RESUME, APPEND_LOG, RESULTS_PATH
    if num_workers is not None:
        NUM_WORKERS = num_workers
    if is_resume is not None:
        IS_RESUME = is_resume
    if append_log is not None:
        APPEND_LOG = append_log
    if IS_RESUME and APPEND_LOG:
        RESULTS_PATH = _get_results_file_path(RESULTS_DIR)
