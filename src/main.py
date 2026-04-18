import argparse
import warnings
import torch
import random
import numpy as np
from typing import Final
import glob_config
from glob_config import SEED, load_config
from test_suite import run_budget_experiments
from visualize.results_logger import ResultsLogger


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSAL Clustering experiment runner")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--append-log", action="store_true", help="Append to last results file when resuming (default: new file)")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*tuple.*subarray.*", category=DeprecationWarning) # was getting annoying
    args = _parse_args()
    set_seed(SEED)

    N_EPOCHS: Final[int] = 30
    LR: Final[float] = 0.01
    BATCH_SIZE: Final[int] = 128

    load_config(is_resume=args.resume, append_log=args.append_log)
    ResultsLogger.init(glob_config.RESULTS_PATH, append=glob_config.APPEND_LOG)

    run_budget_experiments(N_EPOCHS, LR, BATCH_SIZE)
