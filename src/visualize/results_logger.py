import csv
import os
from glob_config import RESULTS_DIR, RESULTS_PATH
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class LogEntry:
    model: str
    budget: float
    n_epochs: int
    train_loss: float
    test_loss: float
    test_acc: float
    train_time: float
    test_time: float
    total_elapsed_time: float


class ResultsLogger:
    _initialized: bool = False

    def __new__(cls):
        raise TypeError("Use ResultsLogger as a static class.")

    @classmethod
    def _ensure_open(cls) -> None:
        if cls._initialized:
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(RESULTS_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["model", "budget", "n_epochs", "train_loss", "test_loss", "test_acc", "train_time", "test_time", "total_elapsed_time"])
        cls._initialized = True

    @classmethod
    def write_log(cls, log: LogEntry) -> None:
        cls._ensure_open()
        with open(RESULTS_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                log.model,
                f"{log.budget:.2f}",
                log.n_epochs,
                f"{log.train_loss:.6f}",
                f"{log.test_loss:.6f}",
                f"{log.test_acc:.6f}",
                f"{log.train_time:.3f}",
                f"{log.test_time:.3f}",
                f"{log.total_elapsed_time:.3f}"
            ])