import csv
import os
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

    # could have used __iter__ too, but this will do
    def get_values(self):
        return [
            self.model,
            f"{self.budget:.2f}",
            self.n_epochs,
            f"{self.train_loss:.6f}",
            f"{self.test_loss:.6f}",
            f"{self.test_acc:.6f}",
            f"{self.train_time:.3f}",
            f"{self.test_time:.3f}",
            f"{self.total_elapsed_time:.3f}",
        ]


class ResultsLogger:
    _path: str | None = None

    def __new__(cls):
        raise TypeError("Use ResultsLogger as a static class.")

    @classmethod
    def init(cls, path: str, append: bool = False) -> None:
        cls._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not append:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(LogEntry.__dataclass_fields__.keys())

    @classmethod
    def write_log(cls, log: LogEntry) -> None:
        if cls._path is None:
            raise RuntimeError(
                "Call ResultsLogger.init() before writing logs."
            )
        with open(cls._path, "a", newline="") as f:
            csv.writer(f).writerow(log.get_values())
