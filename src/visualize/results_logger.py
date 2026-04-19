import csv
import os
from dataclasses import dataclass
from model.model_utils import ModelName


@dataclass(slots=True, frozen=True)
class LogEntry:
    """
    Immutable record of a single experiment result.

    Stores all metrics from one training run for serialization
    to CSV via ResultsLogger.

    Arguments
    ---------
    model : str
        Name of the model (e.g. "ResNet18_pretrained").
    budget : float
        Annotation budget fraction used for this run.
    epoch : int
        Current epoch number (one row per epoch).
    train_loss : float
        Training loss for this epoch.
    test_loss : float
        Test loss for this epoch.
    test_acc : float
        Test accuracy for this epoch, in range [0, 1].
    train_time : float
        Training time for this epoch in seconds.
    test_time : float
        Evaluation time for this epoch in seconds.
    total_elapsed_time : float
        Cumulative wall-clock time since training started, in seconds.

    Example
    -------
    >>> entry = LogEntry(
    ...     model="ResNet18_pretrained", budget=0.1, epoch=5,
    ...     train_loss=0.5, test_loss=0.6, test_acc=0.82,
    ...     train_time=4.2, test_time=0.8, total_elapsed_time=25.1,
    ... )
    """

    model: ModelName
    budget: float
    epoch: int
    train_loss: float
    test_loss: float
    test_acc: float
    train_time: float
    test_time: float
    total_elapsed_time: float

    # could have used __iter__ too, but this will do
    def get_values(self):
        """Return formatted field values for CSV serialization.

        Returns
        -------
        list
            All fields as a flat list with numeric values formatted
            to fixed decimal places.

        Example
        -------
        >>> entry.get_values()
        ['ResNet18_pretrained', '0.10', 30, '0.500000', ...]
        """
        return [
            self.model,
            f"{self.budget:.2f}",
            self.epoch,
            f"{self.train_loss:.6f}",
            f"{self.test_loss:.6f}",
            f"{self.test_acc:.6f}",
            f"{self.train_time:.3f}",
            f"{self.test_time:.3f}",
            f"{self.total_elapsed_time:.3f}",
        ]


class ResultsLogger:
    """
    Static class for writing experiment results to a CSV file.

    Must be initialized with init() before any calls to write_log().
    Not intended to be instantiated since (methods are class methods).

    Example
    -------
    >>> ResultsLogger.init("data/results/results_1.csv")
    >>> ResultsLogger.write_log(entry)
    """

    _path: str | None = None

    def __new__(cls):
        raise TypeError("Use ResultsLogger as a static class.")

    @classmethod
    def init(cls, path: str, append: bool = False) -> None:
        """Initialize the results CSV file and set the output path.

        Creates the output directory if it does not exist. If append
        is False, overwrites any existing file and writes the header row.

        Arguments
        ---------
        path : str
            File path for the CSV output.
        append : bool
            If True, skips writing the header and appends to the existing
            file. Default: False.

        Example
        -------
        >>> ResultsLogger.init("data/results/results_1.csv", append=False)
        """
        cls._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not append:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(LogEntry.__dataclass_fields__.keys())

    @classmethod
    def write_log(cls, log: LogEntry) -> None:
        """Append a LogEntry to the results CSV file.

        Arguments
        ---------
        log : LogEntry
            The experiment result to write.

        Example
        -------
        >>> ResultsLogger.write_log(entry)
        """
        if cls._path is None:
            raise RuntimeError(
                "Call ResultsLogger.init() before writing logs."
            )
        with open(cls._path, "a", newline="") as f:
            csv.writer(f).writerow(log.get_values())
