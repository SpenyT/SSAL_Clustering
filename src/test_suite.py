import os

from data.dataset_type import IndexedCIFARSubset
from data.dataset import create_loader, get_indexed_datasets
from pipeline.resnet18_baseline import run_scratch, run_pretrained, Verbosity
from pipeline.ssalc_pipeline import run_ssalc
from glob_config import ANNOTATION_BUDGETS
from model.checkpoint import checkpoint_path


def run_resnet_budget_experiment(
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
    verbosity: Verbosity = "summary",
) -> None:
    """
    Run all baseline and SSALC experiments across all annotation budgets.

    For each budget in ANNOTATION_BUDGETS, runs ResNet-18 from scratch and
    ResNet-18 pretrained. Results are logged via ResultsLogger.

    Arguments
    ---------
    epochs : int
        Number of training epochs per experiment. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    batch_size : int
        Batch size for train and test loaders. Default: 128.
    verbosity : {"full", "summary", "quiet"}
        Controls tqdm and print output. "full": all bars persist and
        per-epoch lines are printed. "summary": bars are transient,
        only the final result line is printed. "quiet": bars are
        transient, nothing is printed. Default: "summary".

    Example
    -------
    >>> run_resnet_budget_experiment(epochs=30)
    >>> run_resnet_budget_experiment(epochs=30, verbosity="full")
    """

    train_dataset, test_dataset = get_indexed_datasets()
    test_loader = create_loader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    for budget in ANNOTATION_BUDGETS:
        resnet_subset = IndexedCIFARSubset.from_dataset(
            train_dataset, budget=budget
        )
        print(
            f"\nAnnotation Budget: {int(budget * 100)}%"
            f" ({len(resnet_subset)} samples)"
        )

        train_loader = create_loader(
            resnet_subset, batch_size=batch_size, shuffle=True
        )
        run_scratch(train_loader, test_loader, budget, epochs, lr, verbosity)
        run_pretrained(
            train_loader, test_loader, budget, epochs, lr, verbosity
        )


def run_ssalc_budget_experiment(
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
    skip_completed: bool = False,
) -> None:
    """
    Run SSALC across all annotation budgets.

    For each budget in ANNOTATION_BUDGETS, runs the full SSALC pipeline:
    active learning rounds, pseudo-labeling, and ResNet-18 training.
    Results are logged via ResultsLogger.

    Arguments
    ---------
    epochs : int
        Number of training epochs per experiment. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    batch_size : int
        Batch size for train, test, and feature extraction loaders. Default: 128.
    skip_completed : bool
        If True, skip any budget whose checkpoint file already exists on disk.
        Useful for resuming a interrupted Colab run. Default: False.

    Returns
    -------
    None
        Results are written to the log via ResultsLogger and checkpoints
        are saved for each budget.

    Example
    -------
    >>> run_ssalc_budget_experiment(epochs=30)
    >>> run_ssalc_budget_experiment(epochs=30, skip_completed=True)
    """
    train_dataset, test_dataset = get_indexed_datasets()
    test_loader = create_loader(test_dataset, batch_size=batch_size, shuffle=False)

    for budget in ANNOTATION_BUDGETS:
        print(f"\nAnnotation Budget: {int(budget * 100)}% ({int(budget * 50000)} samples)")
        if skip_completed and os.path.exists(checkpoint_path("SSALC", budget)):
            print(f"  Skipping — checkpoint already exists.")
            continue
        run_ssalc(
            train_dataset=train_dataset,
            test_loader=test_loader,
            budget=budget,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
