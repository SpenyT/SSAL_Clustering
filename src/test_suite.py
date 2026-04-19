from data.dataset_type import IndexedCIFARSubset
from data.dataset import create_loader, get_indexed_datasets
from pipeline.resnet18_baseline import run_scratch, run_pretrained
from glob_config import ANNOTATION_BUDGETS

def run_resnet_budget_experiment(
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
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

    Example
    -------
    >>> run_budget_experiments(epochs=30)
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
        run_scratch(train_loader, test_loader, budget, epochs, lr)
        run_pretrained(train_loader, test_loader, budget, epochs, lr)