from data.dataset_type import IndexedCIFARSubset
from data.dataset import get_indexed_datasets, create_loader
from pipeline.resnet18_baseline import run_scratch, run_pretrained
from pipeline.ssalc_pipeline import run_ssalc
from glob_config import ANNOTATION_BUDGETS


def run_budget_experiments(
    epochs: int = 30, lr: float = 0.01, batch_size: int = 128
) -> None:
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

        run_ssalc(train_dataset, test_loader, budget, epochs, lr, batch_size)
