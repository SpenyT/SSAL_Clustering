from typing import Final
from torch.utils.data import DataLoader
from data.dataset_type import IndexedCIFARSubset
from data.dataset import get_indexed_datasets
from pipeline.resnet18_baseline import run_scratch, run_pretrained


from glob_config import ANNOTATION_BUDGETS, NUM_WORKERS, PIN_MEMORY


def run_budget_experiments(epochs: int = 30, lr: float = 0.01, batch_size: int = 128) -> None:
    train_dataset, test_dataset = get_indexed_datasets()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
    
    for budget in ANNOTATION_BUDGETS:
        resnet_subset = IndexedCIFARSubset.from_dataset(train_dataset, budget=budget)
        print(f"\nAnnotation Budget: {int(budget * 100)}% ({len(resnet_subset)} samples)")
        
        train_loader = DataLoader(resnet_subset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
        run_scratch(train_loader, test_loader, budget, epochs, lr)
        run_pretrained(train_loader, test_loader, budget, epochs, lr)
        
        #TODO: add SSALC