import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet import load_resnet18
from model.model_utils import prepare_model
from data.dataset_type import IndexedCIFAR100, IndexedCIFARSubset
from glob_config import ANNOTATION_BUDGETS, DEVICE, PIN_MEMORY


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for (imgs, labels), _ in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for (imgs, labels), _ in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / len(loader), correct / n


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.01
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_batches = len(train_loader)
    batch_size = train_loader.batch_size
    n_samples = len(train_loader.dataset)

    t0 = time.time()
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", leave=False):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        tqdm.write(f"Epoch {epoch:>3}/{epochs} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

    elapsed = time.time() - t0
    avg_batch_time = elapsed / (epochs * n_batches)
    print(
        f"Done | time: {elapsed:.1f}s | avg_batch: {avg_batch_time:.3f}s | "
        f"samples: {n_samples} | batch_size: {batch_size} | batches/epoch: {n_batches} | "
        f"train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
    )


def run_pretrained(train_loader: DataLoader, test_loader: DataLoader, epochs: int = 30, lr: float = 0.01) -> None:
    print("\nResNet18 (pretrained):")
    model = prepare_model(load_resnet18(with_pretrained_weights=True))
    training_loop(model, train_loader, test_loader, epochs, lr)


def run_scratch(train_loader: DataLoader, test_loader: DataLoader, epochs: int = 30, lr: float = 0.01) -> None:
    print("\nResNet18 (scratch):")
    model = prepare_model(load_resnet18(with_pretrained_weights=False))
    training_loop(model, train_loader, test_loader, epochs, lr)


def run_budget_experiments(
    train_dataset: IndexedCIFAR100,
    test_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
) -> None:
    for budget in ANNOTATION_BUDGETS:
        subset = IndexedCIFARSubset.from_dataset(train_dataset, budget=budget)
        print(f"\nAnnotation Budget: {int(budget * 100)}% ({len(subset)} samples)")
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=PIN_MEMORY, persistent_workers=True)

        print("-- pretrained --")
        model = prepare_model(load_resnet18(with_pretrained_weights=True))
        training_loop(model, train_loader, test_loader, epochs, lr)

        print("-- scratch --")
        model = prepare_model(load_resnet18(with_pretrained_weights=False))
        training_loop(model, train_loader, test_loader, epochs, lr)