import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet import load_resnet18
from model.config import DEVICE, prepare_model


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

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        tqdm.write(f"Epoch {epoch:>3}/{epochs} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")


def run_pretrained(train_loader: DataLoader, test_loader: DataLoader, epochs: int = 30, lr: float = 0.01) -> None:
    print("\nResNet18 (pretrained):")
    model = prepare_model(load_resnet18(with_pretrained_weights=True))
    training_loop(model, train_loader, test_loader, epochs, lr)


def run_scratch(train_loader: DataLoader, test_loader: DataLoader, epochs: int = 30, lr: float = 0.01) -> None:
    print("\nResNet18 (scratch):")
    model = prepare_model(load_resnet18(with_pretrained_weights=False))
    training_loop(model, train_loader, test_loader, epochs, lr)


# TODO: add annotation budget experiments (train on subsets of the data)
