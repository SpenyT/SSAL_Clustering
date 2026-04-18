import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.resnet import load_resnet18
from model.model_utils import prepare_model, try_compile
from model.checkpoint import save_checkpoint, load_checkpoint, checkpoint_path, find_latest_checkpoint
import glob_config
from glob_config import DEVICE, USE_AMP
from visualize.results_logger import ResultsLogger, LogEntry


# if gpu ends up using amp, scaler just scales back up to float32
_scaler = GradScaler(device=DEVICE.type) if (DEVICE.type == "cuda" and USE_AMP) else None


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for (imgs, labels), _ in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if USE_AMP:
            with autocast(device_type=DEVICE.type):
                loss = criterion(model(imgs), labels)
            if _scaler is not None:
                _scaler.scale(loss).backward()
                _scaler.step(optimizer)
                _scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
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
        if USE_AMP:
            with autocast(device_type=DEVICE.type):
                logits = model(imgs)
                total_loss += criterion(logits, labels).item()
        else:
            logits = model(imgs)
            total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / len(loader), correct / n


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    path: str,
    model_name: str,
    budget: float,
    epochs: int = 30,
    lr: float = 0.01
) -> tuple[float, float, float, float, float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_batches = len(train_loader)
    batch_size = train_loader.batch_size
    n_samples = len(train_loader.dataset)

    train_loss = test_loss = test_acc = 0.0
    start_epoch = 0
    if glob_config.IS_RESUME:
        latest = find_latest_checkpoint(model_name, budget)
        ckpt = load_checkpoint(latest, model, optimizer, scheduler, _scaler) if latest else None
        if ckpt is not None:
            start_epoch = ckpt["epoch"]
            train_loss, test_loss, test_acc = ckpt["train_loss"], ckpt["test_loss"], ckpt["test_acc"]
            print(f"Resumed from epoch {start_epoch}/{epochs}")

    total_train_time = total_eval_time = 0.0
    t0 = time.time()
    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs", leave=False):
        t_train = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        total_train_time += time.perf_counter() - t_train

        t_eval = time.perf_counter()
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        total_eval_time += time.perf_counter() - t_eval

        scheduler.step()
        tqdm.write(f"Epoch {epoch:>3}/{epochs} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        save_checkpoint(path, epoch, model, optimizer, scheduler, _scaler, train_loss, test_loss, test_acc)

    elapsed = time.time() - t0
    completed = epochs - start_epoch
    avg_batch_time = elapsed / (completed * n_batches) if completed > 0 else 0.0
    print(
        f"Done | time: {elapsed:.1f}s | avg_batch: {avg_batch_time:.3f}s | "
        f"samples: {n_samples} | batch_size: {batch_size} | batches/epoch: {n_batches} | "
        f"train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
    )
    return train_loss, test_loss, test_acc, total_train_time, total_eval_time, elapsed


def run_pretrained(train_loader: DataLoader, test_loader: DataLoader, budget: float, epochs: int = 30, lr: float = 0.01) -> None:
    print("\n-- ResNet18 (pretrained) --")
    model = try_compile(prepare_model(load_resnet18(with_pretrained_weights=True)))
    train_loss, test_loss, test_acc, train_time, eval_time, elapsed = training_loop(
        model, train_loader, test_loader, checkpoint_path("ResNet18_pretrained", budget), "ResNet18_pretrained", budget, epochs, lr
    )
    ResultsLogger.write_log(LogEntry(
        model="ResNet18_pretrained", budget=budget, n_epochs=epochs,
        train_loss=train_loss, test_loss=test_loss, test_acc=test_acc,
        train_time=train_time, test_time=eval_time, total_elapsed_time=elapsed,
    ))


def run_scratch(train_loader: DataLoader, test_loader: DataLoader, budget: float, epochs: int = 30, lr: float = 0.01) -> None:
    print("\n-- ResNet18 (scratch) --")
    model = try_compile(prepare_model(load_resnet18(with_pretrained_weights=False)))
    train_loss, test_loss, test_acc, train_time, eval_time, elapsed = training_loop(
        model, train_loader, test_loader, checkpoint_path("ResNet18_scratch", budget), "ResNet18_scratch", budget, epochs, lr
    )
    ResultsLogger.write_log(LogEntry(
        model="ResNet18_scratch", budget=budget, n_epochs=epochs,
        train_loss=train_loss, test_loss=test_loss, test_acc=test_acc,
        train_time=train_time, test_time=eval_time, total_elapsed_time=elapsed,
    ))
