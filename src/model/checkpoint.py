import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Optimizer
from torch.amp import GradScaler
from glob_config import DEVICE, CHECKPOINT_DIR


def checkpoint_path(model_name: str, budget: float) -> str:
    return f"{CHECKPOINT_DIR}/{model_name}_budget{budget:.2f}.pt"


def find_latest_checkpoint(model_name: str, budget: float) -> str | None:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    prefix = f"{model_name}_budget{budget:.2f}"
    matches = [
        f
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not matches:
        return None
    return max(
        (os.path.join(CHECKPOINT_DIR, f) for f in matches),
        key=os.path.getmtime,
    )


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    scaler: GradScaler | None,
    train_loss: float,
    test_loss: float,
    test_acc: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": (
                scaler.state_dict() if scaler is not None else None
            ),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    scaler: GradScaler | None,
) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        return ckpt
    except Exception as e:
        print(f"Warning: could not load checkpoint {path}: {e}")
        return None
