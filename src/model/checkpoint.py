import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Optimizer
from torch.amp import GradScaler
from glob_config import DEVICE, CHECKPOINT_DIR
from model.model_utils import ModelName


def checkpoint_path(model_name: ModelName, budget: float) -> str:
    """
    Build the canonical file path for a model checkpoint.

    Arguments
    ---------
    model_name : ModelName
        The registered model name (e.g. "ResNet18_pretrained").
    budget : float
        Annotation budget fraction used when the model was trained.

    Returns
    -------
    str
        The full file path for the checkpoint file.

    Example
    -------
    >>> checkpoint_path("ResNet18_pretrained", 0.1)
    'checkpoints/ResNet18_pretrained_budget0.10.pt'
    """
    return f"{CHECKPOINT_DIR}/{model_name}_budget{budget:.2f}.pt"


def find_latest_checkpoint(model_name: ModelName, budget: float) -> str | None:
    """
    Find the most recently modified checkpoint for a given model and budget.

    Scans CHECKPOINT_DIR for files matching the pattern
    "<model_name>_budget<budget>.pt" and returns the one with the
    latest modification time.

    Arguments
    ---------
    model_name : ModelName
        The registered model name (e.g. "ResNet18_pretrained").
    budget : float
        Annotation budget fraction used when the model was trained.

    Returns
    -------
    str or None
        Absolute path to the latest matching checkpoint, or None if
        CHECKPOINT_DIR does not exist or no matching file is found.

    Example
    -------
    >>> find_latest_checkpoint("ResNet18_pretrained", 0.1)
    'checkpoints/ResNet18_pretrained_budget0.10.pt'
    """
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
    history: list[dict],
) -> None:
    """
    Serialize training state to a checkpoint file.

    Creates the parent directory if it does not exist, then saves all
    state dicts and training metrics in a single .pt file.

    Reference
    ---------
    - https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    - https://apxml.com/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/saving-loading-model-checkpoints

    Arguments
    ---------
    path : str
        Destination file path for the checkpoint.
    epoch : int
        The epoch number just completed.
    model : nn.Module
        The model whose weights will be saved.
    optimizer : Optimizer
        The optimizer whose state will be saved.
    scheduler : lr_scheduler._LRScheduler
        The learning rate scheduler whose state will be saved.
    scaler : GradScaler or None
        The AMP gradient scaler, or None if AMP is not used.
    train_loss : float
        Training loss for the current epoch.
    test_loss : float
        Validation/test loss for the current epoch.
    test_acc : float
        Validation/test accuracy for the current epoch.
    history : list[dict]
        Full per-epoch training history accumulated so far.

    Returns
    -------
    None

    Example
    -------
    >>> save_checkpoint("checkpoints/run.pt", epoch=5, model=model,
    ...                 optimizer=opt, scheduler=sched, scaler=None,
    ...                 train_loss=0.4, test_loss=0.5, test_acc=0.82,
    ...                 history=[])
    """
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
            "history": history,
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
    """
    Load a checkpoint and restore all training state in-place.

    Restores the model, optimizer, scheduler, and scaler (if provided)
    from the checkpoint at the given path. Returns the raw checkpoint
    dict so callers can access epoch number, loss history, etc.

    Reference
    ---------
    - https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    - https://apxml.com/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/saving-loading-model-checkpoints

    Arguments
    ---------
    path : str
        Path to the checkpoint file.
    model : nn.Module
        Model to restore weights into (modified in-place).
    optimizer : Optimizer
        Optimizer to restore state into (modified in-place).
    scheduler : lr_scheduler._LRScheduler
        Scheduler to restore state into (modified in-place).
    scaler : GradScaler or None
        AMP scaler to restore state into, or None if AMP is not used.

    Returns
    -------
    dict or None
        The raw checkpoint dict on success, or None if the file does
        not exist or loading fails.

    Example
    -------
    >>> ckpt = load_checkpoint("checkpoints/run.pt", model, opt, sched, scaler)
    >>> if ckpt:
    ...     start_epoch = ckpt["epoch"] + 1
    """
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


def load_model(model_name: ModelName, budget: float) -> nn.Module:
    """
    Load a trained model from its checkpoint by name and budget.

    Looks up the model architecture from a built-in registry, finds the
    latest matching checkpoint, and restores only the model weights.
    The returned model is in eval mode on DEVICE.

    Reference
    ---------
    - https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    - https://apxml.com/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/saving-loading-model-checkpoints


    Arguments
    ---------
    model_name : str
        One of the registered model names (e.g. "ResNet18_pretrained",
        "ResNet18_scratch").
    budget : float
        Annotation budget fraction used when the model was trained.

    Returns
    -------
    nn.Module
        The model with checkpoint weights loaded, in eval mode.

    Raises
    ------
    ValueError
        If model_name is not in the registry.
    FileNotFoundError
        If no checkpoint exists for the given model_name and budget.

    Example
    -------
    >>> model = load_model("ResNet18_pretrained", budget=0.1)
    """
    from model.resnet import load_resnet18

    _registry: dict[str, callable] = {
        "ResNet18_pretrained": lambda: load_resnet18(
            with_pretrained_weights=False
        ),
        "ResNet18_scratch": lambda: load_resnet18(
            with_pretrained_weights=False
        ),
    }

    if model_name not in _registry:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Registered: {list(_registry.keys())}"
        )

    path = find_latest_checkpoint(model_name, budget)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {model_name} at budget={budget:.2f}"
        )

    model = _registry[model_name]()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()
