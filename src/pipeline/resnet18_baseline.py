import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Literal
from tqdm.auto import tqdm
from model.resnet import load_resnet18
from model.model_utils import prepare_model, try_compile
from model.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    checkpoint_path,
    find_latest_checkpoint,
)
import glob_config
from glob_config import DEVICE, USE_AMP
from visualize.results_logger import ResultsLogger, LogEntry

Verbosity = Literal["full", "summary", "quiet"]

# if gpu ends up using amp, scaler just scales back up to float32
_scaler = (
    GradScaler(device=DEVICE.type)
    if (DEVICE.type == "cuda" and USE_AMP)
    else None
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    leave: bool = False,
) -> float:
    """
    Run one training epoch over the dataset.

    Iterates all batches, computes cross-entropy loss, and updates
    model weights. Supports AMP with optional gradient scaling.

    Arguments
    ---------
    model : nn.Module
        The model to train.
    loader : DataLoader
        DataLoader over the training dataset.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    criterion : nn.Module
        Loss function.
    leave : bool
        Whether the tqdm bar persists after completion. Default: False.

    Returns
    -------
    float
        Mean training loss over all batches.

    Example
    -------
    >>> loss = train_epoch(model, train_loader, optimizer, criterion)
    >>> print(f"Train loss: {loss:.4f}")
    Train loss: 1.2345
    """
    model.train()
    total_loss = 0.0
    for (imgs, labels), _ in tqdm(loader, desc="Train", leave=leave):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(
            DEVICE, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
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
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, leave: bool = False
) -> tuple[float, float]:
    """
    Evaluate model loss and accuracy on a dataset.

    Runs inference over all batches without gradient computation.
    Supports AMP if enabled in glob_config.

    Arguments
    ---------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        DataLoader over the evaluation dataset.
    criterion : nn.Module
        Loss function.
    leave : bool
        Whether the tqdm bar persists after completion. Default: False.

    Returns
    -------
    tuple[float, float]
        A (loss, accuracy) pair, where loss is the mean over all
        batches and accuracy is the fraction of correct predictions.

    Example
    -------
    >>> loss, acc = evaluate(model, test_loader, criterion)
    >>> print(f"Loss: {loss:.4f} | Acc: {acc:.4f}")
    Loss: 0.8321 | Acc: 0.7654
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for (imgs, labels), _ in tqdm(loader, desc="Eval", leave=leave):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(
            DEVICE, non_blocking=True
        )
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
    lr: float = 0.01,
    verbosity: Verbosity = "summary",
) -> tuple[float, float, float, float, float, float]:
    """
    Run the full training loop with checkpointing and evaluation.

    Trains the model for the specified number of epochs using SGD with
    cosine annealing. Evaluates on the test set after each epoch and
    saves a checkpoint. Optionally resumes from the latest checkpoint
    if IS_RESUME is set in glob_config.

    Arguments
    ---------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader over the training dataset.
    test_loader : DataLoader
        DataLoader over the test dataset.
    path : str
        File path where checkpoints are saved.
    model_name : str
        Name used to identify checkpoints when resuming.
    budget : float
        Annotation budget fraction, used for checkpoint lookup.
    epochs : int
        Total number of training epochs. Default: 30.
    lr : float
        Initial learning rate for SGD. Default: 0.01.
    verbosity : {"full", "summary", "quiet"}
        Controls tqdm and print output. "full": all bars persist and
        per-epoch lines are printed. "summary": bars are transient,
        only the final result line is printed. "quiet": bars are
        transient, nothing is printed. Default: "summary".

    Returns
    -------
    tuple[float, float, float, float, float, float]
        A (train_loss, test_loss, test_acc, total_train_time,
        total_eval_time, elapsed) tuple from the final epoch.

    Example
    -------
    >>> results = training_loop(
    ...     model, train_loader, test_loader,
    ...     path="data/checkpoints/run.pt",
    ...     model_name="ResNet18_pretrained",
    ...     budget=0.1, epochs=30, lr=0.01,
    ... )
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    n_batches = len(train_loader)
    batch_size = train_loader.batch_size
    n_samples = len(train_loader.dataset)

    train_loss = test_loss = test_acc = 0.0
    start_epoch = 0
    if glob_config.IS_RESUME:
        latest = find_latest_checkpoint(model_name, budget)
        ckpt = (
            load_checkpoint(latest, model, optimizer, scheduler, _scaler)
            if latest
            else None
        )
        if ckpt is not None:
            start_epoch = ckpt["epoch"]
            train_loss, test_loss, test_acc = (
                ckpt["train_loss"],
                ckpt["test_loss"],
                ckpt["test_acc"],
            )
            print(f"Resumed from epoch {start_epoch}/{epochs}")

    leave = verbosity == "full"
    total_train_time = total_eval_time = 0.0
    t0 = time.time()
    for epoch in tqdm(
        range(start_epoch + 1, epochs + 1), desc="Epochs", leave=leave
    ):
        t_train = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, leave=leave)
        total_train_time += time.perf_counter() - t_train

        t_eval = time.perf_counter()
        test_loss, test_acc = evaluate(model, test_loader, criterion, leave=leave)
        total_eval_time += time.perf_counter() - t_eval

        scheduler.step()
        if verbosity == "full":
            tqdm.write(f"Epoch {
                epoch:>3}/{epochs} | train_loss: {
                train_loss:.4f} | test_loss: {
                test_loss:.4f} | test_acc: {
                test_acc:.4f}")
        save_checkpoint(
            path,
            epoch,
            model,
            optimizer,
            scheduler,
            _scaler,
            train_loss,
            test_loss,
            test_acc,
        )

    elapsed = time.time() - t0
    completed = epochs - start_epoch
    avg_batch_time = (
        elapsed / (completed * n_batches) if completed > 0 else 0.0
    )
    if verbosity != "quiet":
        print(
            f"Done | time: {elapsed:.1f}s | avg_batch: {avg_batch_time:.3f}s | "
            f"samples: {n_samples} | batch_size: {batch_size} | "
            f"batches/epoch: {n_batches} | "
            f"train_loss: {
                train_loss:.4f} | test_loss: {
                test_loss:.4f} | test_acc: {
                test_acc:.4f}"
        )
    return (
        train_loss,
        test_loss,
        test_acc,
        total_train_time,
        total_eval_time,
        elapsed,
    )


def run_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    budget: float,
    pretrained: bool,
    epochs: int = 30,
    lr: float = 0.01,
    verbosity: Verbosity = "summary",
) -> None:
    """
    Train a ResNet-18 baseline and log the results.

    Loads a ResNet-18 (pretrained or from scratch), runs the full
    training loop, and writes results to the results log via
    ResultsLogger.

    Arguments
    ---------
    train_loader : DataLoader
        DataLoader over the training subset.
    test_loader : DataLoader
        DataLoader over the test dataset.
    budget : float
        Annotation budget fraction, in range (0, 1].
    pretrained : bool
        If True, initializes with ImageNet weights. If False, trains
        from scratch.
    epochs : int
        Number of training epochs. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    verbosity : {"full", "summary", "quiet"}
        Controls tqdm and print output. "full": all bars persist and
        per-epoch lines are printed. "summary": bars are transient,
        only the final result line is printed. "quiet": bars are
        transient, nothing is printed. Default: "summary".

    Example
    -------
    >>> run_baseline(train_loader, test_loader, budget=0.1, pretrained=True)
    """
    label = "ResNet18_pretrained" if pretrained else "ResNet18_scratch"
    print(f"\n-- ResNet18 ({'pretrained' if pretrained else 'scratch'}) --")
    model = try_compile(
        prepare_model(load_resnet18(with_pretrained_weights=pretrained))
    )
    train_loss, test_loss, test_acc, train_time, eval_time, elapsed = (
        training_loop(
            model,
            train_loader,
            test_loader,
            checkpoint_path(label, budget),
            label,
            budget,
            epochs,
            lr,
            verbosity,
        )
    )
    ResultsLogger.write_log(
        LogEntry(
            model=label,
            budget=budget,
            n_epochs=epochs,
            train_loss=train_loss,
            test_loss=test_loss,
            test_acc=test_acc,
            train_time=train_time,
            test_time=eval_time,
            total_elapsed_time=elapsed,
        )
    )


def run_pretrained(
    train_loader: DataLoader,
    test_loader: DataLoader,
    budget: float,
    epochs: int = 30,
    lr: float = 0.01,
    verbosity: Verbosity = "summary",
) -> None:
    """
    Train a ResNet-18 initialized with ImageNet-pretrained weights.

    Convenience wrapper around run_baseline(pretrained=True).

    Arguments
    ---------
    train_loader : DataLoader
        DataLoader over the training subset.
    test_loader : DataLoader
        DataLoader over the test dataset.
    budget : float
        Annotation budget fraction, in range (0, 1].
    epochs : int
        Number of training epochs. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    verbosity : {"full", "summary", "quiet"}
        Controls tqdm and print output. "full": all bars persist and
        per-epoch lines are printed. "summary": bars are transient,
        only the final result line is printed. "quiet": bars are
        transient, nothing is printed. Default: "summary".

    Example
    -------
    >>> run_pretrained(train_loader, test_loader, budget=0.1)
    """
    run_baseline(
        train_loader,
        test_loader,
        budget,
        pretrained=True,
        epochs=epochs,
        lr=lr,
        verbosity=verbosity,
    )


def run_scratch(
    train_loader: DataLoader,
    test_loader: DataLoader,
    budget: float,
    epochs: int = 30,
    lr: float = 0.01,
    verbosity: Verbosity = "summary",
) -> None:
    """
    Train a ResNet-18 initialized with random weights.

    Convenience wrapper around run_baseline(pretrained=False).

    Arguments
    ---------
    train_loader : DataLoader
        DataLoader over the training subset.
    test_loader : DataLoader
        DataLoader over the test dataset.
    budget : float
        Annotation budget fraction, in range (0, 1].
    epochs : int
        Number of training epochs. Default: 30.
    lr : float
        Initial learning rate. Default: 0.01.
    verbosity : {"full", "summary", "quiet"}
        Controls tqdm and print output. "full": all bars persist and
        per-epoch lines are printed. "summary": bars are transient,
        only the final result line is printed. "quiet": bars are
        transient, nothing is printed. Default: "summary".

    Example
    -------
    >>> run_scratch(train_loader, test_loader, budget=0.1)
    """
    run_baseline(
        train_loader,
        test_loader,
        budget,
        pretrained=False,
        epochs=epochs,
        lr=lr,
        verbosity=verbosity,
    )
