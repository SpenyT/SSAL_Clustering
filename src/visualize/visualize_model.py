import os
import pickle
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from glob_config import DEVICE, CIFAR_DIR, PLOTS_DIR
from model.model_utils import ModelName
from model.checkpoint import load_model

sns.set_theme(style="white")


@functools.lru_cache(maxsize=1)
def _cifar100_meta() -> tuple[list[str], list[str], list[int]]:
    """Load and cache fine names, coarse names, and fine→coarse mapping."""
    with open(f"{CIFAR_DIR}/meta", "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    fine_names = [n.decode() for n in meta[b"fine_label_names"]]
    coarse_names = [n.decode() for n in meta[b"coarse_label_names"]]

    with open(f"{CIFAR_DIR}/test", "rb") as f:
        data = pickle.load(f, encoding="bytes")
    fine_to_coarse = [0] * 100
    for fine, coarse in zip(data[b"fine_labels"], data[b"coarse_labels"]):
        fine_to_coarse[fine] = coarse

    return fine_names, coarse_names, fine_to_coarse


def _predict(
    model: torch.nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for (imgs, labels), _ in loader:
            imgs = imgs.to(DEVICE)
            preds.append(model(imgs).argmax(dim=1).cpu())
            targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def _save(fig: plt.Figure, save_dir: str | None, name: str) -> None:
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")


def plot_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    save_dir: str | None = None,
) -> None:
    """
    Plot a normalized confusion matrix over all 100 CIFAR-100 fine classes.

    Arguments
    ---------
    model : nn.Module
        Trained model in eval mode on DEVICE.
    loader : DataLoader
        Test DataLoader returning ((imgs, labels), idx) batches.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> model = load_model("ResNet18_pretrained", budget=0.1)
    >>> plot_confusion_matrix(model, test_loader)
    """
    fine_names, _, _ = _cifar100_meta()
    preds, targets = _predict(model, loader)
    cm = confusion_matrix(targets, preds, normalize="true")
    overall_acc = cm.diagonal().mean()
    print(f"Overall accuracy (mean per-class): {overall_acc:.4f}")

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        ax=ax,
        xticklabels=fine_names,
        yticklabels=fine_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — 100 Fine Classes (row-normalized)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    _save(fig, save_dir, "confusion_matrix_fine")
    plt.show()


def plot_superclass_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    save_dir: str | None = None,
) -> None:
    """
    Plot a normalized confusion matrix over CIFAR-100's 20 superclasses.

    The fine→superclass mapping is loaded directly from the CIFAR-100
    data files, so it is always authoritative.

    Arguments
    ---------
    model : nn.Module
        Trained model in eval mode on DEVICE.
    loader : DataLoader
        Test DataLoader returning ((imgs, labels), idx) batches.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> model = load_model("ResNet18_pretrained", budget=0.1)
    >>> plot_superclass_confusion_matrix(model, test_loader)
    """
    _, coarse_names, fine_to_coarse = _cifar100_meta()
    preds, targets = _predict(model, loader)
    super_preds = np.array([fine_to_coarse[p] for p in preds])
    super_targets = np.array([fine_to_coarse[t] for t in targets])
    cm = confusion_matrix(super_targets, super_preds, normalize="true")
    per_superclass_acc = cm.diagonal()
    col_w = max(len(n) for n in coarse_names)
    print("Per-superclass accuracy:")
    for name, acc in sorted(
        zip(coarse_names, per_superclass_acc), key=lambda x: x[1]
    ):
        print(f"  {name:<{col_w}}  {acc:.4f}")
    print(f"  {'mean':<{col_w}}  {per_superclass_acc.mean():.4f}")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        xticklabels=coarse_names,
        yticklabels=coarse_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — 20 Superclasses (row-normalized)")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    _save(fig, save_dir, "confusion_matrix_superclass")
    plt.show()


def plot_per_class_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    save_dir: str | None = None,
) -> None:
    """
    Horizontal bar chart of per-class accuracy, sorted lowest to highest.

    A vertical dashed line marks the mean accuracy across all classes.

    Arguments
    ---------
    model : nn.Module
        Trained model in eval mode on DEVICE.
    loader : DataLoader
        Test DataLoader returning ((imgs, labels), idx) batches.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> model = load_model("ResNet18_pretrained", budget=0.1)
    >>> plot_per_class_accuracy(model, test_loader)
    """
    fine_names, _, _ = _cifar100_meta()
    preds, targets = _predict(model, loader)
    cm = confusion_matrix(targets, preds, normalize="true")
    acc = cm.diagonal()
    order = np.argsort(acc)

    col_w = max(len(n) for n in fine_names)
    print("Per-class accuracy (worst -> best):")
    for i in order:
        print(f"  {fine_names[i]:<{col_w}}  {acc[i]:.4f}")
    print(f"  {'mean':<{col_w}}  {acc.mean():.4f}")

    fig, ax = plt.subplots(figsize=(6, 22))
    ax.barh([fine_names[i] for i in order], acc[order])
    ax.axvline(
        acc.mean(),
        color="red",
        linestyle="--",
        label=f"mean = {acc.mean():.3f}",
    )
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy (sorted)")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_dir, "per_class_accuracy")
    plt.show()


def plot_all(
    model_name: ModelName,
    budget: float,
    loader: DataLoader,
    save_dir: str | None = None,
) -> None:
    """
    Load a model from checkpoint and run all three model diagnostic plots.

    Arguments
    ---------
    model_name : ModelName
        Registered model name (e.g. "ResNet18_pretrained").
    budget : float
        Annotation budget the model was trained on.
    loader : DataLoader
        Test DataLoader returning ((imgs, labels), idx) batches.
    save_dir : str | None
        Directory to save plot PNGs. Default: None (no save).

    Example
    -------
    >>> plot_all("ResNet18_pretrained", budget=0.1, loader=test_loader)
    >>> plot_all("ResNet18_pretrained", budget=1.0, loader=test_loader, save_dir="data/plots")
    """
    model = load_model(model_name, budget)
    plot_confusion_matrix(model, loader, save_dir)
    plot_superclass_confusion_matrix(model, loader, save_dir)
    plot_per_class_accuracy(model, loader, save_dir)


if __name__ == "__main__":
    import argparse
    from glob_config import ANNOTATION_BUDGETS
    from model.model_utils import MODELS
    from data.dataset import create_loader, get_indexed_datasets

    parser = argparse.ArgumentParser(
        description="Plot model diagnostics from a checkpoint."
    )
    parser.add_argument(
        "--model",
        default="ResNet18_pretrained",
        choices=MODELS,
        help="Model name (default: ResNet18_pretrained).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=ANNOTATION_BUDGETS[-1],
        help=f"Annotation budget (default: {ANNOTATION_BUDGETS[-1]}).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--save",
        nargs="?",
        const=PLOTS_DIR,
        default=None,
        metavar="DIR",
        help=f"Save plots as PNGs (default dir: {PLOTS_DIR}).",
    )
    args = parser.parse_args()

    _, test_dataset = get_indexed_datasets()
    test_loader = create_loader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    plot_all(args.model, args.budget, test_loader, save_dir=args.save)
