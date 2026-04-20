import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob_config
from model.model_utils import ModelName

sns.set_theme(style="darkgrid")


def _load() -> pd.DataFrame:
    return pd.read_csv(glob_config.RESULTS_PATH)


def _last_epoch(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["model", "budget"])["epoch"].idxmax()]


def _save(fig: plt.Figure, save_dir: str | None, name: str) -> None:
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight"
    )


def plot_epoch_curves(
    models: list[ModelName] | None = None,
    budgets: list[float] | None = None,
    metric: str = "test_acc",
    save_dir: str | None = None,
) -> None:
    """
    Plot a training metric over epochs, one subplot per annotation budget.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    budgets : list[float] | None
        Budgets to include. Default: all budgets in the CSV.
    metric : str
        Column name to plot (e.g. "test_acc", "test_loss", "train_loss").
        Default: "test_acc".
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> plot_epoch_curves(models=["ResNet18_pretrained"], metric="test_loss")
    """
    df = _load()
    if models:
        df = df[df["model"].isin(models)]
    budgets = sorted(df["budget"].unique()) if budgets is None else budgets
    df = df[df["budget"].isin(budgets)]

    n = len(budgets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    axes = [axes] if n == 1 else list(axes)

    for ax, budget in zip(axes, budgets):
        sub = df[df["budget"] == budget]
        for model_name, grp in sub.groupby("model"):
            ax.plot(grp["epoch"], grp[metric], label=model_name)
        ax.set_title(f"Budget {int(budget * 100)}%")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(metric.replace("_", " ").title())

    plt.suptitle(metric.replace("_", " ").title() + " per Epoch", y=1.02)
    plt.tight_layout()
    _save(fig, save_dir, f"epoch_curves_{metric}")
    plt.show()


def plot_accuracy_vs_budget(
    models: list[ModelName] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Plot final test accuracy against annotation budget for each model.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> plot_accuracy_vs_budget()
    """
    df = _last_epoch(_load())
    if models:
        df = df[df["model"].isin(models)]

    fig = plt.figure(figsize=(7, 4))
    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("budget")
        plt.plot(
            grp["budget"] * 100, grp["test_acc"], marker="o", label=model_name
        )
    plt.xlabel("Annotation Budget (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Annotation Budget")
    plt.legend()
    plt.tight_layout()
    _save(fig, save_dir, "accuracy_vs_budget")
    plt.show()


def plot_loss_vs_budget(
    models: list[ModelName] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Plot final test loss against annotation budget for each model.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> plot_loss_vs_budget()
    """
    df = _last_epoch(_load())
    if models:
        df = df[df["model"].isin(models)]

    fig = plt.figure(figsize=(7, 4))
    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("budget")
        plt.plot(
            grp["budget"] * 100, grp["test_loss"], marker="o", label=model_name
        )
    plt.xlabel("Annotation Budget (%)")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs. Annotation Budget")
    plt.legend()
    plt.tight_layout()
    _save(fig, save_dir, "loss_vs_budget")
    plt.show()


def plot_accuracy_vs_time(
    models: list[ModelName] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Scatter plot of final test accuracy vs. total training time per run.

    Each point is one (model, budget) pair. Budget percentage is annotated
    next to each point. Useful for comparing efficiency trade-offs.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> plot_accuracy_vs_time()
    """
    df = _last_epoch(_load())
    if models:
        df = df[df["model"].isin(models)]

    fig = plt.figure(figsize=(7, 4))
    for model_name, grp in df.groupby("model"):
        plt.scatter(
            grp["total_elapsed_time"] / 60, grp["test_acc"], label=model_name
        )
        for _, row in grp.iterrows():
            plt.annotate(
                f"{int(row['budget'] * 100)}%",
                (row["total_elapsed_time"] / 60, row["test_acc"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    plt.xlabel("Total Training Time (min)")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs. Training Time")
    plt.legend()
    plt.tight_layout()
    _save(fig, save_dir, "accuracy_vs_time")
    plt.show()


def plot_train_time(
    models: list[ModelName] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Grouped bar chart of average per-epoch training time by annotation budget.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    save_dir : str | None
        Directory to save the plot PNG. Default: None (no save).

    Example
    -------
    >>> plot_train_time()
    """
    df = _load()
    if models:
        df = df[df["model"].isin(models)]

    avg = df.groupby(["budget", "model"])["train_time"].mean().reset_index()
    pivot = avg.pivot(index="budget", columns="model", values="train_time")
    pivot.index = [f"{int(b * 100)}%" for b in pivot.index]

    ax = pivot.plot(kind="bar", figsize=(8, 4))
    fig = ax.get_figure()
    ax.set_xlabel("Annotation Budget")
    ax.set_ylabel("Avg. Train Time per Epoch (s)")
    ax.set_title("Training Time per Epoch by Budget")
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    _save(fig, save_dir, "train_time_per_budget")
    plt.show()


def plot_all(
    models: list[ModelName] | None = None,
    save_dir: str | None = None,
) -> None:
    """
    Run all result plots.

    Arguments
    ---------
    models : list[ModelName] | None
        Models to include. Default: all models in the CSV.
    save_dir : str | None
        Directory to save plot PNGs. Default: None (no save).

    Example
    -------
    >>> plot_all()
    >>> plot_all(models=["ResNet18_pretrained", "ResNet18_scratch"])
    >>> plot_all(save_dir="data/plots")
    """
    plot_accuracy_vs_budget(models, save_dir)
    plot_loss_vs_budget(models, save_dir)
    plot_accuracy_vs_time(models, save_dir)
    plot_train_time(models, save_dir)
    for metric in ("test_acc", "test_loss", "train_loss"):
        plot_epoch_curves(models, metric=metric, save_dir=save_dir)
