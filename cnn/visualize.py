"""
Visualization utilities for training metrics.

Generates:
- Loss curves (train/val)
- Accuracy curves per taxonomic level
- Confusion matrices
- Classification reports
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_history(
    history: dict,
    output_dir: Path,
    show: bool = False,
) -> None:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc_*', 'val_acc_*'
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # Accuracy curves (order + family for 2-head model)
    levels = [l for l in ["order", "family"] if f"train_acc_{l}" in history]
    if not levels:
        # Fallback for old 3-head model
        levels = [l for l in ["class", "order", "family"] if f"train_acc_{l}" in history]

    fig, axes = plt.subplots(1, len(levels), figsize=(7 * len(levels), 5))
    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        train_key = f"train_acc_{level}"
        val_key = f"val_acc_{level}"

        if train_key in history:
            ax.plot(epochs, history[train_key], "b-", label="Train", linewidth=2)
            ax.plot(epochs, history[val_key], "r-", label="Val", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_title(f"{level.capitalize()} Accuracy", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curves.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print(f"  Plots saved to {output_dir}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    max_labels: int = 30,
    show: bool = False,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        output_path: Path to save figure
        max_labels: Max labels to show (for readability)
        show: Whether to display plot
    """
    # Get unique labels actually present in the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(unique_labels)

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Normalize
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Filter labels to only those present
    display_labels = [labels[i] if i < len(labels) else str(i) for i in unique_labels]

    # Limit labels for readability
    if n_labels > max_labels:
        # Show top classes by frequency in y_true
        class_counts = np.bincount(y_true, minlength=n_labels)
        # Only consider indices that exist in unique_labels
        valid_counts = [(i, class_counts[idx]) for i, idx in enumerate(unique_labels) if idx < len(class_counts)]
        valid_counts.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in valid_counts[:max_labels]]
        top_indices.sort()  # Keep order
        cm_norm = cm_norm[np.ix_(top_indices, top_indices)]
        display_labels = [display_labels[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_labels, fontsize=8)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    level_name: str,
    output_path: Path,
) -> dict:
    """
    Generate and save classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        level_name: Taxonomic level name
        output_path: Path to save report

    Returns:
        Report dict with metrics
    """
    # Get unique labels actually present in the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    display_labels = [labels[i] if i < len(labels) else str(i) for i in unique_labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )

    # Save text report
    report_text = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=display_labels,
        zero_division=0,
    )

    with open(output_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"{level_name.upper()} CLASSIFICATION REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(report_text)
        f.write("\n")

    return report


def save_training_history(history: dict, output_path: Path) -> None:
    """Save training history to JSON."""
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(output_path: Path) -> dict:
    """Load training history from JSON."""
    with open(output_path) as f:
        return json.load(f)
