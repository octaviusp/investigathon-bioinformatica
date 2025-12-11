"""
Training script for Multi-Task CNN.

Usage:
    poetry run python -m cnn.train
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import CONFIG
from .dataset import prepare_data
from .model import MultiTaskCNN, multi_task_loss
from .visualize import (
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    save_training_history,
)


def get_device(preferred: str = "mps") -> torch.device:
    """Get available device (MPS > CUDA > CPU)."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict,
) -> tuple[float, dict]:
    """
    Train for one epoch.

    Returns:
        (average_loss, accuracy_dict)
    """
    model.train()
    total_loss = 0.0
    correct = {"class": 0, "order": 0, "family": 0}
    total = 0

    for images, labels in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        outputs = model(images)
        loss = multi_task_loss(outputs, targets, loss_weights)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Compute accuracy
        for level in ["class", "order", "family"]:
            preds = outputs[level].argmax(dim=1)
            correct[level] += (preds == targets[level]).sum().item()

    avg_loss = total_loss / total
    accuracy = {level: correct[level] / total for level in correct}

    return avg_loss, accuracy


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict,
) -> tuple[float, dict]:
    """
    Validate model.

    Returns:
        (average_loss, accuracy_dict)
    """
    model.eval()
    total_loss = 0.0
    correct = {"class": 0, "order": 0, "family": 0}
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in labels.items()}

            outputs = model(images)
            loss = multi_task_loss(outputs, targets, loss_weights)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            for level in ["class", "order", "family"]:
                preds = outputs[level].argmax(dim=1)
                correct[level] += (preds == targets[level]).sum().item()

    avg_loss = total_loss / total
    accuracy = {level: correct[level] / total for level in correct}

    return avg_loss, accuracy


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_encoders: dict,
) -> dict:
    """
    Evaluate model on test set and collect predictions.

    Returns:
        Dict with y_true and y_pred for each level
    """
    model.eval()
    results = {
        level: {"y_true": [], "y_pred": []}
        for level in ["class", "order", "family"]
    }

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            for level in ["class", "order", "family"]:
                preds = outputs[level].argmax(dim=1).cpu().numpy()
                true = labels[level].numpy()
                results[level]["y_true"].extend(true)
                results[level]["y_pred"].extend(preds)

    # Convert to numpy
    for level in results:
        results[level]["y_true"] = np.array(results[level]["y_true"])
        results[level]["y_pred"] = np.array(results[level]["y_pred"])
        results[level]["labels"] = label_encoders[level].classes_.tolist()

    return results


def train(
    config: dict = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Main training function.

    Args:
        config: Configuration dict (uses CONFIG if None)

    Returns:
        (trained_model, training_history)
    """
    if config is None:
        config = CONFIG

    print("=" * 60)
    print("ARTHROPODA TAXONOMIC CLASSIFICATION CNN")
    print("=" * 60)

    # Setup device
    device = get_device(config["device"])
    print(f"\nDevice: {device}")

    # Prepare data
    print("\n--- Preparing Data ---")
    train_dataset, val_dataset, test_dataset, label_encoders, num_classes = prepare_data(
        csv_path=config["csv_path"],
        fasta_path=config["fasta_path"],
        phylum_filter=config["phylum_filter"],
        sample_size=config["sample_size"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        img_size=config["img_size"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    print("\n--- Creating Model ---")
    model = MultiTaskCNN(
        num_classes=num_classes["class"],
        num_orders=num_classes["order"],
        num_families=num_classes["family"],
    )
    model = model.to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["lr_scheduler_patience"],
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_class": [],
        "train_acc_order": [],
        "train_acc_family": [],
        "val_acc_class": [],
        "val_acc_order": [],
        "val_acc_family": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training loop
    print("\n--- Training ---")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, config["loss_weights"]
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, device, config["loss_weights"]
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for level in ["class", "order", "family"]:
            history[f"train_acc_{level}"].append(train_acc[level])
            history[f"val_acc_{level}"].append(val_acc[level])

        epoch_time = time.time() - epoch_start

        # Print progress
        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: C={val_acc['class']:.3f} O={val_acc['order']:.3f} F={val_acc['family']:.3f} | "
            f"{epoch_time:.1f}s"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # Save model
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "config": config,
        },
        config["model_path"],
    )
    print(f"\nModel saved: {config['model_path']}")

    # Save history
    save_training_history(history, config["history_path"])
    print(f"History saved: {config['history_path']}")

    # Plot training curves
    print("\n--- Generating Plots ---")
    plot_training_history(history, output_dir)

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    results = evaluate(model, test_loader, device, label_encoders)

    # Classification reports and confusion matrices
    report_path = output_dir / "classification_report.txt"
    if report_path.exists():
        report_path.unlink()

    for level in ["class", "order"]:  # Skip family (too many classes for confusion matrix)
        y_true = results[level]["y_true"]
        y_pred = results[level]["y_pred"]
        labels = results[level]["labels"]

        # Accuracy
        acc = (y_true == y_pred).mean()
        print(f"  {level.capitalize()} accuracy: {acc:.4f}")

        # Confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            f"{level.capitalize()} Confusion Matrix",
            output_dir / f"confusion_matrix_{level}.png",
        )

        # Classification report
        save_classification_report(
            y_true,
            y_pred,
            labels,
            level,
            report_path,
        )

    # Family (report only, no confusion matrix)
    y_true = results["family"]["y_true"]
    y_pred = results["family"]["y_pred"]
    labels = results["family"]["labels"]
    acc = (y_true == y_pred).mean()
    print(f"  Family accuracy: {acc:.4f}")
    save_classification_report(y_true, y_pred, labels, "family", report_path)

    print(f"\nReports saved: {report_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train()
