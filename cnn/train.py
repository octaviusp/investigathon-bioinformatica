"""
Training script for Insecta Multi-Task CNN (Order + Family).

Usage:
    poetry run python -m cnn.train
"""

import signal
import sys
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

# Global state for interrupt handling
_training_state = {
    "model": None,
    "optimizer": None,
    "history": None,
    "epoch": 0,
    "best_model_state": None,
    "config": None,
    "num_classes": None,
    "interrupted": False,
}


def _save_checkpoint(reason: str = "checkpoint"):
    """Save current training state to checkpoint."""
    state = _training_state
    if state["model"] is None or state["config"] is None:
        return

    output_dir = Path(state["config"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_epoch_{state['epoch']}.pt"

    torch.save(
        {
            "epoch": state["epoch"],
            "model_state_dict": state["model"].state_dict(),
            "optimizer_state_dict": state["optimizer"].state_dict(),
            "best_model_state": state["best_model_state"],
            "history": state["history"],
            "num_classes": state["num_classes"],
            "config": state["config"],
        },
        checkpoint_path,
    )

    # Also save history
    if state["history"]:
        save_training_history(state["history"], state["config"]["history_path"])

    print(f"\n[CHECKPOINT] Saved: {checkpoint_path} ({reason})")
    return checkpoint_path


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n\n{'='*60}")
    print("INTERRUPT RECEIVED - Saving checkpoint...")
    print(f"{'='*60}")

    _training_state["interrupted"] = True
    checkpoint_path = _save_checkpoint("interrupted")

    if checkpoint_path:
        print(f"\nTraining interrupted at epoch {_training_state['epoch']}")
        print(f"Resume with: checkpoint={checkpoint_path}")

    sys.exit(0)


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
    epoch: int = 0,
    total_epochs: int = 0,
) -> tuple[float, dict]:
    """
    Train for one epoch with progress logging.

    Returns:
        (average_loss, accuracy_dict)
    """
    model.train()
    total_loss = 0.0
    correct = {"order": 0, "family": 0}
    total = 0
    num_batches = len(loader)
    log_interval = max(1, num_batches // 5)  # Log ~5 times per epoch

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        outputs = model(images)
        loss = multi_task_loss(outputs, targets, loss_weights)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Compute accuracy (order + family only)
        for level in ["order", "family"]:
            preds = outputs[level].argmax(dim=1)
            correct[level] += (preds == targets[level]).sum().item()

        # Progress log
        if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
            progress = 100.0 * (batch_idx + 1) / num_batches
            current_loss = total_loss / total
            current_acc = {level: correct[level] / total for level in correct}
            print(
                f"  [{epoch}/{total_epochs}] Batch {batch_idx+1:4d}/{num_batches} ({progress:5.1f}%) | "
                f"Loss: {current_loss:.4f} | "
                f"Acc: O={current_acc['order']:.3f} F={current_acc['family']:.3f}"
            )

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
    correct = {"order": 0, "family": 0}
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in labels.items()}

            outputs = model(images)
            loss = multi_task_loss(outputs, targets, loss_weights)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            for level in ["order", "family"]:
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
        for level in ["order", "family"]
    }

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            for level in ["order", "family"]:
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

    # Register signal handlers for graceful interrupt
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("=" * 60)
    print("INSECTA TAXONOMIC CLASSIFICATION CNN (Order + Family)")
    print("=" * 60)
    print("Press Ctrl+C to save checkpoint and exit gracefully")

    # Setup device
    device = get_device(config["device"])
    print(f"\nDevice: {device}")

    # Prepare data (using pre-filtered Insecta dataset)
    print("\n--- Preparing Data ---")
    train_dataset, val_dataset, test_dataset, label_encoders, num_classes = prepare_data(
        csv_path=config["csv_path"],
        fasta_path=config["fasta_path"],
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

    # Create model (2-head: order + family)
    print("\n--- Creating Model ---")
    model = MultiTaskCNN(
        num_orders=num_classes["order"],
        num_families=num_classes["family"],
        dropout=config.get("dropout", 0.5),
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

    # Training history (order + family only)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_order": [],
        "train_acc_family": [],
        "val_acc_order": [],
        "val_acc_family": [],
    }

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Update global state for interrupt handling
    _training_state.update({
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "config": config,
        "num_classes": num_classes,
    })

    # Training loop
    print("\n--- Training ---")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        _training_state["epoch"] = epoch

        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        print("\n[TRAIN]")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, config["loss_weights"],
            epoch=epoch, total_epochs=config["epochs"]
        )

        # Validate
        print("\n[VALIDATE]")
        val_loss, val_acc = validate(
            model, val_loader, device, config["loss_weights"]
        )

        # Update scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for level in ["order", "family"]:
            history[f"train_acc_{level}"].append(train_acc[level])
            history[f"val_acc_{level}"].append(val_acc[level])

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / epoch) * (config["epochs"] - epoch)

        # Print epoch summary
        print(f"\n[EPOCH {epoch} SUMMARY]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc - Order: {train_acc['order']:.4f} | Family: {train_acc['family']:.4f}")
        print(f"  Val Acc   - Order: {val_acc['order']:.4f} | Family: {val_acc['family']:.4f}")
        print(f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min")

        if new_lr != old_lr:
            print(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            _training_state["best_model_state"] = best_model_state
            print(f"  *** New best model (val_loss={best_val_loss:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['early_stopping_patience']})")
            if patience_counter >= config["early_stopping_patience"]:
                print(f"\n[EARLY STOPPING] No improvement for {config['early_stopping_patience']} epochs")
                break

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            _save_checkpoint(f"epoch_{epoch}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time / 60:.1f} minutes")
    print(f"{'='*60}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # Save model with label_encoders for inference
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "config": config,
            "label_encoders": label_encoders,  # Save for inference decoding
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

    # Classification reports and confusion matrices (order + family)
    report_path = output_dir / "classification_report.txt"
    if report_path.exists():
        report_path.unlink()

    # Order - confusion matrix + report
    y_true = results["order"]["y_true"]
    y_pred = results["order"]["y_pred"]
    labels = results["order"]["labels"]
    acc = (y_true == y_pred).mean()
    print(f"  Order accuracy: {acc:.4f}")

    plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        "Order Confusion Matrix",
        output_dir / "confusion_matrix_order.png",
    )

    save_classification_report(
        y_true,
        y_pred,
        labels,
        "order",
        report_path,
    )

    # Family (report only, too many classes for confusion matrix)
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
