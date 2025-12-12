"""
Configuration for CNN training.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
# Use pre-filtered Insecta dataset (652K sequences, 27 orders, 917 families)
CSV_PATH = DATA_DIR / "data_clean_insecta.csv"
FASTA_PATH = DATA_DIR / "data_clean_insecta.fasta"

# Data configuration
CONFIG = {
    # Dataset (Insecta-only, pre-filtered)
    "csv_path": CSV_PATH,
    "fasta_path": FASTA_PATH,
    "sample_size": 200000,  # Sample from 652K Insecta sequences
    "img_size": 32,
    # Expected class counts (for reference)
    "num_orders": 27,
    "num_families": 917,
    # Splits
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    # Training
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 50,
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5,
    "dropout": 0.5,  # Increased from 0.4 to reduce overfitting
    # Loss weights per taxonomic level (order + family only)
    "loss_weights": {"order": 1.0, "family": 0.5},
    # Device: "auto" (CUDA > MPS > CPU), or "cuda", "mps", "cpu"
    "device": "auto",
    # Output
    "output_dir": OUTPUT_DIR,
    "model_path": OUTPUT_DIR / "model_checkpoint.pt",
    "history_path": OUTPUT_DIR / "training_history.json",
}

# Color mapping for DNA → RGB pixels
COLOR_MAP = {
    "A": (255, 0, 0),  # Red
    "T": (0, 0, 255),  # Blue
    "C": (0, 255, 0),  # Green
    "G": (255, 255, 0),  # Yellow
    "N": (128, 128, 128),  # Gray
    "-": (0, 0, 0),  # Black (padding)
}
