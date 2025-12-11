"""
Configuration for CNN training.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CSV_PATH = DATA_DIR / "data_clean_phylum.csv"
FASTA_PATH = DATA_DIR / "data_clean_phylum.fasta"

# Data configuration
CONFIG = {
    # Dataset
    "csv_path": CSV_PATH,
    "fasta_path": FASTA_PATH,
    "phylum_filter": "Arthropoda",
    "sample_size": 1000,  # Small for testing, use 50000 for full training
    "img_size": 32,
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
    # Loss weights per taxonomic level
    "loss_weights": {"class": 1.0, "order": 1.0, "family": 0.5},
    # Device (mps for Apple Silicon, cuda for NVIDIA, cpu for fallback)
    "device": "mps",
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
