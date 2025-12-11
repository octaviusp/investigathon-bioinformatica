"""
Dataset class for Arthropoda DNA sequences.

Handles:
- Loading and filtering Arthropoda from phylum data
- Stratified sampling
- DNA → pixel image conversion
- Label encoding for class/order/family
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from .config import COLOR_MAP


def sequence_to_image(seq: str, size: int = 32) -> np.ndarray:
    """
    Convert DNA sequence to size×size RGB image.

    Each nucleotide becomes one pixel:
    - A = Red (255, 0, 0)
    - T = Blue (0, 0, 255)
    - C = Green (0, 255, 0)
    - G = Yellow (255, 255, 0)
    - N = Gray (128, 128, 128)
    - - = Black (0, 0, 0) for padding

    Args:
        seq: DNA sequence string
        size: Image dimension (size×size pixels)

    Returns:
        RGB image array of shape (size, size, 3)
    """
    pixels = size * size
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for i, base in enumerate(seq[:pixels]):
        row, col = i // size, i % size
        img[row, col] = COLOR_MAP.get(base.upper(), (0, 0, 0))

    return img


class ArthropodaDataset(Dataset):
    """
    PyTorch Dataset for Arthropoda DNA sequences.

    Loads sequences from CSV+FASTA, filters Arthropoda, samples stratified,
    and provides (image, labels) pairs.
    """

    def __init__(
        self,
        csv_path: Path,
        fasta_path: Path,
        seq_ids: list[str] = None,
        labels_df: pd.DataFrame = None,
        label_encoders: dict = None,
        img_size: int = 32,
        transform=None,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with taxonomic labels
            fasta_path: Path to FASTA with sequences
            seq_ids: List of sequence IDs to include (if None, loads all)
            labels_df: Pre-filtered DataFrame with labels
            label_encoders: Dict of LabelEncoder for each level
            img_size: Output image size
            transform: Optional torchvision transforms
        """
        self.csv_path = Path(csv_path)
        self.fasta_path = Path(fasta_path)
        self.img_size = img_size
        self.transform = transform

        if seq_ids is not None and labels_df is not None:
            # Use pre-split data
            self.seq_ids = seq_ids
            self.labels_df = labels_df.set_index("seq_id").loc[seq_ids].reset_index()
            self.label_encoders = label_encoders
        else:
            raise ValueError("Must provide seq_ids, labels_df, and label_encoders")

        # Load sequences from FASTA
        self.sequences = self._load_sequences()

        # Encode labels
        self.class_labels = self.label_encoders["class"].transform(
            self.labels_df["class"].values
        )
        self.order_labels = self.label_encoders["order"].transform(
            self.labels_df["order"].values
        )
        self.family_labels = self.label_encoders["family"].transform(
            self.labels_df["family"].values
        )

    def _load_sequences(self) -> dict:
        """Load sequences from FASTA for the given seq_ids."""
        seq_id_set = set(self.seq_ids)
        sequences = {}

        for record in SeqIO.parse(self.fasta_path, "fasta"):
            if record.id in seq_id_set:
                sequences[record.id] = str(record.seq).upper()
                if len(sequences) == len(seq_id_set):
                    break

        return sequences

    def __len__(self) -> int:
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> tuple:
        seq_id = self.seq_ids[idx]
        seq = self.sequences.get(seq_id, "")

        # Convert to image
        img = sequence_to_image(seq, self.img_size)

        # To tensor: HWC → CHW, normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.transform:
            img = self.transform(img)

        labels = {
            "class": torch.tensor(self.class_labels[idx], dtype=torch.long),
            "order": torch.tensor(self.order_labels[idx], dtype=torch.long),
            "family": torch.tensor(self.family_labels[idx], dtype=torch.long),
        }

        return img, labels


def prepare_data(
    csv_path: Path,
    fasta_path: Path,
    phylum_filter: str = "Arthropoda",
    sample_size: int = 500,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    img_size: int = 32,
) -> tuple:
    """
    Prepare train/val/test datasets.

    Args:
        csv_path: Path to CSV
        fasta_path: Path to FASTA
        phylum_filter: Phylum to filter (e.g., "Arthropoda")
        sample_size: Number of sequences to sample
        train_split, val_split, test_split: Split ratios
        random_state: Random seed
        img_size: Image size

    Returns:
        (train_dataset, val_dataset, test_dataset, label_encoders, num_classes)
    """
    print(f"Loading data from {csv_path}...")

    # Load and filter
    df = pd.read_csv(csv_path)
    df_filtered = df[df["phylum"] == phylum_filter].copy()
    print(f"  {phylum_filter} sequences: {len(df_filtered):,}")

    # Remove rare classes (need at least 2 members for stratified split)
    class_counts = df_filtered["class"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_filtered = df_filtered[df_filtered["class"].isin(valid_classes)]
    print(f"  After removing rare classes: {len(df_filtered):,}")

    # Stratified sample by class
    if len(df_filtered) > sample_size:
        print(f"  Stratified sampling {sample_size:,} sequences...")
        df_sampled, _ = train_test_split(
            df_filtered,
            train_size=sample_size,
            stratify=df_filtered["class"],
            random_state=random_state,
        )
    else:
        df_sampled = df_filtered.copy()

    # After sampling, ensure each class has enough samples for stratified split
    # Need at least 3 samples per class (1 for each split)
    min_samples_per_class = 3
    class_counts = df_sampled["class"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_sampled = df_sampled[df_sampled["class"].isin(valid_classes)].copy()

    print(f"  Sampled: {len(df_sampled):,} sequences")

    # Create label encoders
    label_encoders = {
        "class": LabelEncoder().fit(df_sampled["class"]),
        "order": LabelEncoder().fit(df_sampled["order"]),
        "family": LabelEncoder().fit(df_sampled["family"]),
    }

    num_classes = {
        "class": len(label_encoders["class"].classes_),
        "order": len(label_encoders["order"].classes_),
        "family": len(label_encoders["family"].classes_),
    }

    print(f"  Classes: {num_classes['class']}, Orders: {num_classes['order']}, Families: {num_classes['family']}")

    # Train/val/test split (use simple random split if stratified fails)
    try:
        train_df, temp_df = train_test_split(
            df_sampled,
            train_size=train_split,
            stratify=df_sampled["class"],
            random_state=random_state,
        )
        val_ratio = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            stratify=temp_df["class"],
            random_state=random_state,
        )
    except ValueError:
        # Fallback to non-stratified split
        print("  Warning: Using non-stratified split due to class imbalance")
        train_df, temp_df = train_test_split(
            df_sampled,
            train_size=train_split,
            random_state=random_state,
        )
        val_ratio = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=random_state,
        )

    print(f"  Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Create datasets
    train_dataset = ArthropodaDataset(
        csv_path,
        fasta_path,
        seq_ids=train_df["seq_id"].tolist(),
        labels_df=df_sampled,
        label_encoders=label_encoders,
        img_size=img_size,
    )

    val_dataset = ArthropodaDataset(
        csv_path,
        fasta_path,
        seq_ids=val_df["seq_id"].tolist(),
        labels_df=df_sampled,
        label_encoders=label_encoders,
        img_size=img_size,
    )

    test_dataset = ArthropodaDataset(
        csv_path,
        fasta_path,
        seq_ids=test_df["seq_id"].tolist(),
        labels_df=df_sampled,
        label_encoders=label_encoders,
        img_size=img_size,
    )

    return train_dataset, val_dataset, test_dataset, label_encoders, num_classes
