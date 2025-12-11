"""
CNN Module for Arthropoda Taxonomic Classification

Multi-task CNN that classifies DNA sequences into taxonomic levels:
- Class (19 categories)
- Order (130 categories)
- Family (1,989 categories)

Usage:
    poetry run python -m cnn.train
"""

from .config import CONFIG
from .dataset import ArthropodaDataset, sequence_to_image
from .model import MultiTaskCNN

__all__ = ["CONFIG", "ArthropodaDataset", "sequence_to_image", "MultiTaskCNN"]
