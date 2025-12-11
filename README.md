# COI Taxonomic Classifier

DNA-based taxonomic classifier using the COI (Cytochrome Oxidase I) gene as a biological barcode. Classifies Arthropoda sequences into taxonomic hierarchy (class → order → family) using a multi-task CNN.

## Features

- **Sequence Alignment**: MUSCLE v5 alignment with taxonomic filtering
- **FFT Analysis**: Hypervariable region detection using spectral variance
- **CNN Classification**: Multi-task neural network for taxonomic prediction
- **Visualization**: Colored alignment PNGs and training metrics

## Quick Start

```bash
# Install dependencies
poetry install

# Train CNN classifier
poetry run python -m cnn.train

# Align sequences by taxonomic group
poetry run python main.py Arthropoda -n 50 --visualize --fft
```

## CNN Module

Multi-task CNN classifier that converts DNA sequences to 32×32 RGB images and predicts taxonomic levels simultaneously.

### Architecture

```
DNA Sequence → 32×32 RGB Image → 4 Conv Blocks → 3 Classification Heads
                                                  ├── Class (19 categories)
                                                  ├── Order (130 categories)
                                                  └── Family (1989 categories)
```

### Training

```bash
# Train with default config (1000 samples for testing)
poetry run python -m cnn.train

# Ctrl+C saves checkpoint automatically
```

### Configuration

Edit `cnn/config.py`:
- `sample_size`: Number of sequences (1000 for testing, 50000 for full training)
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size (default: 64)

### Output

```
output/
├── model_checkpoint.pt       # Best model weights
├── training_history.json     # Loss/accuracy per epoch
├── loss_curves.png           # Training curves
├── accuracy_curves.png       # Per-level accuracy
├── confusion_matrix_*.png    # Confusion matrices
└── classification_report.txt # Precision/recall/F1
```

## Data Pipeline

```
Raw MIDORI2 (1.7M sequences)
        ↓
[data_cleaning_pipeline.cpp]  # Dedup, split by level, filter outliers
        ↓
Cleaned datasets (data_clean_*.csv/fasta)
        ↓
[main.py]                     # Filter, align with MUSCLE
        ↓
[cnn/train.py]                # Train CNN classifier
```

## Requirements

- Python >= 3.13
- PyTorch, torchvision, scikit-learn
- MUSCLE v5: `brew install brewsci/bio/muscle`

## Project Structure

```
├── main.py                    # CLI for sequence alignment
├── fft.py                     # FFT hypervariable detection
├── visualize_alignment.py     # Alignment visualization
├── data_cleaning_pipeline.cpp # C++ data cleaning
├── cnn/                       # CNN module
│   ├── config.py              # Training configuration
│   ├── dataset.py             # Dataset and pixel conversion
│   ├── model.py               # MultiTaskCNN architecture
│   ├── train.py               # Training loop
│   └── visualize.py           # Metrics visualization
├── data/                      # Input datasets
└── output/                    # Results and models
```
