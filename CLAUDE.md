# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DNA-based taxonomic classifier using the COI (Cytochrome Oxidase I) gene as a biological barcode. The system identifies species from ~650bp DNA sequences (alphabet: A, C, T, G) using the MIDORI2 database (~1.7M sequences).

## Build & Run Commands

```bash
# Install dependencies
poetry install

# Main CLI - Align sequences by taxonomic group
poetry run python main.py <grupo> [opciones]
poetry run python main.py Insecta -n 10 --visualize --fft
poetry run python main.py Arthropoda -n 50

# Standalone modules
poetry run python fft.py alignment.fasta -w 50 -s 10    # FFT analysis
poetry run python visualize_alignment.py input.fasta    # Visualization

# CNN Training - Multi-task taxonomic classifier
poetry run python -m cnn.train                          # Train CNN model

# C++ data cleaning pipeline (run once to generate clean datasets)
g++ -O3 -std=c++17 -o data_cleaning_pipeline data_cleaning_pipeline.cpp
./data_cleaning_pipeline
```

## External Dependencies

- **MUSCLE v5**: Required for sequence alignment. Install: `brew install brewsci/bio/muscle`
- Path hardcoded: `/opt/homebrew/bin/muscle`

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ data_cleaning_  │────▶│ data_clean_  │────▶│    main.py      │
│ pipeline.cpp    │     │ {level}.csv  │     │ (CLI entrypoint)│
│                 │     │ {level}.fasta│     └────────┬────────┘
└─────────────────┘     └──────────────┘              │
                                                      ▼
                        ┌───────────────────────────────────────┐
                        │           Processing Pipeline          │
                        ├───────────────────────────────────────┤
                        │ 1. Auto-detect taxonomic level        │
                        │ 2. Filter sequences by group          │
                        │ 3. Align with MUSCLE                  │
                        │ 4. FFT hypervariable detection (opt)  │
                        │ 5. PNG visualization (opt)            │
                        └───────────────────────────────────────┘
```

## Key Modules

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point. Handles taxonomic detection, filtering, MUSCLE alignment |
| `fft.py` | FFT-based hypervariable region detection using sliding window spectral variance |
| `visualize_alignment.py` | Generates colored alignment PNGs (A=green, T=red, C=blue, G=orange) |
| `data_cleaning_pipeline.cpp` | C++ pipeline: dedup by hash, split by taxonomy level, filter outliers by length |
| `cnn/` | PyTorch CNN module for multi-task taxonomic classification |

## CNN Module (`cnn/`)

Multi-task CNN classifier for Arthropoda taxonomic classification.

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters, paths, training configuration |
| `dataset.py` | `ArthropodaDataset` class, DNA→pixel conversion |
| `model.py` | `MultiTaskCNN` architecture (4 conv blocks, 3 output heads) |
| `train.py` | Training loop with checkpoints, early stopping, progress logging |
| `visualize.py` | Loss curves, accuracy plots, confusion matrices |

**Architecture**: DNA sequence → 32×32 RGB image → CNN → 3 classification heads (class, order, family)

**Training**: `poetry run python -m cnn.train` (Ctrl+C saves checkpoint)

## Data Structure

- **Input**: `data/MIDORI2_UNIQ_NUC_GB268_CO1.{taxon,fasta}` - Raw MIDORI2 dataset
- **Cleaned**: `data/data_clean_{level}.{csv,fasta}` where level ∈ {species, genus, family, order, class, phylum}
- **Alignment Output**: `output/{grupo}_aligned.fasta`, `output/{grupo}_alignment.png`, `output/{grupo}_fft_profile.png`
- **CNN Output**: `output/model_checkpoint.pt`, `output/training_history.json`, `output/*_curves.png`

## Three Core Problems (Research Goals)

1. **Feature Selection**: Find optimal subsequence range [start, end] with maximum entropy/variance between species (FFT approach implemented)
2. **Data Representation**: DNA→pixel image conversion (A=Red, T=Blue, C=Green, G=Yellow) for CNN input
3. **Classification**: Multi-task CNN classifier for taxonomic hierarchy (class→order→family)

## Taxonomic Levels

Search order (most specific → most general): species → genus → family → order → class → phylum
