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

## Data Structure

- **Input**: `data/MIDORI2_UNIQ_NUC_GB268_CO1.{taxon,fasta}` - Raw MIDORI2 dataset
- **Cleaned**: `data/data_clean_{level}.{csv,fasta}` where level ∈ {species, genus, family, order, class, phylum}
- **Output**: `output/{grupo}_aligned.fasta`, `output/{grupo}_alignment.png`, `output/{grupo}_fft_profile.png`

## Three Core Problems (Research Goals)

1. **Feature Selection**: Find optimal subsequence range [start, end] with maximum entropy/variance between species (FFT approach implemented)
2. **Data Representation**: Determine optimal K-mer size for vectorization (Bag of Words on DNA)
3. **Classification**: Train supervised model (RF/SVM/KNN) using optimized features

## Taxonomic Levels

Search order (most specific → most general): species → genus → family → order → class → phylum
