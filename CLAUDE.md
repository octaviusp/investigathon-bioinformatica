# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DNA-based taxonomic classifier using the COI (Cytochrome Oxidase I) gene as a biological barcode. The system identifies species from ~650bp DNA sequences (alphabet: A, C, T, G).

## Three Core Problems

1. **Feature Selection**: Find optimal subsequence range [start, end] with maximum entropy/variance between species
2. **Data Representation**: Determine optimal K-mer size for vectorization (Bag of Words approach on DNA)
3. **Classification**: Train supervised model (RF/SVM/KNN) using optimized features

## Build & Run Commands

```bash
# Install dependencies (Poetry)
poetry install

# Run Python scripts
poetry run python <script.py>

# Run Jupyter notebooks
poetry run jupyter notebook
```

## Data Files

- **Training**: `MIDORI2...` - Labeled sequences with full taxonomy (Kingdom → Species)
- **Test**: `query.fasta` - Anonymous sequences to classify

## Architecture Pipeline

```
Raw FASTA → Feature Selection (entropy analysis) → K-mer Vectorization → PCA → Classifier → Taxonomy
```

## Key Technical Approaches

- **Feature Selection**: Column-wise standard deviation/entropy analysis
- **Vectorization**: K-mer frequency counting (like NLP n-grams)
- **Dimensionality Reduction**: PCA for visualization and cluster separation
- **Classification**: Supervised ML (Random Forest, SVM, KNN)
