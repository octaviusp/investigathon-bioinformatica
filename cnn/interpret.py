"""
CNN Interpretation Module: Extract and analyze learned filters as DNA motifs.

This module provides tools to interpret what the CNN has learned:
1. Extract convolutional filters and convert them to DNA motif patterns
2. Visualize activation maps for specific sequences
3. Find which filters correlate with each taxonomic class
4. Generate biological hypotheses from learned patterns

The key insight: CNN filters learned on DNA→pixel images are essentially
motif detectors. A 3×3 filter on the first conv layer detects patterns
across 3 consecutive bases (considering the 2D arrangement).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from .config import COLOR_MAP, CONFIG
from .model import MultiTaskCNN


# Reverse color map: RGB → DNA base
RGB_TO_BASE = {v: k for k, v in COLOR_MAP.items()}


def load_model(model_path: Path = None, device: str = None) -> tuple[MultiTaskCNN, dict]:
    """
    Load trained model and metadata.

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if model_path is None:
        model_path = CONFIG["model_path"]
    if device is None:
        device = CONFIG["device"]

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # num_classes is a dict with 'order', 'family' keys (2-head model)
    num_classes_dict = checkpoint.get("num_classes", {})
    num_orders = num_classes_dict.get("order", 27)
    num_families = num_classes_dict.get("family", 917)

    # Get dropout from config in checkpoint or use default
    config = checkpoint.get("config", {})
    dropout = config.get("dropout", 0.5)

    model = MultiTaskCNN(
        num_orders=num_orders,
        num_families=num_families,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def extract_conv_filters(model: MultiTaskCNN) -> dict[str, np.ndarray]:
    """
    Extract convolutional filter weights from all layers.

    Returns:
        Dict mapping layer name to filter weights (out_channels, in_channels, H, W)
    """
    filters = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.detach().cpu().numpy()
            filters[name] = weights

    return filters


def filter_to_dna_motif(filter_weights: np.ndarray) -> dict:
    """
    Convert a single conv1 filter (3, 3, 3) to DNA motif interpretation.

    The filter has shape (in_channels=3, H=3, W=3) where:
    - 3 input channels = RGB
    - 3×3 = spatial receptive field

    Each position in the 3×3 grid corresponds to a base position.
    The RGB weights indicate which base the filter is "looking for".

    Returns:
        Dict with motif info: consensus, position_weights, confidence
    """
    # filter_weights shape: (3, 3, 3) = (RGB, H, W)
    # Reshape to (9 positions, 3 RGB channels)
    h, w = filter_weights.shape[1], filter_weights.shape[2]
    positions = h * w

    # Transpose to (H, W, RGB) then reshape to (positions, RGB)
    reshaped = filter_weights.transpose(1, 2, 0).reshape(positions, 3)

    # For each position, compute similarity to each base's color
    bases = ['A', 'T', 'C', 'G']
    base_colors = np.array([
        [1.0, 0.0, 0.0],    # A = Red
        [0.0, 0.0, 1.0],    # T = Blue
        [0.0, 1.0, 0.0],    # C = Green
        [1.0, 1.0, 0.0],    # G = Yellow
    ])

    # Normalize filter weights per position
    reshaped_norm = reshaped / (np.linalg.norm(reshaped, axis=1, keepdims=True) + 1e-8)

    # Compute dot product with each base color (cosine similarity)
    # Shape: (positions, 4 bases)
    similarities = reshaped_norm @ base_colors.T

    # Build consensus motif
    consensus = []
    position_weights = []

    for pos in range(positions):
        sims = similarities[pos]
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]

        # If similarity is low, mark as ambiguous
        if max_sim < 0.3:
            consensus.append('N')
        else:
            # Check if multiple bases have similar scores
            sorted_sims = np.sort(sims)[::-1]
            if sorted_sims[0] - sorted_sims[1] < 0.2:
                # Ambiguous: could be multiple bases
                top_bases = [bases[i] for i in range(4) if sims[i] > sorted_sims[1] - 0.1]
                consensus.append(f"[{''.join(sorted(top_bases))}]")
            else:
                consensus.append(bases[max_idx])

        position_weights.append({
            'A': float(sims[0]),
            'T': float(sims[1]),
            'C': float(sims[2]),
            'G': float(sims[3]),
        })

    return {
        'consensus': ''.join(consensus),
        'position_weights': position_weights,
        'confidence': float(np.mean(np.max(similarities, axis=1))),
        'raw_weights': filter_weights.tolist(),
    }


def analyze_all_conv1_filters(model: MultiTaskCNN) -> list[dict]:
    """
    Analyze all filters in the first convolutional layer.

    Returns:
        List of motif interpretations for each filter
    """
    filters = extract_conv_filters(model)
    conv1_weights = filters['conv1.conv']  # Shape: (32, 3, 3, 3)

    motifs = []
    for i in range(conv1_weights.shape[0]):
        filter_w = conv1_weights[i]  # (3, 3, 3) = (in_channels, H, W)
        motif = filter_to_dna_motif(filter_w)
        motif['filter_idx'] = i
        motifs.append(motif)

    return motifs


def get_activation_maps(
    model: MultiTaskCNN,
    image: torch.Tensor,
    layer_name: str = 'conv1',
) -> np.ndarray:
    """
    Get activation maps for a specific layer given an input image.

    Args:
        model: Trained model
        image: Input tensor (1, 3, 32, 32) or (3, 32, 32)
        layer_name: Which conv layer to extract from

    Returns:
        Activation maps as numpy array (num_filters, H, W)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if layer_name in name and isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    with torch.no_grad():
        model(image)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Get the activation
    key = f'{layer_name}.conv'
    if key in activations:
        return activations[key][0].cpu().numpy()

    return None


def compute_filter_class_correlation(
    model: MultiTaskCNN,
    dataloader,
    device: str = None,
    level: str = 'class',
) -> np.ndarray:
    """
    Compute correlation between each conv1 filter activation and each class.

    This reveals which filters are "responsible" for detecting each taxon.

    Args:
        model: Trained model
        dataloader: DataLoader with labeled samples
        device: Device to use
        level: Taxonomic level ('class', 'order', 'family')

    Returns:
        Correlation matrix (num_filters, num_classes)
    """
    if device is None:
        device = CONFIG["device"]

    model.eval()

    # Collect activations and labels
    all_activations = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)

        # Get conv1 activations
        acts = get_activation_maps(model, images, 'conv1')
        if acts is not None:
            # Global average pool the activation maps
            # acts shape: (batch, 32, 16, 16) after conv1
            pooled = acts.mean(axis=(1, 2))  # (32,) per sample
            all_activations.append(pooled)
            all_labels.append(labels[level].numpy())

    # Stack all
    activations = np.vstack(all_activations)  # (N, 32)
    labels = np.concatenate(all_labels)  # (N,)

    # Compute correlation for each filter with each class
    num_filters = activations.shape[1]
    num_classes = labels.max() + 1

    correlations = np.zeros((num_filters, num_classes))

    for c in range(num_classes):
        class_mask = (labels == c).astype(float)
        if class_mask.sum() < 2:
            continue

        for f in range(num_filters):
            filter_acts = activations[:, f]
            # Pearson correlation
            corr = np.corrcoef(filter_acts, class_mask)[0, 1]
            correlations[f, c] = corr if not np.isnan(corr) else 0

    return correlations


def plot_filter_motifs(motifs: list[dict], output_path: Path, top_n: int = 16):
    """
    Visualize the top N most confident filter motifs.
    """
    # Sort by confidence
    sorted_motifs = sorted(motifs, key=lambda x: x['confidence'], reverse=True)[:top_n]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()

    base_colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'gold', 'N': 'gray'}

    for ax, motif in zip(axes, sorted_motifs):
        # Create position weight matrix visualization
        weights = motif['position_weights']

        # Bar plot for each position
        x = np.arange(len(weights))
        width = 0.2

        for i, base in enumerate(['A', 'T', 'C', 'G']):
            values = [w[base] for w in weights]
            ax.bar(x + i * width, values, width, label=base, color=base_colors[base], alpha=0.8)

        ax.set_title(f"Filter #{motif['filter_idx']}\n{motif['consensus']}", fontsize=9)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([f'P{i+1}' for i in range(len(weights))], fontsize=7)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Weight', fontsize=8)

    # Add legend to last subplot
    axes[-1].legend(loc='upper right', fontsize=8)

    plt.suptitle('Conv1 Filter Motifs (DNA Pattern Detectors)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Filter motifs saved to {output_path}")


def plot_filter_class_heatmap(
    correlations: np.ndarray,
    class_names: list[str],
    output_path: Path,
    top_filters: int = 20,
):
    """
    Plot heatmap of filter-class correlations.
    """
    # Select top filters by max correlation
    max_corr = np.abs(correlations).max(axis=1)
    top_idx = np.argsort(max_corr)[-top_filters:]

    subset = correlations[top_idx]

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(subset, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f'Filter #{i}' for i in top_idx], fontsize=9)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('Taxonomic Class', fontsize=11)
    ax.set_ylabel('Conv1 Filter', fontsize=11)
    ax.set_title('Filter-Class Correlation\n(Which filters detect which taxa)', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Filter-class heatmap saved to {output_path}")


def plot_activation_map(
    image: torch.Tensor,
    activations: np.ndarray,
    sequence: str,
    output_path: Path,
    top_filters: int = 8,
):
    """
    Visualize activation maps for a specific sequence.

    Shows which parts of the sequence activated which filters.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # Original image
    ax = axes[0, 0]
    img_np = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img_np)
    ax.set_title('Input (DNA→Pixel)', fontsize=10)
    ax.axis('off')

    # Top activated filters
    mean_activation = activations.mean(axis=(1, 2))
    top_idx = np.argsort(mean_activation)[-top_filters:]

    for i, (ax, filt_idx) in enumerate(zip(axes.flatten()[1:], top_idx)):
        act_map = activations[filt_idx]
        im = ax.imshow(act_map, cmap='hot')
        ax.set_title(f'Filter #{filt_idx}\nMean: {mean_activation[filt_idx]:.3f}', fontsize=9)
        ax.axis('off')

    plt.suptitle(f'Activation Maps\nSequence: {sequence[:30]}...', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Activation map saved to {output_path}")


def generate_interpretation_report(
    model: MultiTaskCNN,
    output_dir: Path,
    class_names: list[str] = None,
) -> dict:
    """
    Generate a complete interpretation report.

    Returns:
        Dict with all analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CNN INTERPRETATION ANALYSIS")
    print("=" * 60)

    # 1. Extract and analyze filter motifs
    print("\n[1/3] Analyzing conv1 filters as DNA motifs...")
    motifs = analyze_all_conv1_filters(model)

    # Sort by confidence
    sorted_motifs = sorted(motifs, key=lambda x: x['confidence'], reverse=True)

    print(f"  Extracted {len(motifs)} filters")
    print(f"  Top 5 most confident motifs:")
    for m in sorted_motifs[:5]:
        print(f"    Filter #{m['filter_idx']}: {m['consensus']} (conf: {m['confidence']:.3f})")

    # Plot filter motifs
    plot_filter_motifs(motifs, output_dir / 'filter_motifs.png')

    # Save motifs to JSON
    motifs_json = output_dir / 'filter_motifs.json'
    with open(motifs_json, 'w') as f:
        json.dump(motifs, f, indent=2)
    print(f"  Motif data saved to {motifs_json}")

    # 2. Extract all layer filters for visualization
    print("\n[2/3] Extracting filter statistics...")
    filters = extract_conv_filters(model)

    filter_stats = {}
    for name, weights in filters.items():
        filter_stats[name] = {
            'shape': list(weights.shape),
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
        }
        print(f"  {name}: shape={weights.shape}, mean={weights.mean():.4f}, std={weights.std():.4f}")

    # 3. Summary
    print("\n[3/3] Generating summary...")

    report = {
        'motifs': motifs,
        'filter_stats': filter_stats,
        'num_filters_per_layer': {k: v['shape'][0] for k, v in filter_stats.items()},
    }

    # Save full report
    report_path = output_dir / 'interpretation_report.json'
    with open(report_path, 'w') as f:
        # Convert to serializable format
        serializable = {
            'filter_stats': filter_stats,
            'num_motifs': len(motifs),
            'top_motifs': [
                {'filter': m['filter_idx'], 'consensus': m['consensus'], 'confidence': m['confidence']}
                for m in sorted_motifs[:10]
            ],
        }
        json.dump(serializable, f, indent=2)

    print(f"\n  Full report saved to {report_path}")
    print("=" * 60)

    return report


def run_interpretation(model_path: Path = None):
    """
    Main entry point for interpretation analysis.
    """
    print("\nLoading model...")
    model, checkpoint = load_model(model_path)

    # Get class names if available
    class_names = checkpoint.get('class_labels', None)

    output_dir = CONFIG['output_dir'] / 'interpretation'

    report = generate_interpretation_report(model, output_dir, class_names)

    return report


if __name__ == "__main__":
    run_interpretation()
