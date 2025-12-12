"""
Insecta DNA Sequence Classifier - Inference Script

Classifies DNA sequences into Order and Family taxonomic levels.

Usage:
    poetry run python inference.py --sequence "ATCGTACG..."
    poetry run python inference.py -s "ATCGTACG..."
"""

import argparse
import re

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from cnn.config import CONFIG
from cnn.dataset import sequence_to_image
from cnn.interpret import load_model


def clean_sequence(seq: str) -> str:
    """
    Clean DNA sequence by removing invalid characters.

    Args:
        seq: Raw DNA sequence (may contain invalid chars like ^, ?, newlines)

    Returns:
        Cleaned uppercase sequence with only valid DNA characters (ATCGN-)
    """
    # Remove whitespace and newlines
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    # Keep only valid DNA characters
    seq = re.sub(r"[^ATCGNatcgn\-]", "", seq)
    return seq.upper()


def predict_sequence(sequence: str) -> dict:
    """Run prediction on a single DNA sequence."""
    device = CONFIG["device"]

    # Load model
    model, checkpoint = load_model(device=device)

    # Try to get label_encoders from checkpoint, otherwise rebuild
    label_encoders = checkpoint.get("label_encoders", None)
    if label_encoders is None:
        # Rebuild label encoders from pre-filtered Insecta data
        train_df = pd.read_csv(CONFIG["csv_path"])
        label_encoders = {
            "order": LabelEncoder().fit(train_df["order"]),
            "family": LabelEncoder().fit(train_df["family"]),
        }

    # Clean sequence (remove invalid characters)
    seq = clean_sequence(sequence)
    original_len = len(sequence.replace(" ", "").replace("\n", ""))
    cleaned_len = len(seq)
    removed = original_len - cleaned_len

    # Convert to image tensor
    img = sequence_to_image(seq, CONFIG["img_size"])
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

        # Get top-3 predictions for each level (order + family)
        results = {}
        for level in ["order", "family"]:
            probs = torch.softmax(outputs[level], dim=1)[0]
            top_k = min(3, len(probs))
            top_probs, top_idx = probs.topk(top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_idx):
                name = label_encoders[level].inverse_transform([idx.item()])[0]
                predictions.append({"name": name, "confidence": prob.item()})

            results[level] = predictions

    return results, cleaned_len, removed


def format_output(results: dict, seq_length: int, removed: int, sequence: str) -> str:
    """Format prediction results as readable text."""
    # Clean for display
    display_seq = clean_sequence(sequence)

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════╗",
        "║         INSECTA DNA TAXONOMIC CLASSIFICATION RESULTS             ║",
        "╚══════════════════════════════════════════════════════════════════╝",
        "",
        f"  Sequence: {display_seq[:50]}{'...' if len(display_seq) > 50 else ''}",
        f"  Length:   {seq_length} bp",
    ]

    if removed > 0:
        lines.append(f"  Cleaned:  {removed} invalid characters removed")

    lines.extend([
        "",
        "─" * 70,
    ])

    # Results for each taxonomic level
    level_labels = {
        "order": "ORDER",
        "family": "FAMILY"
    }

    for level in ["order", "family"]:
        preds = results[level]
        top = preds[0]

        lines.append("")
        lines.append(f"  {level_labels[level]}")
        lines.append(f"  ├─ Prediction: {top['name']}")
        lines.append(f"  ├─ Confidence: {top['confidence']:.1%}")
        lines.append(f"  └─ Alternatives:")

        for alt in preds[1:]:
            lines.append(f"       • {alt['name']} ({alt['confidence']:.1%})")

    lines.append("")
    lines.append("─" * 70)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Classify Insecta DNA sequence into Order and Family"
    )
    parser.add_argument(
        "-s", "--sequence",
        type=str,
        required=True,
        help="DNA sequence (A, T, C, G characters)"
    )
    args = parser.parse_args()

    results, seq_length, removed = predict_sequence(args.sequence)
    output = format_output(results, seq_length, removed, args.sequence)
    print(output)


if __name__ == "__main__":
    main()
