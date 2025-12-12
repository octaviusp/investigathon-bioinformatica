"""
Inference module for CNN model predictions on custom input sequences.

Usage:
    poetry run python -m cnn.predict
    poetry run python -m cnn.predict --csv data/input.csv --fasta data/input.fasta
"""

from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder

from .config import CONFIG
from .dataset import sequence_to_image
from .interpret import load_model


def predict(
    csv_path: Path = None,
    fasta_path: Path = None,
    model_path: Path = None,
    device: str = None,
) -> pd.DataFrame:
    """
    Run predictions on input sequences.

    Args:
        csv_path: Path to CSV with seq_id and taxonomic labels
        fasta_path: Path to FASTA with sequences
        model_path: Path to model checkpoint
        device: Device to use

    Returns:
        DataFrame with predictions
    """
    if csv_path is None:
        csv_path = Path("data/input.csv")
    if fasta_path is None:
        fasta_path = Path("data/input.fasta")
    if device is None:
        device = CONFIG["device"]

    print(f"\nLoading model from {CONFIG['model_path']}...")
    model, checkpoint = load_model(model_path, device)

    # Get label encoders from checkpoint or rebuild from training data
    label_encoders = checkpoint.get("label_encoders", {})
    if not label_encoders:
        print("  Rebuilding label encoders from training data...")
        train_df = pd.read_csv(CONFIG["csv_path"])
        train_df = train_df[train_df["phylum"] == CONFIG["phylum_filter"]]
        label_encoders = {
            "class": LabelEncoder().fit(train_df["class"]),
            "order": LabelEncoder().fit(train_df["order"]),
            "family": LabelEncoder().fit(train_df["family"]),
        }

    # Load input data
    print(f"Loading input from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} sequences to predict")

    # Load sequences
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq).upper()

    # Predict
    results = []
    model.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            seq_id = row["seq_id"]
            seq = sequences.get(seq_id, "")

            if not seq:
                print(f"  Warning: No sequence found for {seq_id}")
                continue

            # Convert to image
            img = sequence_to_image(seq, CONFIG["img_size"])
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Forward pass
            outputs = model(img_tensor)

            # Get predictions (argmax)
            pred_class = outputs["class"].argmax(dim=1).item()
            pred_order = outputs["order"].argmax(dim=1).item()
            pred_family = outputs["family"].argmax(dim=1).item()

            # Get probabilities
            prob_class = torch.softmax(outputs["class"], dim=1).max().item()
            prob_order = torch.softmax(outputs["order"], dim=1).max().item()
            prob_family = torch.softmax(outputs["family"], dim=1).max().item()

            # Decode labels if encoders available
            if label_encoders:
                pred_class_name = label_encoders["class"].inverse_transform([pred_class])[0]
                pred_order_name = label_encoders["order"].inverse_transform([pred_order])[0]
                pred_family_name = label_encoders["family"].inverse_transform([pred_family])[0]
            else:
                pred_class_name = str(pred_class)
                pred_order_name = str(pred_order)
                pred_family_name = str(pred_family)

            results.append({
                "seq_id": seq_id,
                "true_class": row.get("class", ""),
                "pred_class": pred_class_name,
                "prob_class": prob_class,
                "true_order": row.get("order", ""),
                "pred_order": pred_order_name,
                "prob_order": prob_order,
                "true_family": row.get("family", ""),
                "pred_family": pred_family_name,
                "prob_family": prob_family,
            })

    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CNN predictions on input sequences")
    parser.add_argument("--csv", type=Path, default=Path("data/input.csv"), help="Input CSV path")
    parser.add_argument("--fasta", type=Path, default=Path("data/input.fasta"), help="Input FASTA path")
    args = parser.parse_args()

    results = predict(csv_path=args.csv, fasta_path=args.fasta)

    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    for _, row in results.iterrows():
        print(f"\n{row['seq_id']}")
        print(f"  Class:  {row['pred_class']:20s} (conf: {row['prob_class']:.2%}) | True: {row['true_class']}")
        print(f"  Order:  {row['pred_order']:20s} (conf: {row['prob_order']:.2%}) | True: {row['true_order']}")
        print(f"  Family: {row['pred_family']:20s} (conf: {row['prob_family']:.2%}) | True: {row['true_family']}")

    # Summary
    print("\n" + "-" * 80)
    if "true_class" in results.columns and results["true_class"].notna().any():
        class_correct = (results["pred_class"] == results["true_class"]).sum()
        order_correct = (results["pred_order"] == results["true_order"]).sum()
        family_correct = (results["pred_family"] == results["true_family"]).sum()
        total = len(results)
        print(f"Accuracy: Class {class_correct}/{total}, Order {order_correct}/{total}, Family {family_correct}/{total}")

    # Save results
    output_path = Path("output/predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
