#!/usr/bin/env python3
"""
Alignment Visualization Module

Genera imágenes PNG de alineamientos de secuencias con colores por nucleótido.

Esquema de colores:
    A (Adenina)  = Verde   (#2ecc71)
    T (Timina)   = Rojo    (#e74c3c)
    C (Citosina) = Azul    (#3498db)
    G (Guanina)  = Naranja (#f39c12)
    - (Gap)      = Gris    (#ecf0f1)
    N (Unknown)  = Gris    (#95a5a6)

Uso standalone:
    poetry run python visualize_alignment.py alignment.fasta -o output.png

Uso como módulo:
    from visualize_alignment import visualize_alignment
    visualize_alignment(Path("input.fasta"), Path("output.png"))
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from matplotlib.colors import ListedColormap

# =============================================================================
# CONFIGURACIÓN DE COLORES
# =============================================================================

NUCLEOTIDE_COLORS = {
    "A": "#2ecc71",  # Verde
    "T": "#e74c3c",  # Rojo
    "C": "#3498db",  # Azul
    "G": "#f39c12",  # Naranja
    "-": "#ecf0f1",  # Gris claro (gaps)
    "N": "#95a5a6",  # Gris (desconocido)
}

CHAR_TO_NUM = {"A": 0, "T": 1, "C": 2, "G": 3, "-": 4, "N": 5}


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================


def visualize_alignment(
    fasta_path: Path, output_path: Path, max_seqs: int = 50
) -> None:
    """
    Genera imagen PNG del alineamiento con colores por nucleótido.

    Args:
        fasta_path: Ruta al archivo FASTA alineado
        output_path: Ruta donde guardar el PNG
        max_seqs: Máximo de secuencias a mostrar (default: 50)

    Returns:
        None. Guarda la imagen en output_path.
    """
    print(f"Generando visualización de {fasta_path}...")

    # Leer secuencias
    records = list(SeqIO.parse(fasta_path, "fasta"))[:max_seqs]

    if not records:
        print("  No hay secuencias para visualizar")
        return

    # Extraer secuencias y labels
    seqs = [str(r.seq).upper() for r in records]
    labels = [r.id[:25] for r in records]

    # Verificar longitudes
    seq_len = len(seqs[0])
    for i, s in enumerate(seqs):
        if len(s) != seq_len:
            print(f"  Warning: Secuencia {i} tiene longitud diferente")

    # Crear matriz numérica
    matrix = np.array([[CHAR_TO_NUM.get(c, 5) for c in s] for s in seqs])

    # Crear colormap
    colors = [
        NUCLEOTIDE_COLORS["A"],
        NUCLEOTIDE_COLORS["T"],
        NUCLEOTIDE_COLORS["C"],
        NUCLEOTIDE_COLORS["G"],
        NUCLEOTIDE_COLORS["-"],
        NUCLEOTIDE_COLORS["N"],
    ]
    cmap = ListedColormap(colors)

    # Calcular tamaño de figura
    n_seqs = len(records)
    fig_height = max(4, n_seqs * 0.4)
    fig_width = min(20, max(10, seq_len / 50))

    # Crear figura
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Mostrar matriz
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=5)

    # Configurar ejes
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Posición en el alineamiento", fontsize=10)
    ax.set_title(
        f"Alineamiento de secuencias ({n_seqs} seqs, {seq_len} bp)",
        fontsize=12,
        fontweight="bold",
    )

    # Leyenda
    legend_patches = [
        mpatches.Patch(color=NUCLEOTIDE_COLORS["A"], label="A (Adenina)"),
        mpatches.Patch(color=NUCLEOTIDE_COLORS["T"], label="T (Timina)"),
        mpatches.Patch(color=NUCLEOTIDE_COLORS["C"], label="C (Citosina)"),
        mpatches.Patch(color=NUCLEOTIDE_COLORS["G"], label="G (Guanina)"),
        mpatches.Patch(color=NUCLEOTIDE_COLORS["-"], label="- (Gap)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Visualización guardada: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI standalone para visualizar alineamientos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualizar alineamiento de secuencias con colores"
    )
    parser.add_argument("fasta", help="Archivo FASTA alineado")
    parser.add_argument("-o", "--output", help="Archivo PNG de salida")
    parser.add_argument(
        "-n",
        "--max-seqs",
        type=int,
        default=50,
        help="Máximo de secuencias a mostrar (default: 50)",
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"Error: No existe {fasta_path}")
        return

    output_path = Path(args.output) if args.output else fasta_path.with_suffix(".png")
    visualize_alignment(fasta_path, output_path, args.max_seqs)


if __name__ == "__main__":
    main()
