#!/usr/bin/env python3
"""
COI Sequence Alignment CLI - MIDORI2 Dataset

Herramienta de línea de comandos para alinear secuencias COI (Cytochrome Oxidase I)
filtradas por grupo taxonómico usando MUSCLE.

Uso:
    poetry run python main.py <grupo> [opciones]

Ejemplos:
    poetry run python main.py Insecta -n 10 --visualize
    poetry run python main.py Arthropoda -n 50
    poetry run python main.py Coleoptera

El script:
    1. Auto-detecta el nivel taxonómico del grupo (phylum, class, order, etc.)
    2. Filtra secuencias del dataset limpio correspondiente
    3. Realiza subsampling si hay demasiadas secuencias
    4. Alinea con MUSCLE v5
    5. Opcionalmente genera visualización PNG con colores por nucleótido

Requisitos:
    - MUSCLE v5: brew install brewsci/bio/muscle
    - Datasets limpios en data/data_clean_*.csv y data/data_clean_*.fasta
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from visualize_alignment import visualize_alignment

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MUSCLE_PATH = "/opt/homebrew/bin/muscle"
MAX_SEQUENCES = 500

# Niveles taxonómicos (búsqueda de más específico a más general)
TAX_LEVELS = ["species", "genus", "family", "order", "class", "phylum"]


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================


def detect_level(group_name: str) -> tuple[str, Path] | tuple[None, None]:
    """
    Auto-detecta en qué nivel taxonómico está el grupo.

    Busca secuencialmente en cada CSV hasta encontrar el grupo.

    Args:
        group_name: Nombre del grupo taxonómico (ej: "Insecta")

    Returns:
        Tupla (nivel, ruta_csv) o (None, None) si no se encuentra
    """
    print(f"Buscando '{group_name}' en los datasets...")

    for level in TAX_LEVELS:
        csv_path = DATA_DIR / f"data_clean_{level}.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path, usecols=[level])
            if group_name in df[level].values:
                print(f"  Encontrado en nivel: {level}")
                return level, csv_path
        except Exception as e:
            print(f"  Error leyendo {csv_path}: {e}")
            continue

    return None, None


def filter_sequences(csv_path: Path, level: str, group_name: str) -> pd.DataFrame:
    """
    Filtra el CSV por grupo taxonómico.

    Args:
        csv_path: Ruta al archivo CSV
        level: Nivel taxonómico (columna a filtrar)
        group_name: Valor del grupo a buscar

    Returns:
        DataFrame filtrado
    """
    print(f"Cargando y filtrando {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df_filtered = df[df[level] == group_name]
    print(f"  Secuencias encontradas: {len(df_filtered):,}")
    return df_filtered


def extract_fasta_sequences(seq_ids: set, fasta_path: Path) -> list:
    """
    Extrae secuencias del FASTA que coincidan con los IDs.

    Args:
        seq_ids: Conjunto de IDs de secuencias a extraer
        fasta_path: Ruta al archivo FASTA

    Returns:
        Lista de SeqRecord
    """
    print(f"Extrayendo secuencias de {fasta_path.name}...")
    sequences = [rec for rec in SeqIO.parse(fasta_path, "fasta") if rec.id in seq_ids]
    print(f"  Secuencias extraídas: {len(sequences):,}")
    return sequences


def align_with_muscle(
    sequences: list, output_path: Path, max_seqs: int = MAX_SEQUENCES
) -> None:
    """
    Alinea secuencias con MUSCLE v5.

    Realiza subsampling aleatorio si hay más secuencias que max_seqs.

    Args:
        sequences: Lista de SeqRecord
        output_path: Ruta para guardar el alineamiento
        max_seqs: Número máximo de secuencias a alinear
    """
    if not os.path.exists(MUSCLE_PATH):
        print(f"Error: MUSCLE no encontrado en {MUSCLE_PATH}")
        print("Instalar con: brew install brewsci/bio/muscle")
        sys.exit(1)

    # Subsampling si es necesario
    if len(sequences) > max_seqs:
        print(f"Subsampling: {max_seqs} de {len(sequences):,} secuencias...")
        sequences = random.sample(sequences, max_seqs)

    # Archivo temporal
    temp_input = Path("temp_input.fasta")
    SeqIO.write(sequences, temp_input, "fasta")

    print(f"Alineando {len(sequences)} secuencias con MUSCLE...")

    try:
        cmd = [MUSCLE_PATH, "-align", str(temp_input), "-output", str(output_path)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Alineamiento completado: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error de MUSCLE: {e.stderr}")
        sys.exit(1)
    finally:
        if temp_input.exists():
            temp_input.unlink()


# =============================================================================
# CLI
# =============================================================================


def main():
    """Punto de entrada principal del CLI."""
    parser = argparse.ArgumentParser(
        description="Alinear secuencias COI por grupo taxonómico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py Insecta -n 10 -v   # 10 secuencias con visualización
  python main.py Arthropoda         # 500 secuencias (default)
  python main.py Coleoptera -n 100  # 100 secuencias
        """,
    )

    parser.add_argument(
        "grupo", help="Nombre del grupo taxonómico (ej: Insecta, Arthropoda)"
    )
    parser.add_argument(
        "-n",
        "--max-seqs",
        type=int,
        default=MAX_SEQUENCES,
        help=f"Máximo de secuencias a alinear (default: {MAX_SEQUENCES})",
    )
    parser.add_argument(
        "-o", "--output", help="Archivo de salida (default: output/{grupo}_aligned.fasta)"
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generar visualización PNG del alineamiento",
    )

    args = parser.parse_args()

    # Crear directorio de salida
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Detectar nivel taxonómico
    level, csv_path = detect_level(args.grupo)
    if level is None:
        print(f"Error: '{args.grupo}' no encontrado en ningún nivel taxonómico")
        print("Verifica el nombre o los archivos data/data_clean_*.csv")
        sys.exit(1)

    # 2. Filtrar CSV
    df = filter_sequences(csv_path, level, args.grupo)
    if len(df) == 0:
        print("No hay secuencias para alinear")
        sys.exit(0)

    # 3. Extraer secuencias del FASTA
    fasta_path = DATA_DIR / f"data_clean_{level}.fasta"
    seq_ids = set(df["seq_id"].values)
    sequences = extract_fasta_sequences(seq_ids, fasta_path)

    if len(sequences) == 0:
        print("Error: No se pudieron extraer secuencias del FASTA")
        sys.exit(1)

    # 4. Alinear
    output_path = (
        Path(args.output) if args.output else OUTPUT_DIR / f"{args.grupo}_aligned.fasta"
    )
    align_with_muscle(sequences, output_path, args.max_seqs)

    print(f"\nResultado: {output_path}")
    print(f"Secuencias alineadas: {min(len(sequences), args.max_seqs)}")

    # 5. Visualizar
    if args.visualize:
        png_path = OUTPUT_DIR / f"{args.grupo}_alignment.png"
        visualize_alignment(output_path, png_path, args.max_seqs)


if __name__ == "__main__":
    main()
