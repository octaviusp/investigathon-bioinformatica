#!/usr/bin/env python3
"""
FFT Analysis Module for Hypervariable Region Detection

Detecta regiones hipervariables en alineamientos de secuencias usando
análisis de Transformada de Fourier con ventana deslizante.

El algoritmo:
1. Convierte cada secuencia a señal numérica (A=1, C=2, G=3, T=4)
2. Aplica FFT en ventanas deslizantes
3. Calcula la varianza del espectro de potencia entre secuencias
4. Identifica la región con máxima varianza (= máxima diferencia entre individuos)

Uso standalone:
    poetry run python fft.py alignment.fasta

Uso como módulo:
    from fft import find_hypervariable_region
    result = find_hypervariable_region(Path("alignment.fasta"))
    print(f"Región: [{result['start']}, {result['end']}]")
"""

from pathlib import Path

import numpy as np
from Bio import SeqIO
from scipy.fft import fft


# =============================================================================
# FUNCIONES DE CONVERSIÓN
# =============================================================================


def dna_to_numerical(seq_str: str, length_fixed: int) -> np.ndarray:
    """
    Convierte secuencia de ADN a señal numérica con padding/trimming.

    Args:
        seq_str: Secuencia de ADN (string)
        length_fixed: Longitud objetivo

    Returns:
        Array numpy de longitud length_fixed
    """
    mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0, "-": 0}
    signal = [mapping.get(base, 0) for base in seq_str.upper()]

    if len(signal) > length_fixed:
        signal = signal[:length_fixed]
    else:
        signal += [0] * (length_fixed - len(signal))

    return np.array(signal)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================


def find_hypervariable_region(
    fasta_path: Path,
    window_size: int = 50,
    step: int = 10,
    n_sample: int = 200,
) -> dict | None:
    """
    Detecta la región hipervariable usando FFT con ventana deslizante.

    Calcula la varianza espectral entre secuencias en cada ventana.
    La región con mayor varianza indica donde hay más diferencias
    entre individuos.

    Args:
        fasta_path: Ruta al archivo FASTA alineado
        window_size: Tamaño de la ventana de análisis (bp)
        step: Paso de la ventana deslizante (bp)
        n_sample: Máximo de secuencias a analizar

    Returns:
        Dict con 'start', 'end', 'variance', 'seq_length' o None si falla
    """
    print(f"  Analizando {fasta_path.name}...")

    # 1. Cargar secuencias
    sequences_raw = list(SeqIO.parse(fasta_path, "fasta"))
    if not sequences_raw:
        print("  Error: No hay secuencias en el archivo")
        return None

    # 2. Muestreo si hay demasiadas
    if len(sequences_raw) > n_sample:
        import random

        sequences = random.sample(sequences_raw, n_sample)
    else:
        sequences = sequences_raw

    # 3. Calcular longitud de referencia (mediana)
    lengths = [len(s) for s in sequences]
    fixed_len = int(np.median(lengths))
    print(f"  Longitud de referencia: {fixed_len} bp | Ventana: {window_size} bp")

    if fixed_len < window_size:
        print(f"  Error: Secuencias muy cortas ({fixed_len} < {window_size})")
        return None

    # 4. Convertir a matriz numérica
    matrix_signals = np.zeros((len(sequences), fixed_len))
    for i, seq in enumerate(sequences):
        matrix_signals[i, :] = dna_to_numerical(str(seq.seq), fixed_len)

    # 5. Sliding Window FFT
    variance_profile = []
    positions = []

    for start_idx in range(0, fixed_len - window_size, step):
        end_idx = start_idx + window_size

        # Extraer bloque de todas las secuencias
        block = matrix_signals[:, start_idx:end_idx]

        # Aplicar FFT a cada secuencia
        fft_block = fft(block, axis=1)

        # Espectro de potencia
        power_spectrum = np.abs(fft_block) ** 2

        # Varianza entre secuencias (suma sobre todas las frecuencias)
        local_variance = np.sum(np.var(power_spectrum, axis=0))

        variance_profile.append(local_variance)
        positions.append(start_idx)

    if not positions:
        print("  Error: No se pudieron calcular ventanas")
        return None

    # 6. Encontrar máximo
    max_idx = np.argmax(variance_profile)
    max_pos_start = positions[max_idx]
    max_pos_end = max_pos_start + window_size
    max_variance = variance_profile[max_idx]

    return {
        "start": max_pos_start,
        "end": max_pos_end,
        "variance": float(max_variance),
        "seq_length": fixed_len,
        "profile": variance_profile,
        "positions": positions,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI standalone para análisis FFT."""
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Detectar regiones hipervariables con FFT"
    )
    parser.add_argument("fasta", help="Archivo FASTA alineado")
    parser.add_argument(
        "-w", "--window", type=int, default=50, help="Tamaño de ventana (default: 50)"
    )
    parser.add_argument(
        "-s", "--step", type=int, default=10, help="Paso de ventana (default: 10)"
    )
    parser.add_argument(
        "-n", "--samples", type=int, default=200, help="Máximo de secuencias (default: 200)"
    )
    parser.add_argument(
        "-o", "--output", help="Archivo PNG para guardar gráfico (opcional)"
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"Error: No existe {fasta_path}")
        return

    result = find_hypervariable_region(
        fasta_path,
        window_size=args.window,
        step=args.step,
        n_sample=args.samples,
    )

    if result:
        print(f"\nRegión hipervariable: [{result['start']}, {result['end']}]")
        print(f"Varianza máxima: {result['variance']:.2f}")

        # Graficar
        plt.figure(figsize=(12, 5))
        plt.plot(result["positions"], result["profile"], color="purple")
        plt.axvspan(
            result["start"],
            result["end"],
            color="red",
            alpha=0.3,
            label=f"Hipervariable [{result['start']}-{result['end']}]",
        )
        plt.xlabel("Posición en la Secuencia (bp)")
        plt.ylabel("Varianza Espectral")
        plt.title("Detección de Regiones Hipervariables con FFT")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Gráfico guardado: {args.output}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
