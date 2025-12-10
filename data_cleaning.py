#!/usr/bin/env python3
"""Data cleaning: eliminar duplicados y análisis por grupo taxonómico."""

import re
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

TAXON_FILE = "data/MIDORI2_UNIQ_NUC_GB268_CO1.taxon"
FASTA_FILE = "data/MIDORI2_UNIQ_NUC_GB268_CO1.fasta"


def parse_fasta_sequences(filepath: str) -> dict:
    """Lee archivo FASTA y retorna dict {seq_id: secuencia}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]  # Sin el '>'
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def parse_taxon_file(filepath: str) -> pd.DataFrame:
    """Parsea archivo .taxon y extrae longitud y taxonomía."""
    records = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                continue

            seq_id, taxonomy = parts

            # Extraer longitud
            match = re.search(r'<(\d+)\.>(\d+)', seq_id)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                length = end - start + 1
            else:
                match2 = re.search(r'\.(\d+)\.(\d+)$', seq_id)
                if match2:
                    start, end = int(match2.group(1)), int(match2.group(2))
                    length = end - start + 1
                else:
                    continue

            if length <= 0 or length > 5000:
                continue

            # Parsear taxonomía
            tax_dict = {}
            for level in taxonomy.split(';'):
                if '__' in level:
                    prefix, name = level.split('__', 1)
                    name_clean = re.sub(r'_\d+$', '', name)
                    tax_dict[prefix] = name_clean

            records.append({
                'seq_id': seq_id,
                'length': length,
                'kingdom': tax_dict.get('k', ''),
                'phylum': tax_dict.get('p', ''),
                'class': tax_dict.get('c', ''),
                'order': tax_dict.get('o', ''),
                'family': tax_dict.get('f', ''),
                'genus': tax_dict.get('g', ''),
                'species': tax_dict.get('s', ''),
                'taxonomy_raw': taxonomy
            })

    return pd.DataFrame(records)


def find_duplicates(df: pd.DataFrame, sequences: dict) -> pd.DataFrame:
    """Encuentra duplicados basándose en hash de secuencia."""
    print("Calculando hashes de secuencias...")

    seq_hashes = []
    for seq_id in df['seq_id']:
        if seq_id in sequences:
            seq = sequences[seq_id]
            seq_hash = hashlib.md5(seq.encode()).hexdigest()
        else:
            seq_hash = None
        seq_hashes.append(seq_hash)

    df = df.copy()
    df['seq_hash'] = seq_hashes

    # Contar duplicados
    hash_counts = df['seq_hash'].value_counts()
    duplicated_hashes = hash_counts[hash_counts > 1]

    print(f"\nSecuencias totales: {len(df):,}")
    print(f"Hashes únicos: {df['seq_hash'].nunique():,}")
    print(f"Secuencias duplicadas (mismo hash): {len(df) - df['seq_hash'].nunique():,}")
    print(f"Grupos con duplicados: {len(duplicated_hashes):,}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, manteniendo la primera ocurrencia."""
    df_clean = df.drop_duplicates(subset='seq_hash', keep='first')
    print(f"\nDespués de eliminar duplicados:")
    print(f"  Secuencias: {len(df_clean):,} (eliminadas: {len(df) - len(df_clean):,})")
    return df_clean


def plot_taxonomic_group(df: pd.DataFrame, group_name: str, level: str = 'auto'):
    """Genera gráficos para un grupo taxonómico específico."""

    # Auto-detectar nivel taxonómico
    if level == 'auto':
        for lvl in ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']:
            if df[lvl].str.contains(group_name, case=False, na=False).any():
                level = lvl
                break
        else:
            print(f"No se encontró '{group_name}' en ningún nivel taxonómico")
            return None

    # Filtrar por grupo
    mask = df[level].str.contains(group_name, case=False, na=False)
    df_group = df[mask]

    if len(df_group) == 0:
        print(f"No se encontraron secuencias para '{group_name}'")
        return None

    print(f"\n{'='*60}")
    print(f"Análisis para: {group_name} (nivel: {level})")
    print(f"{'='*60}")
    print(f"Secuencias encontradas: {len(df_group):,}")
    print(f"\nEstadísticas de longitud:")
    print(f"  Media:   {df_group['length'].mean():.2f}")
    print(f"  Std:     {df_group['length'].std():.2f}")
    print(f"  Min:     {df_group['length'].min()}")
    print(f"  Max:     {df_group['length'].max()}")
    print(f"  Mediana: {df_group['length'].median():.0f}")
    print(f"  Q1:      {df_group['length'].quantile(0.25):.0f}")
    print(f"  Q3:      {df_group['length'].quantile(0.75):.0f}")

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Histograma
    ax1 = axes[0]
    ax1.hist(df_group['length'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df_group['length'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Media: {df_group['length'].mean():.0f}")
    ax1.axvline(df_group['length'].median(), color='green', linestyle='--', linewidth=2,
                label=f"Mediana: {df_group['length'].median():.0f}")
    # ±1 std
    mean_val = df_group['length'].mean()
    std_val = df_group['length'].std()
    ax1.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2,
                label=f"±1 Std: {std_val:.0f}")
    ax1.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2)
    ax1.set_xlabel('Longitud (bp)', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title(f'Distribución de longitudes - {group_name}', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot(df_group['length'], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)

    # Stats como texto
    stats_text = (f"n = {len(df_group):,}\n"
                  f"Media: {df_group['length'].mean():.1f}\n"
                  f"Std: {df_group['length'].std():.1f}\n"
                  f"Min: {df_group['length'].min()}\n"
                  f"Max: {df_group['length'].max()}")
    ax2.text(1.25, df_group['length'].median(), stats_text, fontsize=10,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_ylabel('Longitud (bp)', fontsize=12)
    ax2.set_title(f'Boxplot - {group_name}', fontsize=14)
    ax2.set_xticklabels([group_name])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'length_stats_{group_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Si hay subgrupos, mostrar desglose
    if level != 'species':
        next_levels = {'kingdom': 'phylum', 'phylum': 'class', 'class': 'order',
                       'order': 'family', 'family': 'genus', 'genus': 'species'}
        if level in next_levels:
            next_lvl = next_levels[level]
            subgroup_stats = df_group.groupby(next_lvl)['length'].agg(['mean', 'std', 'count'])
            subgroup_stats = subgroup_stats[subgroup_stats['count'] >= 2].sort_values('count', ascending=False)
            if len(subgroup_stats) > 0:
                print(f"\nDesglose por {next_lvl}:")
                print(subgroup_stats.head(20).round(2).to_string())

    return df_group


def main():
    print("Cargando datos...")

    # Parsear taxon
    df = parse_taxon_file(TAXON_FILE)
    print(f"Secuencias en taxon: {len(df):,}")

    # Parsear FASTA
    print("Cargando secuencias FASTA (esto puede tardar)...")
    sequences = parse_fasta_sequences(FASTA_FILE)
    print(f"Secuencias en FASTA: {len(sequences):,}")

    # Encontrar duplicados
    df = find_duplicates(df, sequences)

    # Eliminar duplicados
    df_clean = remove_duplicates(df)

    # Guardar dataset limpio
    df_clean.to_csv('data/cleaned_sequences.csv', index=False)
    print(f"\nDataset limpio guardado en: data/cleaned_sequences.csv")

    # Análisis interactivo
    if len(sys.argv) > 1:
        group_name = ' '.join(sys.argv[1:])
        plot_taxonomic_group(df_clean, group_name)
    else:
        # Ejemplo con Ripella
        print("\n" + "="*60)
        print("Uso: python data_cleaning.py <grupo_taxonomico>")
        print("Ejemplo: python data_cleaning.py Ripella")
        print("="*60)

        # Demo con algunos grupos
        for group in ['Arthropoda', 'Insecta', 'Chordata']:
            plot_taxonomic_group(df_clean, group)


if __name__ == "__main__":
    main()
