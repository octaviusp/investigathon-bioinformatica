#!/usr/bin/env python3
"""Análisis de longitudes de secuencias por grupo taxonómico."""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TAXON_FILE = "data/MIDORI2_UNIQ_NUC_GB268_CO1.taxon"

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

            # Extraer longitud: formato "ID.<start.>end" o "ID.start.end"
            # La longitud es SIEMPRE end - start + 1
            # Ejemplo: <1.>658 -> 658-1+1=658, <1470.>3000 -> 3000-1470+1=1531
            # MN095727.1.1.651 -> 651-1+1=651
            match = re.search(r'<(\d+)\.>(\d+)', seq_id)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                length = end - start + 1
            else:
                # Formato alternativo: ID.X.start.end (sin < >) ej: MN095727.1.1.651
                # Buscar los últimos dos números separados por punto
                match2 = re.search(r'\.(\d+)\.(\d+)$', seq_id)
                if match2:
                    start, end = int(match2.group(1)), int(match2.group(2))
                    length = end - start + 1
                else:
                    continue

            # Validar que la longitud sea razonable (COI ~650bp, máximo ~3000bp)
            if length <= 0 or length > 5000:
                continue

            # Parsear taxonomía
            tax_dict = {}
            for level in taxonomy.split(';'):
                if '__' in level:
                    prefix, name = level.split('__', 1)
                    # Remover ID numérico del final
                    name_clean = re.sub(r'_\d+$', '', name)
                    tax_dict[prefix] = name_clean

            records.append({
                'seq_id': seq_id.split('.')[0],
                'length': length,
                'kingdom': tax_dict.get('k', ''),
                'phylum': tax_dict.get('p', ''),
                'class': tax_dict.get('c', ''),
                'order': tax_dict.get('o', ''),
                'family': tax_dict.get('f', ''),
                'genus': tax_dict.get('g', ''),
                'species': tax_dict.get('s', '')
            })

    return pd.DataFrame(records)


def plot_stats_by_level(df: pd.DataFrame, level: str, top_n: int = 20):
    """Genera boxplot con estadísticas por nivel taxonómico."""

    # Agrupar y calcular stats
    stats = df.groupby(level)['length'].agg(['mean', 'std', 'count', 'median'])
    stats = stats[stats['count'] >= 5]  # Solo grupos con al menos 5 secuencias
    stats = stats.nlargest(top_n, 'count')

    # Filtrar df para solo esos grupos
    groups = stats.index.tolist()
    df_filtered = df[df[level].isin(groups)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Boxplot
    ax1 = axes[0]
    df_pivot = [df_filtered[df_filtered[level] == g]['length'].values for g in groups]
    bp = ax1.boxplot(df_pivot, labels=groups, vert=True, patch_artist=True)
    ax1.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Longitud (bp)')
    ax1.set_title(f'Distribución de longitudes por {level} (top {top_n})')
    ax1.grid(axis='y', alpha=0.3)

    # Barplot de mean ± std
    ax2 = axes[1]
    x = np.arange(len(groups))
    means = stats.loc[groups, 'mean'].values
    stds = stats.loc[groups, 'std'].values

    bars = ax2.bar(x, means, yerr=stds, capsize=3, alpha=0.7, color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Longitud (bp)')
    ax2.set_title(f'Media ± Desviación Estándar por {level}')
    ax2.grid(axis='y', alpha=0.3)

    # Añadir línea de media global
    global_mean = df['length'].mean()
    ax2.axhline(y=global_mean, color='red', linestyle='--', label=f'Media global: {global_mean:.0f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'seq_length_stats_{level}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Imprimir tabla de stats
    print(f"\n{'='*60}")
    print(f"Estadísticas por {level.upper()}")
    print(f"{'='*60}")
    stats_print = stats.copy()
    stats_print.columns = ['Media', 'Std', 'N', 'Mediana']
    print(stats_print.round(2).to_string())

    return stats


def main():
    print("Cargando datos...")
    df = parse_taxon_file(TAXON_FILE)
    print(f"Total secuencias: {len(df):,}")
    print(f"\nEstadísticas globales de longitud:")
    print(f"  Media: {df['length'].mean():.2f}")
    print(f"  Std:   {df['length'].std():.2f}")
    print(f"  Min:   {df['length'].min()}")
    print(f"  Max:   {df['length'].max()}")
    print(f"  Mediana: {df['length'].median():.0f}")

    # Histograma global con mejor escala
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Histograma completo
    ax1 = axes[0]
    ax1.hist(df['length'], bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df['length'].mean(), color='red', linestyle='--', linewidth=2, label=f"Media: {df['length'].mean():.0f}")
    ax1.axvline(df['length'].median(), color='green', linestyle='--', linewidth=2, label=f"Mediana: {df['length'].median():.0f}")
    ax1.set_xlabel('Longitud (bp)', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución global de longitudes', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Boxplot con outliers marcados
    ax2 = axes[1]
    bp = ax2.boxplot(df['length'], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)

    # Añadir stats como texto
    stats_text = f"Media: {df['length'].mean():.1f}\nStd: {df['length'].std():.1f}\nMin: {df['length'].min()}\nMax: {df['length'].max()}\nQ1: {df['length'].quantile(0.25):.0f}\nQ3: {df['length'].quantile(0.75):.0f}"
    ax2.text(1.3, df['length'].median(), stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_ylabel('Longitud (bp)', fontsize=12)
    ax2.set_title('Boxplot con outliers', fontsize=14)
    ax2.set_xticklabels(['Todas las secuencias'])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('seq_length_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Histograma enfocado en rango típico COI (400-900 bp)
    df_typical = df[(df['length'] >= 400) & (df['length'] <= 900)]
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df_typical['length'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(df_typical['length'].mean(), color='red', linestyle='--', linewidth=2, label=f"Media: {df_typical['length'].mean():.0f}")
    ax.axvline(df_typical['length'].median(), color='green', linestyle='--', linewidth=2, label=f"Mediana: {df_typical['length'].median():.0f}")
    ax.axvline(df_typical['length'].mean() - df_typical['length'].std(), color='orange', linestyle=':', linewidth=2, label=f"±1 Std: {df_typical['length'].std():.0f}")
    ax.axvline(df_typical['length'].mean() + df_typical['length'].std(), color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('Longitud (bp)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(f'Distribución de longitudes (rango 400-900 bp, n={len(df_typical):,})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('seq_length_histogram_focused.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Analizar por diferentes niveles taxonómicos
    for level in ['phylum', 'class', 'order', 'family']:
        plot_stats_by_level(df, level, top_n=15)


if __name__ == "__main__":
    main()
