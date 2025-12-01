"""
Análisis Exploratorio de Datos (EDA) para el dataset MIDORI2.

Genera visualizaciones y estadísticas descriptivas para entender
la distribución y características de los datos taxonómicos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from loader import cargar_dataset


def calcular_estadisticas(df: pd.DataFrame) -> dict:
    """
    Calcula estadísticas descriptivas del dataset.
    
    Args:
        df: DataFrame con los datos cargados
    
    Returns:
        Diccionario con estadísticas calculadas
    """
    longitudes = df['dna_sequence'].str.len()
    
    return {
        'total_registros': len(df),
        'secuencias_unicas': df['sequence_id'].nunique(),
        'especies_unicas': df['species'].nunique(),
        'generos_unicos': df['genus'].nunique(),
        'familias_unicas': df['family'].nunique(),
        'ordenes_unicos': df['order'].nunique(),
        'clases_unicas': df['class'].nunique(),
        'filos_unicos': df['phylum'].nunique(),
        'reinos_unicos': df['kingdom'].nunique(),
        'longitud_promedio': longitudes.mean(),
        'longitud_mediana': longitudes.median(),
        'longitud_min': longitudes.min(),
        'longitud_max': longitudes.max(),
        'longitud_std': longitudes.std(),
    }


def graficar_distribucion_por_taxon(df: pd.DataFrame):
    """
    Genera gráficos de barras mostrando el número de especies por cada nivel taxonómico.
    
    Args:
        df: DataFrame con los datos cargados
    """
    taxones = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribución de Especies por Nivel Taxonómico', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, taxon in enumerate(taxones):
        ax = axes[idx]
        conteo = df[taxon].value_counts().head(20)
        
        ax.barh(range(len(conteo)), conteo.values)
        ax.set_yticks(range(len(conteo)))
        ax.set_yticklabels(conteo.index, fontsize=8)
        ax.set_xlabel('Número de Especies', fontsize=10)
        ax.set_title(f'Top 20 {taxon.capitalize()}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Invertir para mostrar el más grande arriba
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('distribucion_por_taxon.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: distribucion_por_taxon.png")


def graficar_heatmap_distribucion(df: pd.DataFrame):
    """
    Genera un heatmap mostrando la distribución de secuencias por reino y filo.
    
    Args:
        df: DataFrame con los datos cargados
    """
    # Filtrar datos válidos
    df_filtrado = df[(df['kingdom'] != '') & (df['phylum'] != '')]
    
    # Crear tabla de contingencia
    tabla_contingencia = pd.crosstab(df_filtrado['kingdom'], df_filtrado['phylum'])
    
    # Limitar a los top 10 filos más comunes para mejor visualización
    top_filos = df_filtrado['phylum'].value_counts().head(10).index
    tabla_contingencia = tabla_contingencia[top_filos]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(tabla_contingencia.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(tabla_contingencia.columns)))
    ax.set_xticklabels(tabla_contingencia.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(tabla_contingencia.index)))
    ax.set_yticklabels(tabla_contingencia.index, fontsize=9)
    
    ax.set_xlabel('Phylum', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kingdom', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap: Distribución de Secuencias por Reino y Filo', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Agregar valores en las celdas
    for i in range(len(tabla_contingencia.index)):
        for j in range(len(tabla_contingencia.columns)):
            valor = tabla_contingencia.iloc[i, j]
            if valor > 0:
                ax.text(j, i, f'{valor:,}', ha='center', va='center', 
                       fontsize=7, color='black' if valor < tabla_contingencia.values.max() * 0.5 else 'white')
    
    plt.colorbar(im, ax=ax, label='Número de Secuencias')
    plt.tight_layout()
    plt.savefig('heatmap_distribucion.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: heatmap_distribucion.png")


def graficar_distribucion_longitudes(df: pd.DataFrame):
    """
    Genera histograma y boxplot de las longitudes de las secuencias.
    
    Args:
        df: DataFrame con los datos cargados
    """
    longitudes = df['dna_sequence'].str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histograma
    axes[0].hist(longitudes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(longitudes.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {longitudes.mean():.1f}')
    axes[0].axvline(longitudes.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {longitudes.median():.1f}')
    axes[0].set_xlabel('Longitud de Secuencia (bases)', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Distribución de Longitudes de Secuencias', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(longitudes, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Longitud de Secuencia (bases)', fontsize=12)
    axes[1].set_title('Boxplot de Longitudes de Secuencias', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('distribucion_longitudes.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: distribucion_longitudes.png")


def graficar_top_especies(df: pd.DataFrame, top_n: int = 30):
    """
    Genera gráfico de barras con las especies más frecuentes.
    
    Args:
        df: DataFrame con los datos cargados
        top_n: Número de especies a mostrar
    """
    top_especies = df['species'].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(range(len(top_especies)), top_especies.values, color='coral')
    ax.set_yticks(range(len(top_especies)))
    ax.set_yticklabels(top_especies.index, fontsize=9)
    ax.set_xlabel('Número de Secuencias', fontsize=12, fontweight='bold')
    ax.set_ylabel('Especie', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Especies con Más Secuencias', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Agregar valores en las barras
    for i, v in enumerate(top_especies.values):
        ax.text(v + max(top_especies.values) * 0.01, i, f'{v:,}', 
               va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('top_especies.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: top_especies.png")


def graficar_longitudes_por_reino(df: pd.DataFrame):
    """
    Genera boxplot de longitudes de secuencias agrupadas por reino.
    
    Args:
        df: DataFrame con los datos cargados
    """
    df_filtrado = df[df['kingdom'] != '']
    reinos = df_filtrado['kingdom'].value_counts().head(10).index
    df_filtrado = df_filtrado[df_filtrado['kingdom'].isin(reinos)]
    
    longitudes_por_reino = [df_filtrado[df_filtrado['kingdom'] == reino]['dna_sequence'].str.len().values 
                           for reino in reinos]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bp = ax.boxplot(longitudes_por_reino, labels=reinos, patch_artist=True, vert=True)
    
    # Colorear los boxplots
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Longitud de Secuencia (bases)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Reino', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Longitudes de Secuencias por Reino', 
                fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('longitudes_por_reino.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: longitudes_por_reino.png")


def graficar_estadisticas_taxones(df: pd.DataFrame):
    """
    Genera gráfico comparativo del número de taxones únicos por nivel.
    
    Args:
        df: DataFrame con los datos cargados
    """
    taxones = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    conteos = [df[taxon].nunique() for taxon in taxones]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(taxones, conteos, color='teal', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Número de Taxones Únicos', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nivel Taxonómico', fontsize=12, fontweight='bold')
    ax.set_title('Número de Taxones Únicos por Nivel Taxonómico', 
                fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, valor in zip(bars, conteos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{valor:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('estadisticas_taxones.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: estadisticas_taxones.png")


def imprimir_estadisticas(stats: dict):
    """
    Imprime estadísticas descriptivas en consola.
    
    Args:
        stats: Diccionario con estadísticas calculadas
    """
    print("\n" + "="*60)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("="*60)
    print(f"\n📊 Registros totales: {stats['total_registros']:,}")
    print(f"🔬 Secuencias únicas: {stats['secuencias_unicas']:,}")
    print(f"\n📈 Taxones únicos:")
    print(f"   - Reinos: {stats['reinos_unicos']:,}")
    print(f"   - Filos: {stats['filos_unicos']:,}")
    print(f"   - Clases: {stats['clases_unicas']:,}")
    print(f"   - Órdenes: {stats['ordenes_unicos']:,}")
    print(f"   - Familias: {stats['familias_unicas']:,}")
    print(f"   - Géneros: {stats['generos_unicos']:,}")
    print(f"   - Especies: {stats['especies_unicas']:,}")
    print(f"\n📏 Longitudes de secuencias:")
    print(f"   - Promedio: {stats['longitud_promedio']:.2f} bases")
    print(f"   - Mediana: {stats['longitud_mediana']:.2f} bases")
    print(f"   - Mínimo: {stats['longitud_min']} bases")
    print(f"   - Máximo: {stats['longitud_max']} bases")
    print(f"   - Desviación estándar: {stats['longitud_std']:.2f} bases")


def ejecutar_eda(archivo_taxon: str = 'MIDORI2_UNIQ_NUC_GB268_CO1.taxon',
                 archivo_fasta: str = 'MIDORI2_UNIQ_NUC_GB268_CO1.fasta'):
    """
    Ejecuta el análisis exploratorio completo de datos.
    
    Args:
        archivo_taxon: Ruta al archivo .taxon
        archivo_fasta: Ruta al archivo .fasta
    """
    print("="*60)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*60)
    
    # Cargar datos
    print("\n📂 Cargando dataset...")
    df = cargar_dataset(archivo_taxon, archivo_fasta)
    
    # Calcular estadísticas
    print("\n📊 Calculando estadísticas...")
    stats = calcular_estadisticas(df)
    imprimir_estadisticas(stats)
    
    # Generar visualizaciones
    print("\n🎨 Generando visualizaciones...")
    graficar_distribucion_por_taxon(df)
    graficar_heatmap_distribucion(df)
    graficar_distribucion_longitudes(df)
    graficar_top_especies(df)
    graficar_longitudes_por_reino(df)
    graficar_estadisticas_taxones(df)
    
    print("\n" + "="*60)
    print("✅ EDA completado exitosamente")
    print("="*60)
    print("\n📁 Archivos generados:")
    print("   - distribucion_por_taxon.png")
    print("   - heatmap_distribucion.png")
    print("   - distribucion_longitudes.png")
    print("   - top_especies.png")
    print("   - longitudes_por_reino.png")
    print("   - estadisticas_taxones.png")


if __name__ == '__main__':
    ejecutar_eda()

