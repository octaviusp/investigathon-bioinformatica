"""
Script para cargar y unir los archivos .taxon y .fasta en un DataFrame de pandas.

Lee los archivos del dataset MIDORI2 y crea una tabla unificada con:
- Sequence_ID: ID de la secuencia
- DNA_Sequence: Secuencia completa de ADN
- Kingdom, Phylum, Class, Order, Family, Genus, Species: Clasificación taxonómica
"""

import pandas as pd
import re
from pathlib import Path


def extraer_id_base(identificador: str) -> str:
    """
    Extrae el ID base de un identificador completo.
    
    Args:
        identificador: String con formato como 'MG559732.1.<1.>690' o '>MG559732.1.<1.>690'
    
    Returns:
        ID base como 'MG559732.1'
    """
    # Remover el símbolo '>' si existe
    id_limpio = identificador.lstrip('>')
    # Extraer la parte antes del primer punto adicional (ej: 'MG559732.1' de 'MG559732.1.<1.>690')
    match = re.match(r'^([^.]+\.\d+)', id_limpio)
    return match.group(1) if match else id_limpio.split('.')[0]


def normalizar_nombre_taxonomico(nombre: str) -> str:
    """
    Normaliza un nombre taxonómico para uso como label.
    
    Convierte a lowercase y reemplaza espacios por guiones bajos.
    
    Args:
        nombre: Nombre taxonómico original (ej: "Clydonella sawyeri")
    
    Returns:
        Nombre normalizado (ej: "clydonella_sawyeri")
    """
    if not nombre or nombre == '':
        return ''
    return nombre.lower().replace(' ', '_').strip()


def parsear_taxonomia(taxonomia_str: str, normalizar: bool = True) -> dict:
    """
    Parsea el string de taxonomía en formato k__;p__;c__;o__;f__;g__;s__
    
    Args:
        taxonomia_str: String con formato 'k__Reino_ID;p__Filo_ID;...'
        normalizar: Si True, normaliza los nombres (lowercase, espacios a guiones bajos)
    
    Returns:
        Diccionario con las claves: kingdom, phylum, class, order, family, genus, species
    """
    taxonomia = {}
    partes = taxonomia_str.split(';')
    
    for parte in partes:
        # Patrón: prefijo__Nombre_ID -> extraer Nombre (todo entre __ y el último _ seguido de dígitos)
        match = re.match(r'^([kgpcofs])__(.+?)_\d+$', parte)
        if match:
            prefijo = match.group(1)
            nombre = match.group(2)
            
            # Normalizar si está activado
            if normalizar:
                nombre = normalizar_nombre_taxonomico(nombre)
            
            if prefijo == 'k':
                taxonomia['kingdom'] = nombre
            elif prefijo == 'p':
                taxonomia['phylum'] = nombre
            elif prefijo == 'c':
                taxonomia['class'] = nombre
            elif prefijo == 'o':
                taxonomia['order'] = nombre
            elif prefijo == 'f':
                taxonomia['family'] = nombre
            elif prefijo == 'g':
                taxonomia['genus'] = nombre
            elif prefijo == 's':
                taxonomia['species'] = nombre
    
    return taxonomia


def cargar_taxon(archivo_taxon: str, normalizar: bool = True) -> pd.DataFrame:
    """
    Carga el archivo .taxon y parsea la información taxonómica.
    
    Args:
        archivo_taxon: Ruta al archivo .taxon
        normalizar: Si True, normaliza los nombres taxonómicos (lowercase, espacios a guiones bajos)
    
    Returns:
        DataFrame con columnas: sequence_id, kingdom, phylum, class, order, family, genus, species
    """
    datos = []
    
    with open(archivo_taxon, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            
            # Separar por tab
            partes = linea.split('\t')
            if len(partes) < 2:
                continue
            
            id_completo = partes[0]
            taxonomia_str = partes[1]
            
            id_base = extraer_id_base(id_completo)
            taxonomia = parsear_taxonomia(taxonomia_str, normalizar=normalizar)
            
            datos.append({
                'sequence_id': id_base,
                'kingdom': taxonomia.get('kingdom', ''),
                'phylum': taxonomia.get('phylum', ''),
                'class': taxonomia.get('class', ''),
                'order': taxonomia.get('order', ''),
                'family': taxonomia.get('family', ''),
                'genus': taxonomia.get('genus', ''),
                'species': taxonomia.get('species', '')
            })
    
    return pd.DataFrame(datos)


def cargar_fasta(archivo_fasta: str) -> pd.DataFrame:
    """
    Carga el archivo .fasta y extrae las secuencias de ADN.
    
    Args:
        archivo_fasta: Ruta al archivo .fasta
    
    Returns:
        DataFrame con columnas: sequence_id, dna_sequence
    """
    datos = []
    id_actual = None
    secuencia_actual = []
    
    with open(archivo_fasta, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            
            if linea.startswith('>'):
                # Guardar la secuencia anterior si existe
                if id_actual is not None:
                    datos.append({
                        'sequence_id': id_actual,
                        'dna_sequence': ''.join(secuencia_actual)
                    })
                
                # Extraer nuevo ID
                id_completo = linea[1:]  # Remover '>'
                id_actual = extraer_id_base(id_completo)
                secuencia_actual = []
            else:
                # Acumular la secuencia
                if id_actual is not None:
                    secuencia_actual.append(linea)
        
        # Guardar la última secuencia
        if id_actual is not None:
            datos.append({
                'sequence_id': id_actual,
                'dna_sequence': ''.join(secuencia_actual)
            })
    
    return pd.DataFrame(datos)


def cargar_dataset(archivo_taxon: str, archivo_fasta: str, normalizar: bool = True) -> pd.DataFrame:
    """
    Carga y une los archivos .taxon y .fasta en un único DataFrame.
    
    Args:
        archivo_taxon: Ruta al archivo .taxon
        archivo_fasta: Ruta al archivo .fasta
        normalizar: Si True, normaliza los nombres taxonómicos (lowercase, espacios a guiones bajos)
    
    Returns:
        DataFrame unificado con toda la información
    """
    print("Cargando archivo .taxon...")
    df_taxon = cargar_taxon(archivo_taxon, normalizar=normalizar)
    print(f"  ✓ Cargadas {len(df_taxon)} entradas taxonómicas")
    
    print("Cargando archivo .fasta...")
    df_fasta = cargar_fasta(archivo_fasta)
    print(f"  ✓ Cargadas {len(df_fasta)} secuencias")
    
    print("Uniendo datos...")
    df_completo = pd.merge(df_fasta, df_taxon, on='sequence_id', how='inner')
    print(f"  ✓ DataFrame final con {len(df_completo)} registros")
    
    return df_completo


if __name__ == '__main__':
    # Rutas por defecto
    archivo_taxon = 'MIDORI2_UNIQ_NUC_GB268_CO1.taxon'
    archivo_fasta = 'MIDORI2_UNIQ_NUC_GB268_CO1.fasta'
    
    # Cargar dataset
    df = cargar_dataset(archivo_taxon, archivo_fasta)
    
    # Mostrar información básica
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET")
    print("="*60)
    print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\nColumnas: {', '.join(df.columns.tolist())}")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    print(f"\nTipos de datos:")
    print(df.dtypes)
    print(f"\nEstadísticas básicas:")
    print(f"  - Secuencias únicas: {df['sequence_id'].nunique()}")
    print(f"  - Especies únicas: {df['species'].nunique()}")
    print(f"  - Longitud promedio de secuencias: {df['dna_sequence'].str.len().mean():.1f} bases")
    print(f"  - Longitud mínima: {df['dna_sequence'].str.len().min()} bases")
    print(f"  - Longitud máxima: {df['dna_sequence'].str.len().max()} bases")

