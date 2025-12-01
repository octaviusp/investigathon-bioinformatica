# Investigathon

Solución al track Bioinformática del Investigathon.

## 📋 Descripción del Proyecto

Este proyecto aborda problemas de **identificación taxonómica de especies** mediante análisis de secuencias de ADN del gen **CO1** (Citocromo Oxidasa Subunidad I). Utilizamos el dataset **MIDORI2** que contiene secuencias de referencia con clasificación taxonómica completa.

### Contexto Biológico

- **Gen CO1**: Marcador genético estándar para identificación de especies (DNA barcoding)
- **Variabilidad intra-especie**: Las secuencias de la misma especie pueden tener pequeñas diferencias (mutaciones silenciosas)
- **Desafío**: Identificar especies desconocidas comparándolas con una base de datos de referencia

### Arquitectura de Datos

El sistema trabaja con dos tablas relacionadas:

1. **Archivo `.taxon`**: Contiene las etiquetas taxonómicas (Reino, Filo, Clase, Orden, Familia, Género, Especie)
2. **Archivo `.fasta`**: Contiene las secuencias de ADN en formato FASTA
3. **Archivo `query.fasta`**: Secuencias desconocidas que deben ser clasificadas

**Clave de unión**: El `sequence_id` (ej: `MG559732.1`) vincula ambas tablas.

## 🎯 Problemáticas a Resolver

### Problema 1: Análisis de Entropía
- **Objetivo**: Analizar la variabilidad genética dentro y entre especies
- **Desafío**: Las secuencias tienen longitudes diferentes (690, 897 bases, etc.)
- **Enfoque**: Necesita alineamiento o métodos que toleren diferentes longitudes

### Problema 2: Representación de Secuencias
- **Objetivo**: Encontrar una representación numérica de las secuencias para comparación
- **Desafío**: Las secuencias no tienen el mismo largo
- **Enfoque**: Considerar métodos como K-mers, embeddings, o técnicas de alineamiento

### Problema 3: Clasificación de Secuencias Desconocidas
- **Objetivo**: Clasificar las secuencias en `query.fasta` a nivel de especie
- **Desafío**: Determinar la especie más probable para cada secuencia desconocida
- **Enfoque**: Usar algoritmos de clasificación (KNN, árboles de decisión, redes neuronales, etc.)

## ✅ Progreso Actual

### [x] Setup del Proyecto
- [x] Configuración de Poetry como gestor de dependencias
- [x] Estructura básica del proyecto

### [x] Carga y Preprocesamiento de Datos
- [x] Script `loader.py` para cargar archivos `.taxon` y `.fasta`
- [x] Extracción de IDs base para unión de tablas
- [x] Parseo de información taxonómica (Reino → Especie)
- [x] Extracción de secuencias completas de ADN
- [x] Unión de datos en DataFrame de pandas
- [x] Validación de datos cargados

### [ ] Problema 1: Análisis de Entropía
- [ ] Implementar método de alineamiento o normalización de longitudes
- [ ] Calcular entropía por posición en secuencias alineadas
- [ ] Analizar variabilidad intra-especie vs inter-especie
- [ ] Visualización de resultados

### [ ] Problema 2: Representación de Secuencias
- [ ] Implementar extracción de K-mers
- [ ] Evaluar otros métodos de representación (embeddings, one-hot encoding)
- [ ] Comparar diferentes valores de K
- [ ] Reducción de dimensionalidad si es necesario

### [ ] Problema 3: Clasificación
- [ ] Implementar algoritmo de clasificación
- [ ] Entrenar modelo con datos de referencia
- [ ] Clasificar secuencias de `query.fasta`
- [ ] Evaluar precisión y métricas de rendimiento
- [ ] Generar reporte de resultados

## 🚀 Uso

### Instalación de Dependencias

```bash
poetry install
```

### Cargar Dataset

```bash
poetry run python loader.py
```

Esto cargará los archivos y mostrará un resumen del dataset:
- Total de registros
- Número de secuencias únicas
- Número de especies únicas
- Estadísticas de longitud de secuencias

### Usar el Loader en Otros Scripts

```python
from loader import cargar_dataset

df = cargar_dataset('MIDORI2_UNIQ_NUC_GB268_CO1.taxon', 
                    'MIDORI2_UNIQ_NUC_GB268_CO1.fasta')
```

## 📊 Estadísticas del Dataset

- **Total de registros**: ~1,846,396
- **Secuencias únicas**: ~1,773,970
- **Especies únicas**: ~235,083
- **Longitud promedio**: Variable (690-897+ bases)

## 👥 Autores

- Octavio Pavón
- Jeremias Tanoni
- Valentin Altoe
- Delfina Mosqueira

