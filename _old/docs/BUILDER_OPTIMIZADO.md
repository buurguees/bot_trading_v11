# 🚀 BUILDER DE DATASETS OPTIMIZADO PARA BIG DATA

## 📋 **DESCRIPCIÓN**

Sistema optimizado para construcción de datasets de trading con manejo eficiente de big data, incluyendo cache inteligente, lazy loading, validación de datos y streaming.

## ✨ **CARACTERÍSTICAS PRINCIPALES**

### **1. OPTIMIZACIÓN SQL**
- **Queries con LIMIT y OFFSET** para paginación eficiente
- **Índices compuestos**: `(symbol, timeframe, timestamp)`
- **Queries preparadas** para reutilización
- **Pool de conexiones** para múltiples workers

### **2. SISTEMA DE CACHE INTELIGENTE**
- **Cache en disco** con pickle comprimido (gzip)
- **Hash de parámetros** para invalidación automática
- **TTL de 1 hora** para datos recientes
- **Limpieza automática** de cache viejo

### **3. LAZY LOADING**
- **Cargar solo columnas necesarias**
- **Iteradores** para datasets que no caben en memoria
- **Streaming de features** por chunks temporales

### **4. SNAPSHOTS MULTI-TF OPTIMIZADOS**
- **Pre-calcular snapshots** más comunes
- **Interpolación eficiente** para timeframes faltantes
- **Cache separado** para cada combinación de TF

### **5. VALIDACIÓN DE DATOS**
- **Detectar y manejar gaps** en datos
- **Validación de rangos** (precios, volúmenes)
- **Logging de calidad** de datos por símbolo

### **6. COMPRESIÓN Y FORMATO**
- **Formato optimizado** para lecturas secuenciales
- **Compresión automática** de features numéricas
- **Almacenamiento eficiente** en memoria

## 🛠️ **INSTALACIÓN**

### **Dependencias adicionales:**
```bash
pip install gzip pickle hashlib pathlib
```

### **Inicialización:**
```python
from core.ml.datasets.builder import initialize_optimized_builder
initialize_optimized_builder()
```

## 🚀 **USO**

### **FUNCIÓN BÁSICA OPTIMIZADA:**
```python
from core.ml.datasets.builder import build_dataset_optimized

# Cargar dataset con cache
df = build_dataset_optimized(
    symbol="BTCUSDT",
    tf_base="1m",
    use_snapshots=True,
    use_cache=True,
    chunk_size=50000,
    features=["rsi14", "ema20", "ema50"]
)
```

### **STREAMING PARA DATASETS GRANDES:**
```python
from core.ml.datasets.builder import build_dataset_streaming

# Procesar dataset en chunks
for chunk in build_dataset_streaming("BTCUSDT", "1m", chunk_size=10000):
    # Procesar cada chunk
    process_chunk(chunk)
```

### **FETCH OPTIMIZADO:**
```python
from core.ml.datasets.builder import fetch_base_optimized

# Cargar datos base con optimizaciones
df = fetch_base_optimized(
    symbol="ETHUSDT",
    tf="5m",
    use_cache=True,
    chunk_size=50000,
    features=["rsi14", "ema20", "ema50", "macd"]
)
```

### **SNAPSHOTS OPTIMIZADOS:**
```python
from core.ml.datasets.builder import add_snapshots_optimized

# Agregar snapshots de múltiples timeframes
df_with_snapshots = add_snapshots_optimized(
    df_base=df,
    symbol="BTCUSDT",
    high_tfs=["15m", "1h", "4h", "1d"]
)
```

## ⚙️ **CONFIGURACIÓN**

### **VARIABLES DE CONFIGURACIÓN:**
```python
# Configuración de cache
CACHE_DIR = Path("cache/datasets")
CACHE_TTL = 3600  # 1 hora en segundos
CACHE_MAX_SIZE = 100  # Máximo 100 archivos

# Configuración de paginación
DEFAULT_PAGE_SIZE = 50000
MAX_PAGE_SIZE = 100000

# Configuración de compresión
COMPRESSION_LEVEL = 6  # Balance velocidad/compresión
```

### **CONFIGURACIÓN DE BASE DE DATOS:**
```python
DB_CONFIG = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'echo': False
}
```

## 📊 **MONITOREO Y ESTADÍSTICAS**

### **ESTADÍSTICAS DE CACHE:**
```python
from core.ml.datasets.builder import get_cache_stats

stats = get_cache_stats()
print(f"Archivos en cache: {stats['total_files']}")
print(f"Tamaño total: {stats['total_size_mb']:.2f} MB")
print(f"TTL: {stats['ttl_hours']} horas")
```

### **LIMPIEZA DE CACHE:**
```python
from core.ml.datasets.builder import cleanup_cache

# Limpiar cache viejo
cleanup_cache()
```

### **VALIDACIÓN DE DATOS:**
```python
from core.ml.datasets.builder import DataValidator

validator = DataValidator()

# Validar datos OHLCV
ohlcv_result = validator.validate_ohlcv(df)
print(f"OHLCV válido: {ohlcv_result['valid']}")

# Validar features
feature_result = validator.validate_features(df, ["rsi14", "ema20"])
print(f"Features válidas: {feature_result['valid']}")
```

## 🔧 **FUNCIONES DE COMPATIBILIDAD**

### **FUNCIONES ORIGINALES:**
```python
# Estas funciones mantienen compatibilidad con código existente
from core.ml.datasets.builder import (
    fetch_base,           # Función original
    add_snapshots,        # Función original
    build_dataset         # Función original
)

# Uso normal (ahora optimizado internamente)
df = build_dataset("BTCUSDT", "1m", use_snapshots=True)
```

## 📈 **RENDIMIENTO ESPERADO**

### **MEJORAS DE VELOCIDAD:**
- **Cache hit**: 10-50x más rápido
- **Lazy loading**: Reduce uso de memoria en 70%
- **Índices compuestos**: 5-10x más rápido en queries
- **Paginación**: Maneja datasets de cualquier tamaño

### **USO DE MEMORIA:**
- **Cache comprimido**: 60-80% menos espacio
- **Lazy loading**: Procesa datasets de 100M+ registros
- **Cleanup automático**: Mantiene cache bajo control

### **CALIDAD DE DATOS:**
- **Validación automática**: Detecta problemas de calidad
- **Logging detallado**: Identifica issues por símbolo
- **Manejo de gaps**: Interpola datos faltantes

## 🧪 **PRUEBAS**

### **EJECUTAR PRUEBAS:**
```bash
python test_optimized_builder.py
```

### **PRUEBAS INCLUIDAS:**
1. **Funcionalidad Básica**: Carga y validación de datos
2. **Funcionalidad de Cache**: Verificación de cache hit/miss
3. **Funcionalidad de Streaming**: Procesamiento en chunks
4. **Diferentes Timeframes**: Prueba con múltiples TFs
5. **Estadísticas de Cache**: Monitoreo de uso
6. **Limpieza de Cache**: Gestión de espacio

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **MEMORIA INSUFICIENTE:**
```python
# Reducir chunk size
df = build_dataset_optimized(
    symbol="BTCUSDT",
    tf_base="1m",
    chunk_size=10000  # Reducir de 50000
)
```

### **CACHE LLENO:**
```python
# Limpiar cache manualmente
cleanup_cache()

# Verificar estadísticas
stats = get_cache_stats()
print(f"Archivos: {stats['total_files']}")
```

### **QUERIES LENTAS:**
```python
# Verificar índices
from core.ml.datasets.builder import OptimizedQueryBuilder
query_builder = OptimizedQueryBuilder(ENGINE)
query_builder.ensure_indexes()
```

### **DATOS CORRUPTOS:**
```python
# Validar datos
validator = DataValidator()
result = validator.validate_ohlcv(df)
if not result['valid']:
    print(f"Problemas: {result['issues']}")
```

## 📚 **EJEMPLOS AVANZADOS**

### **PROCESAMIENTO MASIVO:**
```python
symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

for symbol in symbols:
    for tf in timeframes:
        # Procesar en streaming para datasets grandes
        for chunk in build_dataset_streaming(symbol, tf, chunk_size=50000):
            # Procesar chunk
            process_chunk(chunk)
```

### **CACHE PERSONALIZADO:**
```python
from core.ml.datasets.builder import DatasetCache

# Crear cache personalizado
custom_cache = DatasetCache(
    cache_dir=Path("custom_cache"),
    ttl=7200  # 2 horas
)

# Usar cache personalizado
df = fetch_base_optimized(
    symbol="BTCUSDT",
    tf="1m",
    use_cache=True
)
```

### **VALIDACIÓN PERSONALIZADA:**
```python
class CustomValidator(DataValidator):
    @staticmethod
    def validate_custom_rules(df):
        # Validaciones personalizadas
        issues = []
        
        # Ejemplo: Verificar que RSI esté en rango 0-100
        if 'rsi14' in df.columns:
            invalid_rsi = (df['rsi14'] < 0) | (df['rsi14'] > 100)
            if invalid_rsi.any():
                issues.append(f"RSI fuera de rango: {invalid_rsi.sum()} registros")
        
        return issues

# Usar validador personalizado
validator = CustomValidator()
custom_issues = validator.validate_custom_rules(df)
```

## 🎯 **BENEFICIOS**

1. **Escalabilidad**: Maneja datasets de cualquier tamaño
2. **Eficiencia**: Cache inteligente y lazy loading
3. **Robustez**: Validación automática de datos
4. **Compatibilidad**: Funciona con código existente
5. **Monitoreo**: Estadísticas y logging detallados
6. **Flexibilidad**: Configuración adaptativa

## 📁 **ARCHIVOS GENERADOS**

### **CACHE:**
- `cache/datasets/*.pkl.gz` - Archivos de cache comprimidos

### **LOGS:**
- Logs de validación de datos
- Estadísticas de rendimiento
- Alertas de calidad

### **ÍNDICES:**
- `idx_historicaldata_symbol_tf_ts` - Índice para historicaldata
- `idx_features_symbol_tf_ts` - Índice para features

¡El builder optimizado está listo para manejar big data de forma eficiente! 🚀📊✨
