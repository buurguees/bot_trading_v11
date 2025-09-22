# 🚀 ENTRENAMIENTO OPTIMIZADO

## 📋 **DESCRIPCIÓN**

Sistema de entrenamiento optimizado para datasets grandes con gestión avanzada de memoria, validación walk-forward robusta y monitoreo en tiempo real.

## ✨ **CARACTERÍSTICAS PRINCIPALES**

### **1. GESTIÓN DE MEMORIA**
- **Procesamiento por chunks** de 50k filas
- **Liberación explícita** de memoria con `gc.collect()`
- **Monitoreo de RAM** con `psutil`
- **Límite de memoria** configurable (85% por defecto)

### **2. VALIDACIÓN WALK-FORWARD**
- **TimeSeriesSplit** con embargo temporal de 30 minutos
- **Mínimo 5 folds** para validación robusta
- **Métricas promediadas** entre folds
- **Prevención de data leakage**

### **3. LOGGING DETALLADO**
- **Progreso cada 10%** del entrenamiento
- **Métricas intermedias** (AUC, Brier) por fold
- **Tiempo estimado** restante
- **Uso de memoria** actual

### **4. EARLY STOPPING**
- **Parar si AUC no mejora** en 3 folds consecutivos
- **Guardar mejor modelo** hasta el momento
- **Prevención de overfitting**

### **5. SISTEMA DE CHECKPOINTS**
- **Guardar estado** cada fold completado
- **Reanudar desde** último checkpoint
- **Recuperación automática** de fallos

### **6. CONFIGURACIÓN ADAPTATIVA**
- **Max_iter automático** según tamaño del dataset
- **Detección de GPU** si está disponible
- **Escalado automático** de batch size

## 🛠️ **INSTALACIÓN**

### **1. Instalar dependencias:**
```bash
pip install -r requirements.txt
pip install psutil>=5.9.0
```

### **2. Configurar sistema:**
```bash
python core/ml/training/configure_optimized_training.py --check-requirements --create-logging
```

### **3. Probar instalación:**
```bash
python test_optimized_training.py
```

## 🚀 **USO**

### **ENTRENAMIENTO BÁSICO:**
```bash
python -m core.ml.training.train_direction \
    --symbol BTCUSDT \
    --tf 1m \
    --horizon 1 \
    --max-bars 0
```

### **ENTRENAMIENTO OPTIMIZADO:**
```bash
python -m core.ml.training.train_direction \
    --symbol BTCUSDT \
    --tf 1m \
    --horizon 1 \
    --max-bars 0 \
    --n-splits 5 \
    --embargo-minutes 30 \
    --chunk-size 50000 \
    --use-gpu
```

### **REANUDAR ENTRENAMIENTO:**
```bash
python -m core.ml.training.train_direction \
    --symbol BTCUSDT \
    --tf 1m \
    --horizon 1 \
    --resume
```

## 📊 **MONITOREO**

### **MONITOR EN TIEMPO REAL:**
```bash
python core/ml/monitoring/monitor_training_progress.py \
    --symbol BTCUSDT \
    --tf 1m \
    --horizon 1 \
    --refresh 30
```

### **RESUMEN RÁPIDO:**
```bash
python core/ml/monitoring/monitor_training_progress.py \
    --symbol BTCUSDT \
    --tf 1m \
    --horizon 1 \
    --summary
```

## ⚙️ **CONFIGURACIÓN**

### **ARCHIVO DE CONFIGURACIÓN:**
```yaml
# config/ml/training.yaml
training:
  optimized:
    enabled: true
    chunk_size: 50000
    memory_threshold: 0.85
    n_splits: 5
    embargo_minutes: 30
    early_stopping_patience: 3
    adaptive_max_iter: true
    use_gpu: false
    checkpoint_enabled: true
```

### **VARIABLES DE ENTORNO:**
```bash
# config/training.env
BT_OPTIMIZED_TRAINING=true
BT_CHUNK_SIZE=50000
BT_MEMORY_THRESHOLD=0.85
BT_N_SPLITS=5
BT_EMBARGO_MINUTES=30
BT_USE_GPU=false
```

## 📁 **ARCHIVOS GENERADOS**

### **CHECKPOINTS:**
- `logs/checkpoint_{symbol}_{tf}_H{horizon}.pkl`

### **MODELOS:**
- `artifacts/direction/{symbol}_{tf}_H{horizon}_logreg_optimized.pkl`

### **LOGS:**
- `logs/train_direction_optimized.log`
- `logs/train_direction.log`

## 🔧 **PARÁMETROS AVANZADOS**

### **GESTIÓN DE MEMORIA:**
- `--chunk-size`: Tamaño de chunk (default: 50000)
- `--memory-threshold`: Límite de memoria (default: 0.85)

### **VALIDACIÓN:**
- `--n-splits`: Número de folds (default: 5)
- `--embargo-minutes`: Embargo temporal (default: 30)

### **ENTRENAMIENTO:**
- `--use-gpu`: Usar GPU si está disponible
- `--resume`: Reanudar desde checkpoint
- `--max-bars`: Límite de barras (0 = todo)

## 📊 **MÉTRICAS MONITOREADAS**

### **POR FOLD:**
- **AUC**: Area Under Curve
- **Brier**: Brier Score Loss
- **Accuracy**: Precisión
- **Tiempo**: Tiempo de entrenamiento

### **SISTEMA:**
- **Memoria**: Uso de RAM
- **CPU**: Uso de procesador
- **GPU**: Uso de GPU (si disponible)

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **MEMORIA INSUFICIENTE:**
```bash
# Reducir chunk size
--chunk-size 25000

# Reducir número de folds
--n-splits 3
```

### **ENTRENAMIENTO LENTO:**
```bash
# Usar GPU si está disponible
--use-gpu

# Reducir dataset para pruebas
--max-bars 10000
```

### **CHECKPOINT CORRUPTO:**
```bash
# Eliminar checkpoint y reiniciar
rm logs/checkpoint_*.pkl
```

## 📈 **RENDIMIENTO ESPERADO**

### **DATASET PEQUEÑO (< 10k filas):**
- **Tiempo**: 2-5 minutos
- **Memoria**: < 2GB
- **Folds**: 3-5

### **DATASET MEDIANO (10k-100k filas):**
- **Tiempo**: 10-30 minutos
- **Memoria**: 2-8GB
- **Folds**: 5-7

### **DATASET GRANDE (> 100k filas):**
- **Tiempo**: 30+ minutos
- **Memoria**: 8+ GB
- **Folds**: 5-10

## 🔄 **INTEGRACIÓN CON RUNNER**

El entrenamiento optimizado se integra automáticamente con el runner diario:

```python
# En runner.py
run_cmd([
    "python", "-m", "core.ml.training.train_direction",
    "--symbol", symbol, "--tf", tf,
    "--horizon", str(horizon),
    "--from", f_train, "--to", t_train,
    "--seed", str(seed),
    "--max-bars", "0",  # Usar todo el histórico
    "--n-splits", "5",  # 5 folds
    "--embargo-minutes", "30",  # 30 min embargo
    "--chunk-size", "50000"  # 50k filas por chunk
])
```

## 📚 **EJEMPLOS DE USO**

### **ENTRENAMIENTO COMPLETO:**
```bash
# BTCUSDT 1m con todo el histórico
python -m core.ml.training.train_direction \
    --symbol BTCUSDT --tf 1m --horizon 1 \
    --max-bars 0 --n-splits 5 --use-gpu
```

### **ENTRENAMIENTO RÁPIDO:**
```bash
# Prueba rápida con dataset pequeño
python -m core.ml.training.train_direction \
    --symbol BTCUSDT --tf 1m --horizon 1 \
    --max-bars 5000 --n-splits 3 --chunk-size 1000
```

### **MONITOREO CONTINUO:**
```bash
# Monitor en segundo plano
nohup python core/ml/monitoring/monitor_training_progress.py \
    --symbol BTCUSDT --tf 1m --horizon 1 \
    --refresh 60 > monitor.log 2>&1 &
```

## 🎯 **BENEFICIOS**

1. **Escalabilidad**: Maneja datasets de cualquier tamaño
2. **Robustez**: Validación walk-forward robusta
3. **Eficiencia**: Gestión optimizada de memoria
4. **Confiabilidad**: Sistema de checkpoints
5. **Monitoreo**: Visibilidad completa del proceso
6. **Flexibilidad**: Configuración adaptativa
7. **Recuperación**: Capacidad de reanudar entrenamientos
