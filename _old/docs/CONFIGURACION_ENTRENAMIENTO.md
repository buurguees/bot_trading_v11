# ⚙️ CONFIGURACIÓN ROBUSTA PARA ENTRENAMIENTOS LARGOS

## 📋 **DESCRIPCIÓN**

Sistema de configuración robusta para entrenamientos de machine learning de larga duración, optimizado para trading con gestión de recursos, monitoreo avanzado y configuración adaptativa.

## ✨ **CARACTERÍSTICAS PRINCIPALES**

### **1. CONFIGURACIÓN DE RECURSOS**
- **Gestión de memoria**: Límites configurables y monitoreo
- **Procesamiento paralelo**: Workers adaptativos
- **Cache inteligente**: Gestión eficiente de memoria
- **Soporte GPU**: Detección y configuración automática

### **2. VALIDACIÓN CRUZADA ROBUSTA**
- **Walk-forward validation**: Para datos temporales
- **Embargo temporal**: Evitar data leakage
- **Múltiples folds**: Configuración flexible
- **Métricas de estabilidad**: Validación de consistencia

### **3. OPTIMIZACIÓN AVANZADA**
- **Early stopping**: Prevenir overfitting
- **Búsqueda de hiperparámetros**: Random, Grid, Bayesian
- **Configuración adaptativa**: Ajuste automático de parámetros
- **Checkpoints**: Recuperación de entrenamientos

### **4. MONITOREO COMPLETO**
- **Métricas en tiempo real**: Performance y recursos
- **Alertas automáticas**: Umbrales configurables
- **Logging detallado**: Múltiples niveles y rotación
- **Dashboard de métricas**: Visualización en tiempo real

## 🛠️ **INSTALACIÓN Y CONFIGURACIÓN**

### **1. CONFIGURACIÓN AUTOMÁTICA:**
```bash
# Ejecutar configurador interactivo
python configure_training.py

# Validar configuración
python validate_training_config.py
```

### **2. CONFIGURACIÓN MANUAL:**
```yaml
# Editar config/ml/training.yaml directamente
resources:
  max_memory_gb: 8
  max_workers: 4
  chunk_size: 50000
  cache_size_gb: 2
  gpu_enabled: false
```

## 🚀 **USO**

### **CONFIGURACIÓN BÁSICA:**
```python
import yaml
from pathlib import Path

# Cargar configuración
with open('config/ml/training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Usar configuración
max_memory = config['resources']['max_memory_gb']
n_splits = config['training']['validation']['n_splits']
min_auc = config['models']['direction']['promotion_thresholds']['min_auc']
```

### **CONFIGURACIÓN AVANZADA:**
```python
# Configuración personalizada por símbolo
symbol_config = config['balance']['symbols']['BTCUSDT']
initial_balance = symbol_config['initial']
target_balance = symbol_config['target']
risk_per_trade = symbol_config['risk_per_trade']

# Configuración de monitoreo
monitoring = config['monitoring']
log_level = monitoring['log_level']
memory_threshold = monitoring['memory_threshold_gb']
```

## ⚙️ **SECCIONES DE CONFIGURACIÓN**

### **1. RECURSOS DEL SISTEMA:**
```yaml
resources:
  max_memory_gb: 8                    # Límite máximo de RAM
  max_workers: 4                      # Workers paralelos
  chunk_size: 50000                   # Tamaño de chunk
  cache_size_gb: 2                    # Tamaño de cache
  gpu_enabled: false                  # Habilitar GPU
  db_pool_size: 20                    # Pool de conexiones DB
  temp_dir: "temp"                    # Directorio temporal
  artifacts_dir: "artifacts"          # Directorio de artefactos
  logs_dir: "logs"                    # Directorio de logs
  checkpoints_dir: "checkpoints"      # Directorio de checkpoints
```

### **2. CONFIGURACIÓN DE ENTRENAMIENTO:**
```yaml
training:
  validation:
    method: "walk_forward"            # Método de validación
    n_splits: 5                       # Número de folds
    embargo_minutes: 30               # Embargo temporal
    test_size: 0.2                    # Proporción de test
  
  optimization:
    early_stopping:
      patience: 3                     # Paciencia para early stopping
      min_delta: 0.001               # Mejora mínima
    max_iter: 5000                    # Iteraciones máximas
    auto_tune: true                   # Ajuste automático
  
  checkpoints:
    enabled: true                     # Habilitar checkpoints
    frequency: "per_fold"             # Frecuencia de guardado
    retention_days: 7                 # Días de retención
```

### **3. CONFIGURACIÓN DE MODELOS:**
```yaml
models:
  direction:
    default_params:
      C: [0.1, 1.0, 10.0]            # Parámetros de regularización
      solver: ["liblinear", "saga"]   # Solvers disponibles
      max_iter: 3000                  # Iteraciones máximas
    
    promotion_thresholds:
      min_auc: 0.52                   # AUC mínimo para promoción
      max_brier: 0.25                 # Brier score máximo
      min_samples: 1000               # Mínimo de muestras
      min_accuracy: 0.50              # Precisión mínima
```

### **4. CONFIGURACIÓN DE BALANCE:**
```yaml
balance:
  initial: 1000.0                     # Balance inicial
  target: 10000.0                     # Balance objetivo
  risk_per_trade: 0.01                # Riesgo por trade (1%)
  
  symbols:
    BTCUSDT:
      initial: 1000.0
      target: 2000.0
      risk_per_trade: 0.01
      max_leverage: 5
      min_trade_size: 0.0001
```

### **5. CONFIGURACIÓN DE MONITOREO:**
```yaml
monitoring:
  log_level: "INFO"                   # Nivel de log
  progress_frequency: 0.1             # Frecuencia de progreso
  metrics_frequency: "per_fold"       # Frecuencia de métricas
  memory_threshold_gb: 6              # Umbral de memoria
  cpu_threshold_percent: 90           # Umbral de CPU
```

## 📊 **VALIDACIÓN DE CONFIGURACIÓN**

### **VALIDACIÓN AUTOMÁTICA:**
```bash
# Validar configuración completa
python validate_training_config.py

# Validar configuración específica
python validate_training_config.py config/ml/training.yaml
```

### **VALIDACIÓN MANUAL:**
```python
from validate_training_config import TrainingConfigValidator

validator = TrainingConfigValidator("config/ml/training.yaml")
is_valid, errors, warnings = validator.validate_all()

if is_valid:
    print("✅ Configuración válida")
else:
    print("❌ Configuración inválida")
    for error in errors:
        print(f"   {error}")
```

## 🔧 **CONFIGURACIÓN INTERACTIVA**

### **CONFIGURADOR INTERACTIVO:**
```bash
# Ejecutar configurador
python configure_training.py

# Con archivo específico
python configure_training.py config/ml/training.yaml
```

### **FLUJO DE CONFIGURACIÓN:**
1. **Recursos**: Memoria, workers, cache
2. **Entrenamiento**: Validación, optimización, checkpoints
3. **Modelos**: Parámetros, umbrales de promoción
4. **Balance**: Configuración por símbolo
5. **Monitoreo**: Logging, alertas, métricas

## 📈 **OPTIMIZACIÓN DE RENDIMIENTO**

### **CONFIGURACIÓN PARA SISTEMAS PEQUEÑOS:**
```yaml
resources:
  max_memory_gb: 2
  max_workers: 2
  chunk_size: 10000
  cache_size_gb: 0.5

training:
  validation:
    n_splits: 3
    embargo_minutes: 15
```

### **CONFIGURACIÓN PARA SISTEMAS GRANDES:**
```yaml
resources:
  max_memory_gb: 32
  max_workers: 16
  chunk_size: 100000
  cache_size_gb: 8

training:
  validation:
    n_splits: 10
    embargo_minutes: 60
```

### **CONFIGURACIÓN PARA GPU:**
```yaml
resources:
  gpu_enabled: true
  max_memory_gb: 16
  max_workers: 8

training:
  optimization:
    max_iter: 10000
    auto_tune: true
```

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **ERRORES COMUNES:**

1. **Memoria insuficiente:**
```yaml
resources:
  max_memory_gb: 4  # Reducir límite
  chunk_size: 25000  # Reducir chunk size
```

2. **Workers excesivos:**
```yaml
resources:
  max_workers: 2  # Reducir workers
```

3. **Cache muy grande:**
```yaml
resources:
  cache_size_gb: 1  # Reducir cache
```

4. **Validación muy estricta:**
```yaml
models:
  direction:
    promotion_thresholds:
      min_auc: 0.51  # Reducir umbral
      max_brier: 0.26  # Aumentar umbral
```

### **ADVERTENCIAS COMUNES:**

1. **Memoria muy alta:**
   - Reducir `max_memory_gb`
   - Reducir `chunk_size`
   - Reducir `cache_size_gb`

2. **Workers muy altos:**
   - Reducir `max_workers`
   - Considerar límites del sistema

3. **Umbrales muy estrictos:**
   - Ajustar `promotion_thresholds`
   - Considerar calidad de datos

## 📁 **ARCHIVOS DE CONFIGURACIÓN**

### **ARCHIVOS PRINCIPALES:**
- `config/ml/training.yaml` - Configuración principal
- `config/ml/training.yaml.backup` - Respaldo automático

### **ARCHIVOS DE VALIDACIÓN:**
- `validate_training_config.py` - Validador de configuración
- `configure_training.py` - Configurador interactivo

### **ARCHIVOS DE LOG:**
- `logs/training.log` - Log principal
- `logs/metrics.json` - Métricas guardadas
- `logs/audit.log` - Log de auditoría

## 🎯 **BENEFICIOS**

1. **Configuración Centralizada**: Un solo archivo para toda la configuración
2. **Validación Automática**: Verificación de consistencia y validez
3. **Configuración Interactiva**: Asistente para configuración fácil
4. **Flexibilidad**: Adaptable a diferentes sistemas y necesidades
5. **Monitoreo**: Configuración completa de alertas y métricas
6. **Mantenimiento**: Gestión automática de recursos y limpieza

## 📚 **EJEMPLOS DE USO**

### **ENTRENAMIENTO BÁSICO:**
```python
import yaml
from core.ml.training.train_direction import main

# Cargar configuración
with open('config/ml/training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Ejecutar entrenamiento
main(
    symbol="BTCUSDT",
    tf="1m",
    horizon=1,
    config=config
)
```

### **ENTRENAMIENTO CON CONFIGURACIÓN PERSONALIZADA:**
```python
# Cargar configuración base
with open('config/ml/training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Personalizar para símbolo específico
config['balance']['symbols']['BTCUSDT']['risk_per_trade'] = 0.005
config['models']['direction']['promotion_thresholds']['min_auc'] = 0.53

# Ejecutar con configuración personalizada
main(symbol="BTCUSDT", config=config)
```

### **ENTRENAMIENTO NOCTURNO:**
```python
# Configuración para entrenamiento nocturno
config['resources']['max_workers'] = 8
config['training']['validation']['n_splits'] = 10
config['monitoring']['log_level'] = 'INFO'
config['training']['checkpoints']['enabled'] = True

# Ejecutar entrenamiento nocturno
main(symbol="BTCUSDT", config=config)
```

¡La configuración robusta está lista para entrenamientos largos y eficientes! ⚙️🚀📊
