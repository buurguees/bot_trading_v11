# 🌙 ENTRENAMIENTO NOCTURNO OPTIMIZADO

## 📋 **DESCRIPCIÓN**

Sistema de entrenamiento nocturno robusto con paralelización segura, gestión avanzada de memoria, manejo de interrupciones y monitoreo en tiempo real.

## ✨ **CARACTERÍSTICAS PRINCIPALES**

### **1. PARALELIZACIÓN SEGURA**
- **ProcessPoolExecutor** con `max_workers=cpu_count()-1`
- **Pool separado por símbolo** para evitar conflictos de memoria
- **Shared memory** para features comunes entre timeframes
- **Reinicio automático** de workers si exceden memoria

### **2. MANEJO DE INTERRUPCIONES**
- **Captura SIGINT/SIGTERM** gracefully
- **Guarda progreso** antes de terminar
- **Cleanup automático** de recursos
- **Recuperación** desde último checkpoint

### **3. GESTIÓN DE MEMORIA AVANZADA**
- **Límite de memoria por proceso** (2GB configurable)
- **Reinicio automático** de workers problemáticos
- **Garbage collection forzado** entre entrenamientos
- **Cache compartido** para features comunes

### **4. SISTEMA DE COLA**
- **Cola prioritaria** por volumen de símbolo
- **Retry automático** (máximo 3 intentos)
- **Skip temporal** si símbolo falla consistentemente
- **Gestión inteligente** de recursos

### **5. MONITOREO EN TIEMPO REAL**
- **Dashboard web** con Flask
- **Métricas de sistema**: CPU, RAM, disco
- **Estimación de tiempo** restante
- **Estado de trabajos** en tiempo real

### **6. CONFIGURACIÓN DINÁMICA**
- **Ajuste automático** de paralelismo
- **Escalado automático** basado en recursos
- **Configuración adaptativa** por sistema

## 🛠️ **INSTALACIÓN**

### **1. Instalación automática:**
```bash
install_night_training.bat
```

### **2. Instalación manual:**
```bash
# Instalar dependencias
pip install psutil>=5.9.0 flask>=2.0.0 requests>=2.25.0

# Crear configuración
python core/ml/training/night_train/configure_night_training.py --create

# Optimizar para sistema
python core/ml/training/night_train/configure_night_training.py --optimize
```

## 🚀 **USO**

### **INICIAR ENTRENAMIENTO:**
```bash
# Básico
python core/ml/training/night_train/start_night_training.py

# Con dashboard
python core/ml/training/night_train/start_night_training.py --dashboard --port 5000

# Verificar requisitos
python core/ml/training/night_train/start_night_training.py --check-requirements

# Estimar tiempo
python core/ml/training/night_train/start_night_training.py --estimate
```

### **MONITOREAR:**
```bash
# Monitor básico
python core/ml/monitoring/monitor_night_training.py

# Monitor detallado
python core/ml/monitoring/monitor_night_training.py --detailed

# Monitor continuo
python core/ml/monitoring/monitor_night_training.py --continuous --interval 30

# Verificar salud
python core/ml/monitoring/monitor_night_training.py --health
```

### **CONFIGURAR:**
```bash
# Ver configuración actual
python core/ml/training/night_train/configure_night_training.py --validate

# Optimizar para sistema
python core/ml/training/night_train/configure_night_training.py --optimize

# Agregar símbolo
python core/ml/training/night_train/configure_night_training.py --add-symbol MATICUSDT

# Configurar timeframes
python core/ml/training/night_train/configure_night_training.py --set-timeframes 1m 5m 1h

# Configurar umbrales
python core/ml/training/night_train/configure_night_training.py --set-thresholds 0.52 0.26 0.50
```

## ⚙️ **CONFIGURACIÓN**

### **ARCHIVO PRINCIPAL:**
```yaml
# config/ml/night_training.yaml
symbols: [BTCUSDT, ETHUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, XRPUSDT]
timeframes: [1m, 5m, 15m, 1h, 4h, 1d]
horizons: [1, 3, 5]

# Paralelización
num_workers: 4
max_memory_per_process: 2.0
memory_threshold: 0.85

# Validación
n_splits: 5
embargo_minutes: 30

# Promoción
promote_if:
  min_auc: 0.52
  max_brier: 0.26
  min_acc: 0.50

# Dashboard
dashboard:
  enabled: true
  port: 5000
```

### **VARIABLES DE ENTORNO:**
```bash
# config/training.env
BT_NIGHT_TRAINING=true
BT_NIGHT_WORKERS=4
BT_NIGHT_MEMORY_LIMIT=2.0
BT_NIGHT_DASHBOARD_PORT=5000
```

## 📊 **DASHBOARD WEB**

### **ACCESO:**
- **URL**: `http://localhost:5000`
- **API**: `http://localhost:5000/api/status`

### **MÉTRICAS MOSTRADAS:**
- **Sistema**: CPU, RAM, disco
- **Progreso**: Completados, en progreso, fallidos
- **Tiempo**: Estimado restante, velocidad
- **Trabajos**: Estado reciente por símbolo

## 🔧 **GESTIÓN DE RECURSOS**

### **MEMORIA:**
- **Límite por proceso**: 2GB (configurable)
- **Umbral de sistema**: 85%
- **Reinicio automático** si se excede
- **Cache compartido** para eficiencia

### **CPU:**
- **Workers**: `cpu_count() - 1`
- **Escalado automático** según carga
- **Balanceo de carga** inteligente

### **DISCO:**
- **Checkpoints** cada 30 segundos
- **Logs rotativos** (10MB, 5 backups)
- **Limpieza automática** de cache

## 📈 **MONITOREO AVANZADO**

### **MÉTRICAS DEL SISTEMA:**
```bash
# CPU usage
# Memory usage (total/used/available)
# Disk usage (free/total)
# Process count
```

### **MÉTRICAS DE ENTRENAMIENTO:**
```bash
# Jobs completed/failed/running
# Average AUC per symbol
# Training time per job
# Memory usage per process
```

### **ALERTAS:**
- **Memoria > 90%**
- **CPU > 95%**
- **Disco > 85%**
- **> 5 trabajos fallidos**

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **MEMORIA INSUFICIENTE:**
```bash
# Reducir workers
python core/ml/training/night_train/configure_night_training.py --optimize

# Reducir memoria por proceso
# Editar config/ml/night_training.yaml
max_memory_per_process: 1.0
```

### **PROCESOS COLGADOS:**
```bash
# Verificar procesos
python core/ml/monitoring/monitor_night_training.py --health

# Reiniciar entrenamiento
# El sistema reinicia automáticamente workers problemáticos
```

### **DASHBOARD NO FUNCIONA:**
```bash
# Verificar puerto
netstat -an | findstr :5000

# Cambiar puerto
python core/ml/training/night_train/start_night_training.py --port 5001
```

## 📁 **ARCHIVOS GENERADOS**

### **CHECKPOINTS:**
- `logs/night_training_checkpoint.json`

### **LOGS:**
- `logs/batch_train_night.log`
- `logs/night_training_launcher.log`

### **ARTEFACTOS:**
- `artifacts/direction/{symbol}_{tf}_H{horizon}_logreg_optimized.pkl`

### **RESULTADOS:**
- `logs/batch_train_results_{timestamp}.json`

## 🔄 **FLUJO DE TRABAJO**

### **1. INICIO:**
1. Verificar requisitos del sistema
2. Cargar configuración
3. Crear cola de trabajos prioritaria
4. Iniciar pool de workers
5. Lanzar dashboard (opcional)

### **2. PROCESAMIENTO:**
1. Obtener trabajo de la cola
2. Asignar a worker disponible
3. Monitorear memoria del proceso
4. Reiniciar si excede límites
5. Actualizar progreso

### **3. FINALIZACIÓN:**
1. Esperar completar trabajos activos
2. Guardar resultados finales
3. Limpiar recursos
4. Generar reporte

## 📊 **RENDIMIENTO ESPERADO**

### **SISTEMA BÁSICO (8GB RAM, 4 cores):**
- **Workers**: 3
- **Tiempo estimado**: 4-6 horas
- **Memoria por proceso**: 1.5GB

### **SISTEMA MEDIANO (16GB RAM, 8 cores):**
- **Workers**: 7
- **Tiempo estimado**: 2-3 horas
- **Memoria por proceso**: 2GB

### **SISTEMA AVANZADO (32GB RAM, 16 cores):**
- **Workers**: 15
- **Tiempo estimado**: 1-2 horas
- **Memoria por proceso**: 2GB

## 🎯 **BENEFICIOS**

1. **Escalabilidad**: Maneja cualquier cantidad de símbolos/timeframes
2. **Robustez**: Recuperación automática de fallos
3. **Eficiencia**: Uso óptimo de recursos del sistema
4. **Visibilidad**: Monitoreo completo en tiempo real
5. **Flexibilidad**: Configuración adaptativa
6. **Confiabilidad**: Sistema de checkpoints y recuperación

## 📚 **EJEMPLOS DE USO**

### **ENTRENAMIENTO BÁSICO:**
```bash
# Configurar y ejecutar
python core/ml/training/night_train/configure_night_training.py --create --optimize
python core/ml/training/night_train/start_night_training.py --dashboard
```

### **ENTRENAMIENTO PERSONALIZADO:**
```bash
# Agregar símbolos específicos
python core/ml/training/night_train/configure_night_training.py --add-symbol MATICUSDT --add-symbol AVAXUSDT

# Configurar timeframes específicos
python core/ml/training/night_train/configure_night_training.py --set-timeframes 1m 5m 1h

# Ajustar umbrales
python core/ml/training/night_train/configure_night_training.py --set-thresholds 0.55 0.24 0.52

# Ejecutar
python core/ml/training/night_train/start_night_training.py
```

### **MONITOREO AVANZADO:**
```bash
# Monitor continuo con alertas
python core/ml/monitoring/monitor_night_training.py --continuous --interval 10

# Verificar salud del sistema
python core/ml/monitoring/monitor_night_training.py --health

# Dashboard en puerto personalizado
python core/ml/training/night_train/start_night_training.py --port 8080
```

¡El sistema de entrenamiento nocturno está listo para procesar grandes volúmenes de datos de forma eficiente y robusta! 🌙🚀
