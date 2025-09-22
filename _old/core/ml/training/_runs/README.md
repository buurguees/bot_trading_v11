# 🚀 SCRIPTS DE EJECUCIÓN

## 🎯 **DESCRIPCIÓN**
Scripts para ejecutar el sistema de entrenamiento automatizado.

## 📋 **SCRIPTS DISPONIBLES**

### **1. start_training.bat**
- **Propósito:** Entrenamiento estándar con auto-backfill
- **Uso:** `core/ml/training/_runs/start_training.bat`
- **Configuración:** Auto-backfill habilitado
- **Recomendado:** No (usa auto-backfill)

### **2. start_training_no_backfill.bat**
- **Propósito:** Entrenamiento sin auto-backfill
- **Uso:** `core/ml/training/_runs/start_training_no_backfill.bat`
- **Configuración:** Sin auto-backfill
- **Recomendado:** Sí (para sistemas con features en tiempo real)

### **3. start_training_optimized.bat**
- **Propósito:** Entrenamiento optimizado con backtests históricos
- **Uso:** `core/ml/training/_runs/start_training_optimized.bat`
- **Configuración:** Backtests con 365+ días de datos
- **Recomendado:** Sí (mejor evaluación)

## ⚙️ **CONFIGURACIONES**

### **start_training_optimized.bat**
- **Ciclo:** 60 minutos
- **Backtests:** 365+ días de datos
- **Auto-backfill:** Deshabilitado
- **Umbrales:** Ajustados para objetivos realistas

### **start_training_no_backfill.bat**
- **Ciclo:** 30 minutos
- **Backtests:** 7 días de datos
- **Auto-backfill:** Deshabilitado
- **Umbrales:** Ajustados para objetivos realistas

### **start_training.bat**
- **Ciclo:** 30 minutos
- **Backtests:** 7 días de datos
- **Auto-backfill:** Habilitado
- **Umbrales:** Ajustados para objetivos realistas

## 🚀 **USO RECOMENDADO**

```bash
# Opción 1: Menú interactivo
start_training.bat

# Opción 2: Directo (recomendado)
core/ml/training/_runs/start_training_optimized.bat
```

## 📊 **MONITOREO**

Después de iniciar el entrenamiento, usa:
```bash
monitor_system.bat
```

## ⚠️ **NOTAS IMPORTANTES**

1. **Siempre usar** `start_training_optimized.bat` para mejor rendimiento
2. **Verificar** que no hay duplicados antes de iniciar
3. **Monitorear** regularmente el progreso
4. **Aplicar mantenimiento** si es necesario
