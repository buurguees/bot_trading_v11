# 📊 MONITOREO DEL SISTEMA

## 🎯 **DESCRIPCIÓN**
Scripts de monitoreo para el sistema de trading automatizado.

## 📋 **SCRIPTS DISPONIBLES**

### **1. monitor_emergency.py**
- **Propósito:** Monitoreo de emergencia del sistema
- **Uso:** `python core/ml/monitoring/monitor_emergency.py`
- **Frecuencia:** Cada 30 minutos
- **Muestra:** PnL, trades, win rate, estado del sistema

### **2. monitor_historical_backtests.py**
- **Propósito:** Monitoreo de backtests con datos históricos
- **Uso:** `python core/ml/monitoring/monitor_historical_backtests.py`
- **Frecuencia:** Cada 2 horas
- **Muestra:** Backtests con 365+ días de datos

### **3. check_recent_activity.py**
- **Propósito:** Verificar actividad reciente del sistema
- **Uso:** `python core/ml/monitoring/check_recent_activity.py`
- **Frecuencia:** Cada hora
- **Muestra:** Entrenamientos, predicciones, planes, backtests

### **4. check_duplicates.py**
- **Propósito:** Verificar duplicados en la base de datos
- **Uso:** `python core/ml/monitoring/check_duplicates.py`
- **Frecuencia:** Diariamente
- **Muestra:** Duplicados en todas las tablas

### **5. check_promotion_candidates.py**
- **Propósito:** Verificar candidatos a promoción
- **Uso:** `python core/ml/monitoring/check_promotion_candidates.py`
- **Frecuencia:** Cada 30 minutos
- **Muestra:** Modelos que pueden promoverse

### **6. check_pnl_changes.py**
- **Propósito:** Verificar cambios en PnL
- **Uso:** `python core/ml/monitoring/check_pnl_changes.py`
- **Frecuencia:** Cada hora
- **Muestra:** Cambios en PnL recientes vs históricos

## 🚀 **USO RÁPIDO**

```bash
# Monitoreo completo
monitor_system.bat

# Monitoreo específico
python core/ml/monitoring/monitor_emergency.py
```

## 📊 **MÉTRICAS CLAVE**

- **PnL Total:** Rentabilidad del sistema
- **Win Rate:** Porcentaje de trades exitosos
- **Trades Totales:** Número de operaciones
- **Modelos Promovidos:** Versiones en producción
- **Duplicados:** Integridad de datos
- **Actividad:** Estado del sistema
