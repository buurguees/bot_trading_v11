# 📁 ESTRUCTURA DEL PROYECTO BOT TRADING V11

## 🎯 **DESCRIPCIÓN GENERAL**
Este proyecto implementa un sistema de trading automatizado con machine learning, backtesting y monitoreo en tiempo real.

## 📂 **ESTRUCTURA DE DIRECTORIOS**

```
bot_trading_v11/
├── 📁 core/                              # Código principal del sistema
│   ├── 📁 ml/                           # Machine Learning
│   │   ├── 📁 training/                 # Entrenamiento de modelos
│   │   │   ├── 📁 daily_train/         # Entrenamiento diario
│   │   │   └── 📁 _runs/               # Scripts de ejecución
│   │   │       ├── start_training.bat
│   │   │       ├── start_training_no_backfill.bat
│   │   │       └── start_training_optimized.bat
│   │   ├── 📁 backtests/               # Backtesting
│   │   │   └── 📁 _scripts/            # Scripts de backtesting
│   │   └── 📁 monitoring/              # Monitoreo del sistema
│   │       ├── check_duplicates.py
│   │       ├── check_recent_activity.py
│   │       ├── check_pnl_changes.py
│   │       ├── check_promotion_candidates.py
│   │       ├── monitor_emergency.py
│   │       └── monitor_historical_backtests.py
│   ├── 📁 data/                        # Gestión de datos
│   └── 📁 features/                    # Cálculo de features
├── 📁 config/                          # Configuraciones
│   ├── 📁 ml/                         # Configuración ML
│   ├── 📁 system/                     # Configuración sistema
│   └── 📁 trading/                    # Configuración trading
├── 📁 scripts/                        # Scripts generales
│   ├── 📁 setup/                      # Scripts de configuración
│   │   ├── setup_full_data.py
│   │   └── verify_data_setup.py
│   ├── 📁 maintenance/                # Scripts de mantenimiento
│   │   ├── fix_prediction_duplicates.py
│   │   └── emergency_fix.py
│   └── 📁 monitoring/                 # Scripts de monitoreo (duplicados)
├── 📁 logs/                           # Logs del sistema
├── 📁 artifacts/                      # Modelos entrenados
├── 📁 data/                           # Datos históricos
├── 📁 db/                             # Base de datos
└── 📁 tests/                          # Tests del sistema
```

## 🚀 **SCRIPTS PRINCIPALES**

### **Entrenamiento**
- `start_training.bat` - Menú principal de entrenamiento
- `core/ml/training/_runs/start_training_optimized.bat` - Entrenamiento con backtests históricos

### **Monitoreo**
- `monitor_system.bat` - Menú principal de monitoreo
- `core/ml/monitoring/monitor_emergency.py` - Monitoreo de emergencia
- `core/ml/monitoring/monitor_historical_backtests.py` - Monitoreo de backtests históricos

### **Mantenimiento**
- `maintenance.bat` - Menú principal de mantenimiento
- `scripts/maintenance/fix_prediction_duplicates.py` - Limpiar duplicados
- `scripts/maintenance/emergency_fix.py` - Correcciones de emergencia

### **Configuración**
- `scripts/setup/setup_full_data.py` - Configurar datos completos
- `scripts/setup/verify_data_setup.py` - Verificar configuración

## 📋 **FLUJO DE TRABAJO RECOMENDADO**

### **1. Configuración Inicial**
```bash
# 1. Configurar datos completos
maintenance.bat → Opción 3

# 2. Verificar configuración
maintenance.bat → Opción 4
```

### **2. Entrenamiento**
```bash
# Iniciar entrenamiento optimizado
start_training.bat → Opción 3
```

### **3. Monitoreo**
```bash
# Monitorear sistema
monitor_system.bat → Opción 1 (emergencia)
monitor_system.bat → Opción 2 (backtests históricos)
```

### **4. Mantenimiento**
```bash
# Limpiar duplicados si es necesario
maintenance.bat → Opción 1

# Aplicar correcciones de emergencia
maintenance.bat → Opción 2
```

## 🔧 **CONFIGURACIONES IMPORTANTES**

### **Variables de Entorno** (`config/training.env`)
- `BT_RETRAIN_MINUTES=60` - Ciclo de entrenamiento
- `BT_OOS_DAYS=365` - Días de backtesting
- `BT_SKIP_BACKFILL=true` - Sin auto-backfill

### **Configuración ML** (`config/ml/training.yaml`)
- Objetivos de balance por símbolo
- Umbrales de promoción ajustados
- Configuración de riesgo

### **Configuración Trading** (`config/trading/symbols.yaml`)
- Símbolos y timeframes
- Apalancamiento máximo reducido
- Parámetros de contratos

## 📊 **MONITOREO RECOMENDADO**

### **Cada 30 minutos:**
- `monitor_system.bat` → Opción 1 (emergencia)

### **Cada 2 horas:**
- `monitor_system.bat` → Opción 2 (backtests históricos)

### **Diariamente:**
- `monitor_system.bat` → Opción 3 (actividad reciente)

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **Si hay duplicados:**
```bash
maintenance.bat → Opción 1
```

### **Si el rendimiento es malo:**
```bash
maintenance.bat → Opción 2
```

### **Si no hay actividad:**
```bash
monitor_system.bat → Opción 3
```

## 📝 **NOTAS IMPORTANTES**

1. **Siempre usar** `start_training_optimized.bat` para entrenamiento
2. **Monitorear regularmente** con los scripts de monitoreo
3. **Aplicar mantenimiento** cuando sea necesario
4. **Verificar logs** en la carpeta `logs/`
5. **Backup de configuraciones** antes de cambios importantes

## 🎯 **OBJETIVOS DEL SISTEMA**

- **Entrenamiento continuo** con datos históricos completos
- **Promoción automática** de modelos mejorados
- **Backtesting robusto** con 365+ días de datos
- **Monitoreo en tiempo real** del rendimiento
- **Mantenimiento automático** de la base de datos
