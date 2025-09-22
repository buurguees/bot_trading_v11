# 📁 SCRIPTS GENERALES

## 🎯 **DESCRIPCIÓN**
Scripts generales para configuración, mantenimiento y monitoreo del sistema.

## 📂 **ESTRUCTURA**

### **📁 setup/**
Scripts de configuración inicial del sistema.

#### **setup_full_data.py**
- **Propósito:** Configurar datos completos del sistema
- **Uso:** `python scripts/setup/setup_full_data.py`
- **Incluye:** Descarga histórica, cálculo de features, verificación

#### **verify_data_setup.py**
- **Propósito:** Verificar configuración de datos
- **Uso:** `python scripts/setup/verify_data_setup.py`
- **Verifica:** Cobertura, features, integridad

### **📁 maintenance/**
Scripts de mantenimiento del sistema.

#### **fix_prediction_duplicates.py**
- **Propósito:** Limpiar duplicados de predicciones
- **Uso:** `python scripts/maintenance/fix_prediction_duplicates.py`
- **Incluye:** Limpieza, índice único, verificación

#### **emergency_fix.py**
- **Propósito:** Aplicar correcciones de emergencia
- **Uso:** `python scripts/maintenance/emergency_fix.py`
- **Incluye:** Ajuste de umbrales, limpieza de memoria

## 🚀 **USO RÁPIDO**

```bash
# Menú interactivo
maintenance.bat

# Scripts específicos
python scripts/setup/setup_full_data.py
python scripts/maintenance/fix_prediction_duplicates.py
```

## 📋 **FLUJO RECOMENDADO**

### **1. Configuración Inicial**
```bash
# 1. Configurar datos completos
python scripts/setup/setup_full_data.py

# 2. Verificar configuración
python scripts/setup/verify_data_setup.py
```

### **2. Mantenimiento Regular**
```bash
# 1. Verificar duplicados
python scripts/maintenance/fix_prediction_duplicates.py

# 2. Aplicar correcciones si es necesario
python scripts/maintenance/emergency_fix.py
```

## ⚠️ **NOTAS IMPORTANTES**

1. **Ejecutar setup** antes del primer uso
2. **Verificar duplicados** regularmente
3. **Aplicar emergency_fix** si hay problemas de rendimiento
4. **Backup** de configuraciones antes de cambios
