# 📈 Detección de Tendencias - Phase 1 Logs

## ✅ Nuevas Funcionalidades Implementadas

### 🕐 Timestamps Mejorados
- **Timestamp de creación**: Hora cuando se procesó la predicción
- **Timestamp de datos**: Hora de los datos históricos analizados
- **Formato**: `[21:39:02] [DIRECTION] XRPUSDT 1m: short (conf: 0.500) [data: 05:05:00]`

### 📊 Detección Automática de Tendencias
- **Algoritmo**: Detecta secuencias consecutivas del mismo tipo de predicción
- **Umbral mínimo**: 3 predicciones consecutivas para considerar una tendencia
- **Agrupación**: Por símbolo y tipo de agente (direction, regime, smc)
- **Métricas**: Muestra inicio, fin, duración y confianza máxima

### 🎯 Formato de Tendencias Detectadas
```
📈 TENDENCIAS ACTIVAS:
  ETHUSDT direction: long x8 (21:38:12→21:38:12) conf:0.70
  ETHUSDT direction: long x16 (21:38:12→21:38:13) conf:0.70
  ETHUSDT direction: short x11 (21:38:15→21:38:16) conf:0.70
```

**Explicación del formato:**
- `ETHUSDT direction`: Símbolo y tipo de agente
- `long x8`: Tipo de predicción y número de repeticiones consecutivas
- `(21:38:12→21:38:12)`: Timestamp de inicio → Timestamp de fin
- `conf:0.70`: Confianza máxima durante la tendencia

### 🔍 Información Temporal Detallada
- **Inicio de tendencia**: Timestamp cuando comenzó la secuencia
- **Fin de tendencia**: Timestamp cuando terminó la secuencia
- **Duración**: Calculada automáticamente
- **Confianza máxima**: La mayor confianza registrada durante la tendencia

## 🚀 Cómo Usar

### 1. Ejecutar GUI
```bash
python scripts/data/gui_training_monitor.py --refresh 5
```

### 2. Ver Tendencias
- Ir a la pestaña "📊 Phase 1 Logs"
- Las tendencias aparecen automáticamente en la parte superior
- Se actualizan cada 10 segundos

### 3. Interpretar los Logs
```
[21:39:02] [DIRECTION] XRPUSDT 1m: short (conf: 0.500) [data: 05:05:00]
     ↑           ↑         ↑      ↑        ↑              ↑
  Hora proc.   Agente   Símbolo  Predic.  Confianza   Hora datos
```

### 4. Filtrar Tendencias
- **Por agente**: Seleccionar direction, regime, o smc
- **Por nivel**: INFO, WARNING, ERROR según confianza
- **Auto-scroll**: Mantener activado para ver tendencias recientes

## 📊 Ejemplo de Salida Real

### Tendencias Detectadas:
```
📈 TENDENCIAS ACTIVAS:
  ETHUSDT direction: long x8 (21:38:12→21:38:12) conf:0.70
  ETHUSDT direction: long x16 (21:38:12→21:38:13) conf:0.70
  ETHUSDT direction: long x13 (21:38:13→21:38:14) conf:0.70
  ETHUSDT direction: long x7 (21:38:14→21:38:15) conf:0.70
  ETHUSDT direction: short x11 (21:38:15→21:38:16) conf:0.70
```

### Logs Individuales:
```
[21:39:02] [DIRECTION] XRPUSDT 1m: short (conf: 0.500) [data: 05:05:00]
[21:39:02] [DIRECTION] XRPUSDT 1m: long (conf: 0.500) [data: 05:04:00]
[21:39:02] [DIRECTION] XRPUSDT 1m: short (conf: 0.500) [data: 05:03:00]
```

## 🔧 Scripts de Prueba

### Probar Detección de Tendencias
```bash
python scripts/test_trend_detection.py
```

### Verificar Datos
```bash
python scripts/check_phase1_data.py
```

### Probar GUI Completo
```bash
python scripts/test_gui_logs.py
```

## 📈 Algoritmo de Detección

### 1. Agrupación
- Agrupa logs por `(símbolo, agente)`
- Ordena por timestamp de creación

### 2. Detección de Secuencias
- Identifica predicciones consecutivas del mismo `pred_label`
- Cuenta repeticiones consecutivas
- Registra confianza máxima

### 3. Criterios de Tendencias
- **Mínimo 3 predicciones** consecutivas
- **Mismo símbolo y agente**
- **Mismo tipo de predicción** (long/short, bull/bear, etc.)

### 4. Ranking
- Ordena por confianza máxima descendente
- Muestra top 5 tendencias más confiables

## 🎯 Beneficios

### Para Análisis
- **Identificación rápida** de patrones de mercado
- **Duración de tendencias** para timing de trades
- **Confianza agregada** para validación

### Para Monitoreo
- **Vista en tiempo real** de tendencias activas
- **Filtrado inteligente** por tipo de agente
- **Exportación** para análisis posterior

### Para Trading
- **Timestamps precisos** para entrada/salida
- **Detección de cambios** de tendencia
- **Validación cruzada** entre diferentes agentes

## 🚀 Próximas Mejoras

1. **Alertas de tendencias**: Notificaciones cuando se detecten tendencias fuertes
2. **Métricas avanzadas**: RSI, MACD durante tendencias
3. **Correlaciones**: Detectar tendencias simultáneas en múltiples símbolos
4. **Persistencia**: Guardar tendencias históricas en base de datos
5. **Visualización**: Gráficos de tendencias en tiempo real

## ✅ Estado Actual

- ✅ **Detección automática** de tendencias funcionando
- ✅ **Timestamps detallados** implementados
- ✅ **Interfaz visual** mejorada
- ✅ **Filtros y exportación** operativos
- ✅ **Scripts de prueba** validados

**El sistema ahora muestra claramente el inicio y fin de cada tendencia detectada, facilitando el análisis temporal de los patrones de mercado.**
