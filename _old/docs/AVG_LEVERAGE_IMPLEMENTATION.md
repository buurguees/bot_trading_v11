# 📊 Implementación de AVG_LEVERAGE en Strategy Memory

## 🎯 Objetivo
Añadir el campo `avg_leverage` a la tabla `strategy_memory` para rastrear el leverage promedio usado en cada estrategia de trading.

## ✅ Cambios Implementados

### 1. **Modificación de la Base de Datos**
- ✅ Añadida columna `avg_leverage DOUBLE PRECISION DEFAULT 0.0` a la tabla `trading.strategy_memory`
- ✅ Actualizados los datos existentes con el promedio de leverage calculado desde `strategy_samples`

### 2. **Actualización del Código**
- ✅ Modificada la función `update_memory()` en `core/ml/backtests/strategy_memory.py`
- ✅ Implementada lógica de agrupación por firma de estrategia
- ✅ Cálculo automático de `avg_leverage` para cada grupo de trades
- ✅ Actualización correcta de estadísticas al fusionar grupos existentes

### 3. **Lógica de Agrupación Mejorada**
- ✅ Los trades se agrupan por firma de estrategia (características similares)
- ✅ Cada grupo calcula sus propias estadísticas:
  - `n_trades`: Número de trades en el grupo
  - `win_rate`: Porcentaje de trades ganadores
  - `avg_pnl`: PnL promedio del grupo
  - `avg_leverage`: Leverage promedio del grupo
  - `avg_hold_bars`: Promedio de barras mantenidas

## 🔧 Funcionamiento

### Agrupación por Firma
Cada trade se clasifica por su "firma" basada en:
- **RSI bucket**: `r0` (<30), `r1` (30-50), `r2` (50-70), `r3` (>70)
- **EMA state**: `mix`, `up`, `down`
- **MACD sign**: `+` (positivo), `-` (negativo)
- **SuperTrend**: `1` (alcista), `-1` (bajista)
- **Trend 1h**: `0`, `1`

**Ejemplo de firma**: `r1|ema:mix|macd:+|st:1|t1h:0`

### Cálculo de Estadísticas
Para cada grupo de trades con la misma firma:
```python
n_trades = len(trades_group)
wins = sum(1 for t in trades_group if t["pnl"] > 0)
win_rate = wins / n_trades
avg_pnl = sum(t["pnl"] for t in trades_group) / n_trades
avg_leverage = sum(t["leverage"] for t in trades_group) / n_trades
```

## 📊 Datos Actuales

### Estadísticas Generales
- **Total estrategias**: 11
- **Leverage promedio**: 9.98
- **Leverage mínimo**: 9.78
- **Leverage máximo**: 10.00
- **Estrategias con leverage**: 11 (100%)

### Top Estrategias por Leverage
1. **BTCUSDT-5m**: 264 trades, leverage=10.00, WR=0.538, PnL=-1.1374
2. **ADAUSDT-5m**: 264 trades, leverage=10.00, WR=0.633, PnL=0.5717
3. **XRPUSDT-5m**: 264 trades, leverage=10.00, WR=0.591, PnL=0.1898
4. **DOGEUSDT-5m**: 1108 trades, leverage=10.00, WR=0.354, PnL=-2.9529
5. **SOLUSDT-5m**: 528 trades, leverage=10.00, WR=0.413, PnL=-0.8030

## 🔍 Consultas Útiles

### Ver estrategias con leverage específico
```sql
SELECT symbol, timeframe, n_trades, avg_leverage, win_rate, avg_pnl
FROM trading.strategy_memory 
WHERE avg_leverage BETWEEN 8.0 AND 12.0
ORDER BY avg_leverage DESC;
```

### Estrategias más rentables por leverage
```sql
SELECT symbol, timeframe, n_trades, avg_leverage, win_rate, avg_pnl,
       (avg_pnl * n_trades) as total_pnl
FROM trading.strategy_memory 
WHERE n_trades >= 100
ORDER BY total_pnl DESC;
```

### Análisis de leverage por símbolo
```sql
SELECT symbol, 
       COUNT(*) as strategies,
       AVG(avg_leverage) as avg_leverage,
       MIN(avg_leverage) as min_leverage,
       MAX(avg_leverage) as max_leverage
FROM trading.strategy_memory 
GROUP BY symbol
ORDER BY avg_leverage DESC;
```

## 🚀 Beneficios

1. **Análisis de Riesgo**: Identificar estrategias que usan leverage excesivo
2. **Optimización**: Ajustar leverage basado en rendimiento histórico
3. **Monitoreo**: Rastrear cambios en el uso de leverage a lo largo del tiempo
4. **Backtesting**: Evaluar estrategias con diferentes niveles de leverage
5. **Gestión de Capital**: Optimizar asignación de capital por estrategia

## 📝 Archivos Modificados

- `core/ml/backtests/strategy_memory.py` - Lógica principal de memoria de estrategias
- `scripts/maintenance/add_avg_leverage_column.sql` - Script de migración de BD
- `AVG_LEVERAGE_IMPLEMENTATION.md` - Este documento

## ✅ Pruebas Realizadas

- ✅ Creación de columna en base de datos
- ✅ Actualización de datos existentes
- ✅ Lógica de agrupación por firma
- ✅ Cálculo correcto de estadísticas
- ✅ Integración con código existente
- ✅ Pruebas con datos reales

## 🎉 Estado: COMPLETADO

La implementación de `avg_leverage` está completamente funcional y integrada en el sistema de memoria de estrategias. El sistema ahora rastrea automáticamente el leverage promedio usado en cada estrategia, proporcionando información valiosa para el análisis y optimización del trading.
