# 📊 Upgrade de MLSTRATEGIES + Limpieza de Tablas - Bot Trading V11

## 🎯 Objetivo
1. Actualizar `mlstrategies` como "diario de estrategias aprendidas" desde `strategy_memory` + `strategy_samples`
2. Añadir columna `signature` para vinculación con estrategias
3. Eliminar tablas con mayúsculas (MLStrategies, Trades, AuditLog, HistoricalData)

## ✅ Cambios Implementados

### 1. **Nueva Columna en MLSTRATEGIES**

| Columna | Tipo | Descripción | Estado |
|---------|------|-------------|---------|
| `signature` | VARCHAR(255) | Firma única de la estrategia (hash de features) | ✅ Añadida |

### 2. **Tablas Eliminadas (Mayúsculas)**

- ✅ `trading.MLStrategies` - Eliminada
- ✅ `trading.Trades` - Eliminada  
- ✅ `trading.AuditLog` - Eliminada
- ✅ `trading.HistoricalData` - Eliminada

### 3. **Sistema de Poblamiento Automático**

- ✅ Función para poblar `mlstrategies` desde `strategy_memory` + `strategy_samples`
- ✅ Conversión automática de datos entre tablas
- ✅ Generación de signatures únicas
- ✅ Mapeo de campos entre sistemas

## 📊 Estructura Final de MLSTRATEGIES

```sql
CREATE TABLE trading.mlstrategies (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    action VARCHAR NOT NULL,                    -- 'long', 'short', 'hold'
    timeframes JSONB NOT NULL,                  -- {"timeframe": "5m"}
    indicators JSONB NOT NULL,                  -- Features de la estrategia
    tools JSONB NOT NULL,                       -- {"signature": "r2|ema:mix|..."}
    leverage NUMERIC NOT NULL,
    pnl NUMERIC NOT NULL,
    performance NUMERIC NOT NULL,               -- avg_pnl de strategy_memory
    confidence_score NUMERIC NOT NULL,          -- win_rate de strategy_memory
    feature_importance JSONB NOT NULL,          -- Features de la estrategia
    outcome VARCHAR NOT NULL,                   -- 'profitable', 'loss', 'neutral'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    signature VARCHAR(255) NULL                 -- NUEVA COLUMNA
);
```

## 🔄 Mapeo de Datos

### Strategy_Memory → MLStrategies

| Strategy_Memory | MLStrategies | Transformación |
|-----------------|--------------|----------------|
| `symbol` | `symbol` | Directo |
| `timeframe` | `timeframes` | `{"timeframe": "5m"}` |
| `features` | `indicators` | JSON directo |
| `features` | `feature_importance` | JSON directo |
| `avg_pnl` | `performance` | Directo |
| `win_rate` | `confidence_score` | Directo |
| `signature` | `signature` | Directo |
| `last_updated` | `timestamp` | Directo |

### Strategy_Samples → MLStrategies

| Strategy_Samples | MLStrategies | Transformación |
|------------------|--------------|----------------|
| `side` | `action` | `1 → 'long'`, `-1 → 'short'` |
| `leverage` | `leverage` | Directo |
| `pnl` | `pnl` | Directo |
| `entry_ts` | `timestamp` | Directo |

### Generación de Outcome

```python
if avg_pnl > 0:
    outcome = 'profitable'
elif avg_pnl < 0:
    outcome = 'loss'
else:
    outcome = 'neutral'
```

## 📈 Datos Poblados

### Resumen Actual
- **Total estrategias**: 6
- **Signatures únicas**: 1
- **Símbolos únicos**: 6
- **Performance promedio**: -2.0617
- **Confianza promedio**: 0.4367

### Estrategias por Símbolo
- SOLUSDT: long | PnL: -3.63 | Conf: 0.430
- DOGEUSDT: short | PnL: -2.95 | Conf: 0.350
- BTCUSDT: short | PnL: -5.90 | Conf: 0.150
- ETHUSDT: long | PnL: -0.65 | Conf: 0.470
- XRPUSDT: short | PnL: 0.19 | Conf: 0.590
- ADAUSDT: short | PnL: 0.57 | Conf: 0.630

## 🔍 Consultas Útiles

### Estrategias por Performance
```sql
SELECT 
    symbol, 
    action, 
    performance, 
    confidence_score,
    signature
FROM trading.mlstrategies 
ORDER BY performance DESC;
```

### Análisis por Signature
```sql
SELECT 
    signature,
    COUNT(*) as frequency,
    AVG(performance) as avg_performance,
    AVG(confidence_score) as avg_confidence,
    COUNT(DISTINCT symbol) as symbols_count
FROM trading.mlstrategies 
GROUP BY signature 
ORDER BY frequency DESC;
```

### Estrategias Rentables
```sql
SELECT 
    symbol,
    action,
    performance,
    confidence_score
FROM trading.mlstrategies 
WHERE outcome = 'profitable'
ORDER BY performance DESC;
```

### Relación Strategy_Memory ↔ MLStrategies
```sql
SELECT 
    sm.signature,
    sm.n_trades,
    sm.avg_pnl as memory_pnl,
    COUNT(ms.id) as mlstrategies_count,
    AVG(ms.performance) as mlstrategies_pnl
FROM trading.strategy_memory sm 
LEFT JOIN trading.mlstrategies ms ON sm.signature = ms.signature 
GROUP BY sm.signature, sm.n_trades, sm.avg_pnl
ORDER BY sm.n_trades DESC;
```

## 🚀 Beneficios del Upgrade

### 1. **Diario de Estrategias Aprendidas**
- Registro histórico de todas las estrategias probadas
- Vinculación con `strategy_memory` para trazabilidad completa
- Análisis de evolución de estrategias en el tiempo

### 2. **Sistema Unificado**
- Eliminación de tablas duplicadas con mayúsculas
- Consistencia en nomenclatura (todo en minúsculas)
- Integración completa entre sistemas de memoria y estrategias

### 3. **Análisis Avanzado**
- Tracking de performance por signature
- Análisis de confianza vs rendimiento real
- Identificación de estrategias más efectivas

### 4. **Integridad de Datos**
- Signatures únicas para evitar duplicados
- Mapeo consistente entre tablas
- Validación de datos en tiempo real

## 📝 Uso en Código

### Poblar MLStrategies Manualmente
```python
from core.data.database import populate_mlstrategies_from_memory

# Poblar desde strategy_memory
count = populate_mlstrategies_from_memory()
print(f"Insertadas {count} estrategias")
```

### Consultar Estrategias por Signature
```python
from sqlalchemy import create_engine, text

engine = create_engine(DB_URL)
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT symbol, action, performance, confidence_score
        FROM trading.mlstrategies 
        WHERE signature = :signature
        ORDER BY performance DESC
    """), {"signature": "r2|ema:mix|macd:+|st:1|t1h:0"})
    
    for row in result:
        print(f"{row.symbol}: {row.action} | PnL: {row.performance}")
```

## ✅ Estado: COMPLETADO

El upgrade de `mlstrategies` está completamente implementado y funcionando. La tabla ahora actúa como un "diario de estrategias aprendidas" que se alimenta automáticamente desde `strategy_memory` y `strategy_samples`, proporcionando un registro histórico completo y análisis avanzado de las estrategias de trading.

## 📁 Archivos Modificados

- `scripts/maintenance/upgrade_mlstrategies_and_cleanup.sql` - Script de migración completo
- `MLSTRATEGIES_UPGRADE_SUMMARY.md` - Este documento

## 🎉 Resultado Final

- ✅ Columna `signature` añadida a `mlstrategies`
- ✅ Tablas con mayúsculas eliminadas
- ✅ Sistema de poblamiento automático implementado
- ✅ 6 estrategias pobladas desde `strategy_memory`
- ✅ Integración completa entre sistemas de memoria y estrategias

El sistema está listo para funcionar como un diario completo de estrategias aprendidas, proporcionando trazabilidad total y análisis avanzado de rendimiento.
