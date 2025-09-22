# 📊 Upgrade de Tabla TRADES - Bot Trading V11

## 🎯 Objetivo
Mejorar la tabla `trading.trades` con nuevas columnas para mejor tracking y análisis de operaciones.

## ✅ Cambios Implementados

### 1. **Nuevas Columnas Añadidas**

| Columna | Tipo | Descripción | Default |
|---------|------|-------------|---------|
| `plan_id` | BIGINT | ID del tradeplan usado para la operación | NULL |
| `order_ids` | JSONB | IDs de órdenes en el exchange | `{}` |
| `fees_paid` | NUMERIC(15,8) | Comisiones pagadas | 0.0 |
| `slip_bps` | NUMERIC(8,2) | Slippage en basis points | 0.0 |
| `entry_balance` | NUMERIC(15,8) | Balance al entrar | NULL |
| `exit_balance` | NUMERIC(15,8) | Balance al salir | NULL |
| `symbol_id` | BIGINT | FK a tabla symbols | NULL |

### 2. **Estandarización de Side**
- ✅ Creado ENUM `trading.trade_side` con valores: `'long'`, `'short'`
- ✅ Columna `side` convertida a ENUM
- ✅ Migración automática de datos existentes

### 3. **Foreign Keys y Constraints**
- ✅ FK a `trading.symbols(id)` en `symbol_id`
- ✅ Constraints de validación:
  - `fees_paid >= 0`
  - `slip_bps >= 0`
  - `entry_balance > 0` (si no es NULL)
  - `exit_balance > 0` (si no es NULL)

### 4. **Índices Optimizados**
- ✅ `idx_trades_plan_id` - Para búsquedas por plan
- ✅ `idx_trades_symbol_id` - Para FK a symbols
- ✅ `idx_trades_entry_balance` - Para análisis de balance
- ✅ `idx_trades_symbol_balance` - Compuesto para análisis

### 5. **Código Actualizado**
- ✅ Función `insert_trade()` actualizada con nuevas columnas
- ✅ Función `get_symbol_id()` para obtener ID de símbolo
- ✅ Manejo automático de valores por defecto
- ✅ Conversión automática de dict a JSON para `order_ids`

## 📊 Estructura Final de la Tabla

```sql
CREATE TABLE trading.trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    side trading.trade_side NOT NULL,  -- ENUM
    quantity NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    pnl NUMERIC NOT NULL,
    entry_timestamp TIMESTAMPTZ NOT NULL,
    exit_timestamp TIMESTAMPTZ NULL,
    duration INTERVAL NULL,
    leverage NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- NUEVAS COLUMNAS
    plan_id BIGINT NULL,
    order_ids JSONB DEFAULT '{}',
    fees_paid NUMERIC(15,8) DEFAULT 0.0,
    slip_bps NUMERIC(8,2) DEFAULT 0.0,
    entry_balance NUMERIC(15,8) NULL,
    exit_balance NUMERIC(15,8) NULL,
    symbol_id BIGINT NULL,
    
    -- CONSTRAINTS
    CONSTRAINT fk_trades_symbol_id FOREIGN KEY (symbol_id) REFERENCES trading.symbols(id),
    CONSTRAINT chk_trades_fees_paid CHECK (fees_paid >= 0),
    CONSTRAINT chk_trades_slip_bps CHECK (slip_bps >= 0),
    CONSTRAINT chk_trades_entry_balance CHECK (entry_balance IS NULL OR entry_balance > 0),
    CONSTRAINT chk_trades_exit_balance CHECK (exit_balance IS NULL OR exit_balance > 0)
);
```

## 🔍 Consultas Útiles

### Análisis de Rendimiento por Balance
```sql
SELECT 
    symbol,
    COUNT(*) as trades,
    AVG(entry_balance) as avg_entry_balance,
    AVG(exit_balance) as avg_exit_balance,
    AVG(exit_balance - entry_balance) as avg_balance_change,
    SUM(pnl) as total_pnl
FROM trading.trades 
WHERE entry_balance IS NOT NULL AND exit_balance IS NOT NULL
GROUP BY symbol
ORDER BY total_pnl DESC;
```

### Análisis de Fees y Slippage
```sql
SELECT 
    symbol,
    COUNT(*) as trades,
    AVG(fees_paid) as avg_fees,
    AVG(slip_bps) as avg_slippage,
    SUM(fees_paid) as total_fees
FROM trading.trades 
GROUP BY symbol
ORDER BY total_fees DESC;
```

### Trades por Plan
```sql
SELECT 
    plan_id,
    symbol,
    COUNT(*) as trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl
FROM trading.trades 
WHERE plan_id IS NOT NULL
GROUP BY plan_id, symbol
ORDER BY total_pnl DESC;
```

### Análisis de Order IDs
```sql
SELECT 
    symbol,
    COUNT(*) as trades,
    COUNT(CASE WHEN order_ids != '{}' THEN 1 END) as trades_with_orders,
    AVG(fees_paid) as avg_fees
FROM trading.trades 
GROUP BY symbol
ORDER BY trades_with_orders DESC;
```

## 🚀 Beneficios del Upgrade

### 1. **Mejor Tracking de Operaciones**
- Vinculación con tradeplans para trazabilidad completa
- IDs de órdenes del exchange para reconciliación
- Balance tracking para verificar cálculos

### 2. **Análisis Avanzado**
- Análisis de fees y slippage por operación
- Verificación de cálculos de PnL vs balance real
- Tracking de rendimiento por plan de trading

### 3. **Integridad de Datos**
- ENUM para side previene valores inválidos
- Constraints aseguran datos consistentes
- FK a symbols mantiene integridad referencial

### 4. **Optimización de Consultas**
- Índices específicos para análisis comunes
- Consultas más eficientes por plan y símbolo
- Mejor rendimiento en análisis de balance

## 📝 Uso en Código

### Insertar Trade Básico
```python
from core.data.database import insert_trade

trade_data = {
    "symbol": "BTCUSDT",
    "side": "long",  # ENUM: 'long' o 'short'
    "quantity": 0.001,
    "price": 50000.0,
    "pnl": 15.5,
    "entry_timestamp": datetime.now(timezone.utc),
    "exit_timestamp": datetime.now(timezone.utc),
    "leverage": 10.0,
    # Campos opcionales (se llenan automáticamente)
    "plan_id": 12345,
    "order_ids": {"entry": "order_123", "exit": "order_456"},
    "fees_paid": 0.5,
    "slip_bps": 2.5,
    "entry_balance": 1000.0,
    "exit_balance": 1015.5
}

success = insert_trade(trade_data)
```

### Obtener Symbol ID
```python
from core.data.database import get_symbol_id

symbol_id = get_symbol_id("BTCUSDT")
```

## ✅ Estado: COMPLETADO

El upgrade de la tabla `trades` está completamente implementado y probado. Todas las nuevas columnas están funcionando correctamente y el código ha sido actualizado para soportar la nueva estructura.

## 📁 Archivos Modificados

- `core/data/database.py` - Funciones `insert_trade()` y `get_symbol_id()`
- `scripts/maintenance/upgrade_trades_table.sql` - Script de migración
- `TRADES_UPGRADE_SUMMARY.md` - Este documento

## 🎉 Resultado Final

La tabla `trades` ahora proporciona un tracking completo y detallado de todas las operaciones, permitiendo análisis avanzados de rendimiento, fees, slippage y balance. El sistema está listo para operaciones en tiempo real con trazabilidad completa.
