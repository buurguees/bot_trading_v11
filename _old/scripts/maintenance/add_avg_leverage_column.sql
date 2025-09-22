-- =====================================================
-- AÑADIR COLUMNA AVG_LEVERAGE A STRATEGY_MEMORY
-- =====================================================
-- Script para añadir la columna avg_leverage a la tabla strategy_memory
-- y actualizar los datos existentes con el promedio de leverage
-- =====================================================

\echo 'Añadiendo columna avg_leverage a strategy_memory...'

-- 1. Añadir la columna avg_leverage
ALTER TABLE trading.strategy_memory 
ADD COLUMN IF NOT EXISTS avg_leverage DOUBLE PRECISION DEFAULT 0.0;

\echo 'Columna avg_leverage añadida exitosamente.'

-- 2. Actualizar los registros existentes con el promedio de leverage
-- calculado desde la tabla strategy_samples
\echo 'Actualizando registros existentes con avg_leverage...'

UPDATE trading.strategy_memory 
SET avg_leverage = subquery.avg_lev
FROM (
    SELECT 
        sm.id,
        COALESCE(AVG(ss.leverage), 0.0) as avg_lev
    FROM trading.strategy_memory sm
    LEFT JOIN trading.strategy_samples ss ON sm.id = ss.memory_id
    WHERE sm.mode = 'backtest'  -- Solo para modo backtest por ahora
    GROUP BY sm.id
) subquery
WHERE trading.strategy_memory.id = subquery.id;

\echo 'Registros existentes actualizados.'

-- 3. Verificar los cambios
\echo 'Verificando cambios...'

SELECT 
    id,
    symbol,
    timeframe,
    signature,
    n_trades,
    win_rate,
    avg_pnl,
    avg_leverage,
    last_updated
FROM trading.strategy_memory 
ORDER BY last_updated DESC
LIMIT 5;

\echo 'Verificación completada.'

-- 4. Mostrar estadísticas de avg_leverage
\echo 'Estadísticas de avg_leverage:'

SELECT 
    COUNT(*) as total_strategies,
    ROUND(AVG(avg_leverage), 2) as avg_leverage_overall,
    ROUND(MIN(avg_leverage), 2) as min_leverage,
    ROUND(MAX(avg_leverage), 2) as max_leverage,
    COUNT(CASE WHEN avg_leverage > 0 THEN 1 END) as strategies_with_leverage
FROM trading.strategy_memory;

\echo 'Script completado exitosamente.'
