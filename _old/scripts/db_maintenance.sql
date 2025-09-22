-- ============================================================================
-- SCRIPT DE MANTENIMIENTO DE BASE DE DATOS
-- Optimización con índices BRIN y VACUUM/ANALYZE
-- ============================================================================

-- Configuración de logging
\echo 'Iniciando mantenimiento de base de datos...'
\echo 'Timestamp: ' || now()

-- ============================================================================
-- 1. ÍNDICES BRIN (Block Range Indexes)
-- Para tablas grandes con datos ordenados por timestamp
-- ============================================================================

\echo 'Creando índices BRIN...'

-- BRIN para historicaldata (tabla más grande)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_brin_timestamp 
ON trading.historicaldata USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_brin_symbol_timestamp 
ON trading.historicaldata USING BRIN (symbol, timestamp);

-- BRIN para features (tabla grande con datos temporales)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_brin_timestamp 
ON trading.features USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_brin_symbol_timestamp 
ON trading.features USING BRIN (symbol, timestamp);

-- BRIN para agentpreds (tabla grande con predicciones)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_brin_timestamp 
ON trading.agentpreds USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_brin_symbol_timestamp 
ON trading.agentpreds USING BRIN (symbol, timestamp);

-- BRIN para agentsignals (tabla grande con señales)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_brin_timestamp 
ON trading.agentsignals USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_brin_symbol_timestamp 
ON trading.agentsignals USING BRIN (symbol, timestamp);

-- BRIN para tradeplans (tabla grande con planes)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tradeplans_brin_bar_ts 
ON trading.tradeplans USING BRIN (bar_ts);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tradeplans_brin_symbol_bar_ts 
ON trading.tradeplans USING BRIN (symbol, bar_ts);

-- BRIN para backtests (tabla grande con resultados)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtests_brin_created_at 
ON trading.backtests USING BRIN (created_at);

-- BRIN para backtesttrades (tabla grande con trades)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_brin_entry_ts 
ON trading.backtesttrades USING BRIN (entry_ts);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_brin_symbol_entry_ts 
ON trading.backtesttrades USING BRIN (symbol, entry_ts);

\echo 'Índices BRIN creados exitosamente.'

-- ============================================================================
-- 2. ÍNDICES ADICIONALES PARA OPTIMIZACIÓN
-- ============================================================================

\echo 'Creando índices adicionales...'

-- Índices compuestos para consultas frecuentes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_symbol_tf_timestamp 
ON trading.historicaldata (symbol, timeframe, timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_symbol_tf_timestamp 
ON trading.features (symbol, timeframe, timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_symbol_tf_timestamp 
ON trading.agentpreds (symbol, timeframe, timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_symbol_tf_timestamp 
ON trading.agentsignals (symbol, timeframe, timestamp);

-- Índices para consultas de rendimiento
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtests_symbol_tf_created_at 
ON trading.backtests (symbol, timeframe, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_symbol_tf_entry_ts 
ON trading.backtesttrades (symbol, timeframe, entry_ts);

\echo 'Índices adicionales creados exitosamente.'

-- ============================================================================
-- 3. VACUUM Y ANALYZE
-- Para tablas grandes y actualizaciones frecuentes
-- ============================================================================

\echo 'Ejecutando VACUUM ANALYZE...'

-- VACUUM ANALYZE para tablas grandes (en orden de tamaño)
VACUUM (ANALYZE, VERBOSE) trading.historicaldata;
VACUUM (ANALYZE, VERBOSE) trading.features;
VACUUM (ANALYZE, VERBOSE) trading.agentpreds;
VACUUM (ANALYZE, VERBOSE) trading.agentsignals;
VACUUM (ANALYZE, VERBOSE) trading.tradeplans;
VACUUM (ANALYZE, VERBOSE) trading.backtesttrades;
VACUUM (ANALYZE, VERBOSE) trading.backtests;

-- VACUUM ANALYZE para tablas de metadatos
VACUUM (ANALYZE, VERBOSE) trading.agentversions;
VACUUM (ANALYZE, VERBOSE) trading.strategy_memory;
VACUUM (ANALYZE, VERBOSE) trading.strategy_samples;

\echo 'VACUUM ANALYZE completado.'

-- ============================================================================
-- 4. ESTADÍSTICAS DE MANTENIMIENTO
-- ============================================================================

\echo 'Generando estadísticas de mantenimiento...'

-- Mostrar estadísticas de tablas
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs
FROM pg_stats 
WHERE schemaname = 'trading' 
    AND tablename IN ('historicaldata', 'features', 'agentpreds', 'agentsignals', 'tradeplans', 'backtests', 'backtesttrades')
ORDER BY tablename, attname;

-- Mostrar tamaño de tablas
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'trading'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Mostrar estadísticas de índices
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) as size
FROM pg_indexes 
WHERE schemaname = 'trading'
ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC;

\echo 'Estadísticas generadas.'

-- ============================================================================
-- 5. LIMPIEZA DE DATOS ANTIGUOS (OPCIONAL)
-- ============================================================================

\echo 'Verificando datos antiguos...'

-- Mostrar rangos de fechas en tablas principales
SELECT 'historicaldata' as tabla, 
       MIN(timestamp) as min_ts, 
       MAX(timestamp) as max_ts, 
       COUNT(*) as total_registros
FROM trading.historicaldata
UNION ALL
SELECT 'features' as tabla, 
       MIN(timestamp) as min_ts, 
       MAX(timestamp) as max_ts, 
       COUNT(*) as total_registros
FROM trading.features
UNION ALL
SELECT 'agentpreds' as tabla, 
       MIN(timestamp) as min_ts, 
       MAX(timestamp) as max_ts, 
       COUNT(*) as total_registros
FROM trading.agentpreds
UNION ALL
SELECT 'agentsignals' as tabla, 
       MIN(timestamp) as min_ts, 
       MAX(timestamp) as max_ts, 
       COUNT(*) as total_registros
FROM trading.agentsignals
UNION ALL
SELECT 'tradeplans' as tabla, 
       MIN(bar_ts) as min_ts, 
       MAX(bar_ts) as max_ts, 
       COUNT(*) as total_registros
FROM trading.tradeplans
ORDER BY tabla;

\echo 'Mantenimiento de base de datos completado.'
\echo 'Timestamp: ' || now()
