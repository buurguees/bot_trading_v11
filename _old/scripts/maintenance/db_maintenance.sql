-- =====================================================
-- MANTENIMIENTO DE BASE DE DATOS - BOT TRADING V11
-- =====================================================
-- Script para crear índices BRIN y ejecutar VACUUM/ANALYZE
-- Optimizado para tablas con datos temporales secuenciales
-- =====================================================

-- Configuración de logging
\echo 'Iniciando mantenimiento de base de datos...'
\echo 'Timestamp: ' || NOW()

-- =====================================================
-- 1. ÍNDICES BRIN (Block Range INdexes)
-- =====================================================
-- Los índices BRIN son ideales para columnas con datos ordenados
-- como timestamps, especialmente en tablas grandes

\echo 'Creando índices BRIN...'

-- HistoricalData - Tabla principal con datos OHLCV
-- BRIN en timestamp (datos ordenados cronológicamente)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_historical_timestamp 
ON trading.historicaldata USING BRIN (timestamp);

-- BRIN en created_at para consultas de auditoría
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_historical_created_at 
ON trading.historicaldata USING BRIN (created_at);

-- Trades - Operaciones del bot
-- BRIN en entry_timestamp (datos ordenados por fecha de entrada)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_trades_entry_timestamp 
ON trading.trades USING BRIN (entry_timestamp);

-- BRIN en created_at para auditoría
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_trades_created_at 
ON trading.trades USING BRIN (created_at);

-- MLStrategies - Estrategias evaluadas
-- BRIN en timestamp (datos ordenados cronológicamente)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_strategies_timestamp 
ON trading.mlstrategies USING BRIN (timestamp);

-- BRIN en created_at para auditoría
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_strategies_created_at 
ON trading.mlstrategies USING BRIN (created_at);

-- AgentVersions - Versiones de modelos ML
-- BRIN en created_at (datos ordenados por fecha de creación)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_agentversions_created_at 
ON trading.agentversions USING BRIN (created_at);

-- AgentPreds - Predicciones de modelos (si existe)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_agentpreds_timestamp 
ON trading.agentpreds USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_agentpreds_created_at 
ON trading.agentpreds USING BRIN (created_at);

-- AgentSignals - Señales generadas (si existe)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_agentsignals_timestamp 
ON trading.agentsignals USING BRIN (timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_agentsignals_created_at 
ON trading.agentsignals USING BRIN (created_at);

-- TradePlans - Planes de trading (si existe)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_tradeplans_bar_ts 
ON trading.tradeplans USING BRIN (bar_ts);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_tradeplans_created_at 
ON trading.tradeplans USING BRIN (created_at);

-- Backtests - Resultados de backtesting (si existe)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_backtests_run_ts 
ON trading.backtests USING BRIN (run_ts);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_backtests_created_at 
ON trading.backtests USING BRIN (created_at);

-- AuditLog - Log de auditoría
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_auditlog_timestamp 
ON trading.auditlog USING BRIN (timestamp);

\echo 'Índices BRIN creados exitosamente.'

-- =====================================================
-- 2. VACUUM Y ANALYZE
-- =====================================================
-- Optimización de estadísticas y limpieza de espacio

\echo 'Ejecutando VACUUM ANALYZE en tablas principales...'

-- HistoricalData - Tabla más grande, necesita mantenimiento frecuente
VACUUM (ANALYZE, VERBOSE) trading.historicaldata;

-- Trades - Tabla de operaciones
VACUUM (ANALYZE, VERBOSE) trading.trades;

-- MLStrategies - Estrategias evaluadas
VACUUM (ANALYZE, VERBOSE) trading.mlstrategies;

-- AgentVersions - Versiones de modelos
VACUUM (ANALYZE, VERBOSE) trading.agentversions;

-- AgentPreds - Predicciones (si existe)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_schema = 'trading' AND table_name = 'agentpreds') THEN
        EXECUTE 'VACUUM (ANALYZE, VERBOSE) trading.agentpreds';
    END IF;
END $$;

-- AgentSignals - Señales (si existe)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_schema = 'trading' AND table_name = 'agentsignals') THEN
        EXECUTE 'VACUUM (ANALYZE, VERBOSE) trading.agentsignals';
    END IF;
END $$;

-- TradePlans - Planes de trading (si existe)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_schema = 'trading' AND table_name = 'tradeplans') THEN
        EXECUTE 'VACUUM (ANALYZE, VERBOSE) trading.tradeplans';
    END IF;
END $$;

-- Backtests - Resultados de backtesting (si existe)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_schema = 'trading' AND table_name = 'backtests') THEN
        EXECUTE 'VACUUM (ANALYZE, VERBOSE) trading.backtests';
    END IF;
END $$;

-- AuditLog - Log de auditoría
VACUUM (ANALYZE, VERBOSE) trading.auditlog;

\echo 'VACUUM ANALYZE completado.'

-- =====================================================
-- 3. ESTADÍSTICAS DE MANTENIMIENTO
-- =====================================================
-- Mostrar información sobre el estado de las tablas

\echo 'Generando estadísticas de mantenimiento...'

-- Información general de tablas
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables 
WHERE schemaname = 'trading'
ORDER BY n_live_tup DESC;

-- Tamaño de tablas
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'trading'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Índices BRIN creados
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'trading' 
  AND indexdef LIKE '%USING BRIN%'
ORDER BY tablename, indexname;

\echo 'Mantenimiento de base de datos completado exitosamente.'
\echo 'Timestamp final: ' || NOW()

-- =====================================================
-- 4. CONFIGURACIÓN RECOMENDADA
-- =====================================================
-- Configuraciones de PostgreSQL para optimizar BRIN

\echo ''
\echo 'CONFIGURACIONES RECOMENDADAS PARA POSTGRESQL:'
\echo '============================================='
\echo '-- Ajustar páginas por rango BRIN (default: 128)'
\echo '-- ALTER SYSTEM SET brin_page_items_per_range = 64;'
\echo ''
\echo '-- Configurar autovacuum para tablas grandes'
\echo '-- ALTER TABLE trading.historicaldata SET (autovacuum_vacuum_scale_factor = 0.1);'
\echo '-- ALTER TABLE trading.historicaldata SET (autovacuum_analyze_scale_factor = 0.05);'
\echo ''
\echo '-- Reiniciar PostgreSQL después de cambios de configuración'
\echo '-- SELECT pg_reload_conf();'
\echo '============================================='
