-- =====================================================
-- UPGRADE MLSTRATEGIES + LIMPIEZA DE TABLAS MAYÚSCULAS
-- =====================================================
-- Script para:
-- 1. Añadir columna signature a mlstrategies
-- 2. Eliminar tablas con mayúsculas (MLStrategies, Trades, AuditLog, HistoricalData)
-- 3. Configurar mlstrategies como "diario de estrategias aprendidas"
-- =====================================================

\echo 'Iniciando upgrade de mlstrategies y limpieza...'
\echo 'Timestamp: ' || NOW()

-- =====================================================
-- 1. AÑADIR COLUMNA SIGNATURE A MLSTRATEGIES
-- =====================================================
\echo 'Añadiendo columna signature a mlstrategies...'

-- Añadir columna signature
ALTER TABLE trading.mlstrategies 
ADD COLUMN IF NOT EXISTS signature VARCHAR(255);

-- Añadir índice en signature para consultas eficientes
CREATE INDEX IF NOT EXISTS idx_mlstrategies_signature 
ON trading.mlstrategies (signature);

-- Añadir comentario
COMMENT ON COLUMN trading.mlstrategies.signature IS 'Firma única de la estrategia (hash de features)';

\echo 'Columna signature añadida exitosamente.'

-- =====================================================
-- 2. VERIFICAR DATOS EN TABLAS CON MAYÚSCULAS
-- =====================================================
\echo 'Verificando datos en tablas con mayúsculas...'

-- Verificar si hay datos en las tablas que vamos a eliminar
DO $$
DECLARE
    ml_count INTEGER;
    trades_count INTEGER;
    audit_count INTEGER;
    hist_count INTEGER;
BEGIN
    -- Contar registros en cada tabla
    SELECT COUNT(*) INTO ml_count FROM trading."MLStrategies";
    SELECT COUNT(*) INTO trades_count FROM trading."Trades";
    SELECT COUNT(*) INTO audit_count FROM trading."AuditLog";
    SELECT COUNT(*) INTO hist_count FROM trading."HistoricalData";
    
    \echo 'Registros encontrados:';
    \echo '  MLStrategies: ' || ml_count;
    \echo '  Trades: ' || trades_count;
    \echo '  AuditLog: ' || audit_count;
    \echo '  HistoricalData: ' || hist_count;
    
    -- Si hay datos, mostrar advertencia
    IF ml_count > 0 OR trades_count > 0 OR audit_count > 0 OR hist_count > 0 THEN
        \echo '⚠️  ADVERTENCIA: Hay datos en las tablas que se van a eliminar!';
        \echo '   Considera hacer backup antes de continuar.';
    ELSE
        \echo '✅ No hay datos en las tablas a eliminar.';
    END IF;
END $$;

-- =====================================================
-- 3. ELIMINAR TABLAS CON MAYÚSCULAS
-- =====================================================
\echo 'Eliminando tablas con mayúsculas...'

-- Eliminar foreign keys primero (si existen)
DO $$
BEGIN
    -- Eliminar FK de MLStrategies si existe
    IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
               WHERE table_schema = 'trading' AND table_name = 'MLStrategies' 
               AND constraint_type = 'FOREIGN KEY') THEN
        ALTER TABLE trading."MLStrategies" DROP CONSTRAINT IF EXISTS fk_mlstrategies_symbol;
        \echo 'FK de MLStrategies eliminada';
    END IF;
    
    -- Eliminar FK de Trades si existe
    IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
               WHERE table_schema = 'trading' AND table_name = 'Trades' 
               AND constraint_type = 'FOREIGN KEY') THEN
        ALTER TABLE trading."Trades" DROP CONSTRAINT IF EXISTS fk_trades_symbol;
        \echo 'FK de Trades eliminada';
    END IF;
END $$;

-- Eliminar tablas
DROP TABLE IF EXISTS trading."MLStrategies" CASCADE;
\echo 'Tabla MLStrategies eliminada';

DROP TABLE IF EXISTS trading."Trades" CASCADE;
\echo 'Tabla Trades eliminada';

DROP TABLE IF EXISTS trading."AuditLog" CASCADE;
\echo 'Tabla AuditLog eliminada';

DROP TABLE IF EXISTS trading."HistoricalData" CASCADE;
\echo 'Tabla HistoricalData eliminada';

-- =====================================================
-- 4. CREAR FUNCIÓN PARA POBLAR MLSTRATEGIES
-- =====================================================
\echo 'Creando función para poblar mlstrategies...'

-- Función para poblar mlstrategies desde strategy_memory + strategy_samples
CREATE OR REPLACE FUNCTION trading.populate_mlstrategies_from_memory()
RETURNS INTEGER AS $$
DECLARE
    strategy_record RECORD;
    sample_record RECORD;
    inserted_count INTEGER := 0;
    strategy_signature VARCHAR(255);
BEGIN
    \echo 'Poblando mlstrategies desde strategy_memory...';
    
    -- Iterar sobre cada estrategia en strategy_memory
    FOR strategy_record IN 
        SELECT 
            sm.id,
            sm.symbol,
            sm.timeframe,
            sm.signature,
            sm.features,
            sm.n_trades,
            sm.win_rate,
            sm.avg_pnl,
            sm.avg_leverage,
            sm.sharpe,
            sm.avg_hold_bars,
            sm.last_updated,
            sm.mode
        FROM trading.strategy_memory sm
        WHERE sm.n_trades > 0
        ORDER BY sm.last_updated DESC
    LOOP
        -- Generar signature si no existe
        IF strategy_record.signature IS NULL OR strategy_record.signature = '' THEN
            strategy_signature := MD5(strategy_record.features::text);
        ELSE
            strategy_signature := strategy_record.signature;
        END IF;
        
        -- Obtener el primer sample de la estrategia para datos básicos
        SELECT 
            ss.entry_ts,
            ss.side,
            ss.leverage,
            ss.pnl,
            ss.reason
        INTO sample_record
        FROM trading.strategy_samples ss
        WHERE ss.memory_id = strategy_record.id
        ORDER BY ss.entry_ts DESC
        LIMIT 1;
        
        -- Insertar en mlstrategies si no existe ya
        INSERT INTO trading.mlstrategies (
            symbol, timestamp, action, timeframes, indicators, tools,
            leverage, pnl, performance, confidence_score, feature_importance,
            outcome, signature
        )
        SELECT 
            strategy_record.symbol,
            COALESCE(sample_record.entry_ts, strategy_record.last_updated),
            CASE 
                WHEN sample_record.side = 1 THEN 'long'
                WHEN sample_record.side = -1 THEN 'short'
                ELSE 'hold'
            END,
            jsonb_build_object('timeframe', strategy_record.timeframe),
            strategy_record.features,
            jsonb_build_object('signature', strategy_signature),
            COALESCE(sample_record.leverage, strategy_record.avg_leverage),
            COALESCE(sample_record.pnl, strategy_record.avg_pnl),
            strategy_record.avg_pnl,
            strategy_record.win_rate,
            strategy_record.features,
            CASE 
                WHEN strategy_record.avg_pnl > 0 THEN 'profitable'
                WHEN strategy_record.avg_pnl < 0 THEN 'loss'
                ELSE 'neutral'
            END,
            strategy_signature
        WHERE NOT EXISTS (
            SELECT 1 FROM trading.mlstrategies ms 
            WHERE ms.signature = strategy_signature 
            AND ms.symbol = strategy_record.symbol
        );
        
        -- Contar inserción
        IF FOUND THEN
            inserted_count := inserted_count + 1;
        END IF;
    END LOOP;
    
    \echo 'Insertadas ' || inserted_count || ' estrategias en mlstrategies';
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

\echo 'Función populate_mlstrategies_from_memory creada exitosamente.';

-- =====================================================
-- 5. EJECUTAR POBLADO INICIAL
-- =====================================================
\echo 'Ejecutando poblamiento inicial...'

SELECT trading.populate_mlstrategies_from_memory();

-- =====================================================
-- 6. CREAR TRIGGER PARA AUTO-POBLADO
-- =====================================================
\echo 'Creando trigger para auto-poblamiento...'

-- Función trigger para auto-poblar mlstrategies
CREATE OR REPLACE FUNCTION trading.auto_populate_mlstrategies()
RETURNS TRIGGER AS $$
BEGIN
    -- Solo poblar si hay cambios en strategy_memory
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        PERFORM trading.populate_mlstrategies_from_memory();
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Crear trigger en strategy_memory
DROP TRIGGER IF EXISTS trigger_auto_populate_mlstrategies ON trading.strategy_memory;
CREATE TRIGGER trigger_auto_populate_mlstrategies
    AFTER INSERT OR UPDATE ON trading.strategy_memory
    FOR EACH STATEMENT
    EXECUTE FUNCTION trading.auto_populate_mlstrategies();

\echo 'Trigger para auto-poblamiento creado exitosamente.';

-- =====================================================
-- 7. VERIFICACIÓN FINAL
-- =====================================================
\echo 'Verificando estructura final...'

-- Mostrar estructura de mlstrategies
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'trading' 
  AND table_name = 'mlstrategies'
ORDER BY ordinal_position;

-- Mostrar conteo de registros
SELECT 
    'mlstrategies' as tabla,
    COUNT(*) as registros
FROM trading.mlstrategies
UNION ALL
SELECT 
    'strategy_memory' as tabla,
    COUNT(*) as registros
FROM trading.strategy_memory
UNION ALL
SELECT 
    'strategy_samples' as tabla,
    COUNT(*) as registros
FROM trading.strategy_samples;

\echo 'Upgrade y limpieza completados exitosamente.'
\echo 'Timestamp final: ' || NOW()

-- =====================================================
-- 8. CONSULTAS DE VERIFICACIÓN
-- =====================================================
\echo ''
\echo 'CONSULTAS DE VERIFICACIÓN:'
\echo '========================='
\echo '-- Ver estrategias en mlstrategies:'
\echo 'SELECT symbol, signature, action, performance, confidence_score FROM trading.mlstrategies ORDER BY timestamp DESC LIMIT 10;'
\echo ''
\echo '-- Ver estrategias por signature:'
\echo 'SELECT signature, COUNT(*) as count, AVG(performance) as avg_performance FROM trading.mlstrategies GROUP BY signature ORDER BY count DESC;'
\echo ''
\echo '-- Ver relación strategy_memory -> mlstrategies:'
\echo 'SELECT sm.signature, sm.n_trades, COUNT(ms.id) as mlstrategies_count FROM trading.strategy_memory sm LEFT JOIN trading.mlstrategies ms ON sm.signature = ms.signature GROUP BY sm.signature, sm.n_trades ORDER BY sm.n_trades DESC;'
\echo '========================='
