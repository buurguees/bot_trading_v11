-- =====================================================
-- UPGRADE DE TABLA TRADES - BOT TRADING V11
-- =====================================================
-- Script para añadir nuevas columnas a la tabla trades
-- Incluye: plan_id, order_ids, fees_paid, slip_bps, entry_balance, exit_balance
-- =====================================================

\echo 'Iniciando upgrade de tabla trades...'
\echo 'Timestamp: ' || NOW()

-- =====================================================
-- 1. CREAR ENUM PARA SIDE
-- =====================================================
\echo 'Creando ENUM para side...'

-- Crear tipo ENUM para side si no existe
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'trade_side') THEN
        CREATE TYPE trading.trade_side AS ENUM ('long', 'short');
        \echo 'ENUM trade_side creado exitosamente'
    ELSE
        \echo 'ENUM trade_side ya existe'
    END IF;
END $$;

-- =====================================================
-- 2. AÑADIR NUEVAS COLUMNAS
-- =====================================================
\echo 'Añadiendo nuevas columnas a trades...'

-- plan_id: referencia al tradeplan usado
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS plan_id BIGINT;

-- order_ids: JSONB para vincular con exchange
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS order_ids JSONB DEFAULT '{}';

-- fees_paid: comisiones pagadas
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS fees_paid NUMERIC(15,8) DEFAULT 0.0;

-- slip_bps: slippage en basis points
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS slip_bps NUMERIC(8,2) DEFAULT 0.0;

-- entry_balance: balance al entrar
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS entry_balance NUMERIC(15,8);

-- exit_balance: balance al salir
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS exit_balance NUMERIC(15,8);

-- symbol_id: FK a tabla symbols
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS symbol_id BIGINT;

\echo 'Nuevas columnas añadidas exitosamente.'

-- =====================================================
-- 3. CREAR ÍNDICES
-- =====================================================
\echo 'Creando índices...'

-- Índice en plan_id
CREATE INDEX IF NOT EXISTS idx_trades_plan_id 
ON trading.trades (plan_id);

-- Índice en symbol_id
CREATE INDEX IF NOT EXISTS idx_trades_symbol_id 
ON trading.trades (symbol_id);

-- Índice en entry_balance para análisis
CREATE INDEX IF NOT EXISTS idx_trades_entry_balance 
ON trading.trades (entry_balance);

-- Índice compuesto para análisis de rendimiento
CREATE INDEX IF NOT EXISTS idx_trades_symbol_balance 
ON trading.trades (symbol_id, entry_balance, exit_balance);

\echo 'Índices creados exitosamente.'

-- =====================================================
-- 4. AÑADIR FOREIGN KEYS
-- =====================================================
\echo 'Añadiendo foreign keys...'

-- FK a tradeplans (si existe la tabla)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_schema = 'trading' AND table_name = 'tradeplans') THEN
        ALTER TABLE trading.trades 
        ADD CONSTRAINT IF NOT EXISTS fk_trades_plan_id 
        FOREIGN KEY (plan_id) REFERENCES trading.tradeplans(id);
        \echo 'FK a tradeplans añadida'
    ELSE
        \echo 'Tabla tradeplans no existe, saltando FK'
    END IF;
END $$;

-- FK a symbols
ALTER TABLE trading.trades 
ADD CONSTRAINT IF NOT EXISTS fk_trades_symbol_id 
FOREIGN KEY (symbol_id) REFERENCES trading.symbols(id);

\echo 'Foreign keys añadidas exitosamente.'

-- =====================================================
-- 5. ESTANDARIZAR COLUMNA SIDE
-- =====================================================
\echo 'Estandarizando columna side...'

-- Crear columna temporal con ENUM
ALTER TABLE trading.trades 
ADD COLUMN IF NOT EXISTS side_enum trading.trade_side;

-- Migrar datos existentes (si los hay)
UPDATE trading.trades 
SET side_enum = CASE 
    WHEN LOWER(side) IN ('long', 'buy', '1') THEN 'long'::trading.trade_side
    WHEN LOWER(side) IN ('short', 'sell', '-1') THEN 'short'::trading.trade_side
    ELSE 'long'::trading.trade_side  -- default
END
WHERE side_enum IS NULL;

-- Hacer la columna NOT NULL
ALTER TABLE trading.trades 
ALTER COLUMN side_enum SET NOT NULL;

-- Eliminar columna antigua y renombrar
ALTER TABLE trading.trades DROP COLUMN IF EXISTS side;
ALTER TABLE trading.trades RENAME COLUMN side_enum TO side;

\echo 'Columna side estandarizada exitosamente.'

-- =====================================================
-- 6. AÑADIR CONSTRAINTS
-- =====================================================
\echo 'Añadiendo constraints...'

-- Constraint para fees_paid >= 0
ALTER TABLE trading.trades 
ADD CONSTRAINT IF NOT EXISTS chk_trades_fees_paid 
CHECK (fees_paid >= 0);

-- Constraint para slip_bps >= 0
ALTER TABLE trading trading.trades 
ADD CONSTRAINT IF NOT EXISTS chk_trades_slip_bps 
CHECK (slip_bps >= 0);

-- Constraint para balances positivos
ALTER TABLE trading.trades 
ADD CONSTRAINT IF NOT EXISTS chk_trades_entry_balance 
CHECK (entry_balance IS NULL OR entry_balance > 0);

ALTER TABLE trading.trades 
ADD CONSTRAINT IF NOT EXISTS chk_trades_exit_balance 
CHECK (exit_balance IS NULL OR exit_balance > 0);

\echo 'Constraints añadidas exitosamente.'

-- =====================================================
-- 7. COMENTARIOS EN COLUMNAS
-- =====================================================
\echo 'Añadiendo comentarios...'

COMMENT ON COLUMN trading.trades.plan_id IS 'ID del tradeplan usado para esta operación';
COMMENT ON COLUMN trading.trades.order_ids IS 'IDs de órdenes en el exchange (JSONB)';
COMMENT ON COLUMN trading.trades.fees_paid IS 'Comisiones pagadas en la operación';
COMMENT ON COLUMN trading.trades.slip_bps IS 'Slippage en basis points (0.01% = 1 bps)';
COMMENT ON COLUMN trading.trades.entry_balance IS 'Balance disponible al entrar en la operación';
COMMENT ON COLUMN trading.trades.exit_balance IS 'Balance disponible al salir de la operación';
COMMENT ON COLUMN trading.trades.symbol_id IS 'Foreign key a tabla symbols';
COMMENT ON COLUMN trading.trades.side IS 'Dirección de la operación (long/short)';

\echo 'Comentarios añadidos exitosamente.'

-- =====================================================
-- 8. VERIFICACIÓN FINAL
-- =====================================================
\echo 'Verificando estructura final...'

SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'trading' 
  AND table_name = 'trades'
ORDER BY ordinal_position;

\echo 'Upgrade de tabla trades completado exitosamente.'
\echo 'Timestamp final: ' || NOW()

-- =====================================================
-- 9. CONSULTAS DE VERIFICACIÓN
-- =====================================================
\echo ''
\echo 'CONSULTAS DE VERIFICACIÓN:'
\echo '========================='
\echo '-- Ver estructura completa:'
\echo 'SELECT * FROM information_schema.columns WHERE table_schema = \'trading\' AND table_name = \'trades\';'
\echo ''
\echo '-- Ver constraints:'
\echo 'SELECT conname, contype FROM pg_constraint WHERE conrelid = \'trading.trades\'::regclass;'
\echo ''
\echo '-- Ver foreign keys:'
\echo 'SELECT tc.constraint_name, tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name FROM information_schema.table_constraints AS tc JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name WHERE tc.constraint_type = \'FOREIGN KEY\' AND tc.table_name = \'trades\';'
\echo '========================='
