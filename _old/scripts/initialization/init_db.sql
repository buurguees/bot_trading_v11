-- Script SQL para bot_trading_v11 (ejecútalo ya conectado a trading_db)

-- 1) Esquema
CREATE SCHEMA IF NOT EXISTS trading;

-- 2) Tabla principal (particionada)
CREATE TABLE IF NOT EXISTS trading.HistoricalData (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    year_column INTEGER GENERATED ALWAYS AS (EXTRACT(YEAR FROM timestamp)) STORED,  -- Clave para partición
    open  NUMERIC(15,8) NOT NULL,
    high  NUMERIC(15,8) NOT NULL,
    low   NUMERIC(15,8) NOT NULL,
    close NUMERIC(15,8) NOT NULL,
    volume NUMERIC(20,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_timeframe CHECK (timeframe IN ('1m','5m','15m','1h','4h','1d')),
    CONSTRAINT unique_ohlcv UNIQUE (symbol, timeframe, timestamp)
)
PARTITION BY RANGE (year_column);

-- 2.1) Particiones (crear si faltan)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname='historicaldata_2025' AND n.nspname='trading'
    ) THEN
        EXECUTE $ct$ CREATE TABLE trading.HistoricalData_2025
                      PARTITION OF trading.HistoricalData
                      FOR VALUES FROM (2024) TO (2025) $ct$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relname='historicaldata_2026' AND n.nspname='trading'
    ) THEN
        EXECUTE $ct$ CREATE TABLE trading.HistoricalData_2026
                      PARTITION OF trading.HistoricalData
                      FOR VALUES FROM (2026) TO (2027) $ct$;
    END IF;
END$$;

-- 2.2) Índices
CREATE INDEX IF NOT EXISTS idx_historical_symbol_timeframe ON trading.HistoricalData (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_historical_timestamp ON trading.HistoricalData (timestamp);
CREATE INDEX IF NOT EXISTS idx_historical_covering
    ON trading.HistoricalData (symbol, timeframe, timestamp, close) INCLUDE (open, high, low, volume);

-- 3) Trades
CREATE TABLE IF NOT EXISTS trading.Trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side   VARCHAR(5)  NOT NULL,
    quantity NUMERIC(15,8) NOT NULL,
    price    NUMERIC(15,8) NOT NULL,
    pnl      NUMERIC(15,8) NOT NULL,
    entry_timestamp TIMESTAMPTZ NOT NULL,
    exit_timestamp  TIMESTAMPTZ,
    duration INTERVAL,
    leverage NUMERIC(5,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_side CHECK (side IN ('long','short')),
    CONSTRAINT chk_leverage CHECK (leverage >= 1 AND leverage <= 125)
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trading.Trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp ON trading.Trades (entry_timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_covering
  ON trading.Trades (symbol, entry_timestamp, side, pnl) INCLUDE (quantity, price);

-- 4) Estrategias ML
CREATE TABLE IF NOT EXISTS trading.MLStrategies (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    action VARCHAR(10) NOT NULL,
    timeframes JSONB NOT NULL,
    indicators JSONB NOT NULL,
    tools JSONB NOT NULL,
    leverage NUMERIC(5,2) NOT NULL,
    pnl NUMERIC(15,8) NOT NULL,
    performance NUMERIC(5,2) NOT NULL,
    confidence_score NUMERIC(5,2) NOT NULL,
    feature_importance JSONB NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_action CHECK (action IN ('long','short','hold')),
    CONSTRAINT chk_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

CREATE INDEX IF NOT EXISTS idx_strategies_symbol ON trading.MLStrategies (symbol);
CREATE INDEX IF NOT EXISTS idx_strategies_timestamp ON trading.MLStrategies (timestamp);
CREATE INDEX IF NOT EXISTS idx_strategies_covering
  ON trading.MLStrategies (symbol, timestamp, action, pnl) INCLUDE (performance, confidence_score);

-- 5) Trigger de updated_at
CREATE OR REPLACE FUNCTION trading.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname='trigger_update_historical_timestamp'
    ) THEN
        CREATE TRIGGER trigger_update_historical_timestamp
        BEFORE UPDATE ON trading.HistoricalData
        FOR EACH ROW EXECUTE FUNCTION trading.update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname='trigger_update_trades_timestamp'
    ) THEN
        CREATE TRIGGER trigger_update_trades_timestamp
        BEFORE UPDATE ON trading.Trades
        FOR EACH ROW EXECUTE FUNCTION trading.update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname='trigger_update_strategies_timestamp'
    ) THEN
        CREATE TRIGGER trigger_update_strategies_timestamp
        BEFORE UPDATE ON trading.MLStrategies
        FOR EACH ROW EXECUTE FUNCTION trading.update_timestamp();
    END IF;
END$$;

-- 6) Usuario y permisos (ajusta la contraseña si cambias el .env)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_user') THEN
        CREATE USER trading_user WITH PASSWORD '160501';
    END IF;
END$$;

GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;
GRANT USAGE ON SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA trading GRANT ALL ON TABLES TO trading_user;

-- 7) Auditoría (opcional)
CREATE TABLE IF NOT EXISTS trading.AuditLog (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL,
    record_id BIGINT NOT NULL,
    changed_by VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_auditlog_timestamp ON trading.AuditLog (timestamp);
