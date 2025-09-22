-- Esquema para resultados de backtests
CREATE SCHEMA IF NOT EXISTS trading;

-- Tabla resumen de backtests
CREATE TABLE IF NOT EXISTS trading."Backtests" (
  id           BIGSERIAL PRIMARY KEY,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol       TEXT NOT NULL,
  timeframe    TEXT NOT NULL,
  from_ts      TIMESTAMPTZ NOT NULL,
  to_ts        TIMESTAMPTZ NOT NULL,
  n_trades     INTEGER NOT NULL,
  gross_pnl    DOUBLE PRECISION NOT NULL,
  fees         DOUBLE PRECISION NOT NULL,
  net_pnl      DOUBLE PRECISION NOT NULL,
  win_rate     DOUBLE PRECISION NOT NULL,
  max_dd       DOUBLE PRECISION NOT NULL,
  comment      TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtests_sym_tf_ts
  ON trading."Backtests" (symbol, timeframe, created_at DESC);

-- Tabla de trades simulados
CREATE TABLE IF NOT EXISTS trading."BacktestTrades" (
  id           BIGSERIAL PRIMARY KEY,
  backtest_id  BIGINT NOT NULL REFERENCES trading."Backtests"(id) ON DELETE CASCADE,
  plan_id      BIGINT,
  symbol       TEXT NOT NULL,
  timeframe    TEXT NOT NULL,
  entry_ts     TIMESTAMPTZ NOT NULL,
  exit_ts      TIMESTAMPTZ NOT NULL,
  side         INTEGER NOT NULL,
  entry_px     DOUBLE PRECISION NOT NULL,
  exit_px      DOUBLE PRECISION NOT NULL,
  qty          DOUBLE PRECISION NOT NULL,
  leverage     DOUBLE PRECISION NOT NULL,
  fee          DOUBLE PRECISION NOT NULL,
  pnl          DOUBLE PRECISION NOT NULL,
  funding      DOUBLE PRECISION DEFAULT 0,
  exit         TEXT,
  reason       JSONB
);

CREATE INDEX IF NOT EXISTS idx_bt_trades_bt
  ON trading."BacktestTrades" (backtest_id);

CREATE INDEX IF NOT EXISTS idx_bt_trades_sym_tf_ts
  ON trading."BacktestTrades" (symbol, timeframe, entry_ts, exit_ts);


