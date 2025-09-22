-- FASE 2: Tablas ML (usar en pgAdmin)
CREATE SCHEMA IF NOT EXISTS ml;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ml.strategies (particionada por created_at)
DROP TABLE IF EXISTS ml.strategies CASCADE;
CREATE TABLE ml.strategies (
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  strategy_id     UUID        NOT NULL DEFAULT gen_random_uuid(),
  symbol          TEXT        NOT NULL,
  timeframe       TEXT        NOT NULL CHECK (timeframe IN ('1m','5m','15m','1h','4h','1d')),
  strategy_key    TEXT        NOT NULL,  -- hash de reglas+riesgos+versiones (reproducible)
  description     TEXT,
  source          TEXT,                  -- 'planner' | 'ppo' | 'ensemble' | 'manual'
  rules           JSONB,                 -- condiciones y parámetros
  risk_profile    JSONB,                 -- risk_pct, leverage, límites…
  metrics_summary JSONB,                 -- sharpe, pf, max_dd, winrate, trades_n, stability...
  status          TEXT NOT NULL DEFAULT 'candidate'
                  CHECK (status IN ('candidate','testing','ready_for_training','promoted','shadow','deprecated','rejected')),
  tags            TEXT[],
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT pk_ml_strategies PRIMARY KEY (created_at, strategy_id)
) PARTITION BY RANGE (created_at);

DO $$
DECLARE d date := date '2020-01-01'; stop date := date '2030-01-01'; part text;
BEGIN
  WHILE d < stop LOOP
    part := format('strategies_%s', to_char(d,'YYYY_MM'));
    EXECUTE format(
      'CREATE TABLE IF NOT EXISTS ml.%I PARTITION OF ml.strategies
         FOR VALUES FROM (%L) TO (%L);',
      part, d::timestamptz, (d + interval '1 month')::timestamptz
    );
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I_sym_tf ON ml.%I (symbol,timeframe,created_at);',
      part || '_sym_tf', part);
    d := d + interval '1 month';
  END LOOP;
END $$;

-- ml.backtest_runs (particionada por started_at)
DROP TABLE IF EXISTS ml.backtest_runs CASCADE;
CREATE TABLE ml.backtest_runs (
  started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  run_id       UUID        NOT NULL DEFAULT gen_random_uuid(),
  strategy_id  UUID        NOT NULL,
  symbol       TEXT        NOT NULL,
  timeframe    TEXT        NOT NULL CHECK (timeframe IN ('1m','5m','15m','1h','4h','1d')),
  engine       TEXT        NOT NULL CHECK (engine IN ('vectorized','event_driven')),
  dataset_start TIMESTAMPTZ,
  dataset_end   TIMESTAMPTZ,
  cv_schema     JSONB,       -- splits walk-forward
  market_costs  JSONB,       -- fees, slippage, latency, funding
  config        JSONB,       -- copia de reglas/params usados
  metrics       JSONB,       -- métricas completas (por split y globales)
  status        TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','error')),
  message       TEXT,
  CONSTRAINT pk_backtest_runs PRIMARY KEY (started_at, run_id)
) PARTITION BY RANGE (started_at);

DO $$
DECLARE d date := date '2020-01-01'; stop date := date '2030-01-01'; part text;
BEGIN
  WHILE d < stop LOOP
    part := format('bt_runs_%s', to_char(d,'YYYY_MM'));
    EXECUTE format(
      'CREATE TABLE IF NOT EXISTS ml.%I PARTITION OF ml.backtest_runs
         FOR VALUES FROM (%L) TO (%L);',
      part, d::timestamptz, (d + interval '1 month')::timestamptz
    );
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I_strat ON ml.%I (strategy_id);', part||'_strat', part);
    d := d + interval '1 month';
  END LOOP;
END $$;

-- ml.backtest_trades (particionada por entry_ts)
DROP TABLE IF EXISTS ml.backtest_trades CASCADE;
CREATE TABLE ml.backtest_trades (
  entry_ts   TIMESTAMPTZ NOT NULL,
  run_id     UUID        NOT NULL,
  trade_id   UUID        NOT NULL DEFAULT gen_random_uuid(),
  symbol     TEXT        NOT NULL,
  side       TEXT        NOT NULL CHECK (side IN ('long','short')),
  entry_price NUMERIC(18,8) NOT NULL,
  exit_ts    TIMESTAMPTZ,
  exit_price NUMERIC(18,8),
  qty        NUMERIC(28,10),
  pnl_usdt   NUMERIC(18,6),
  mae        NUMERIC(18,6),
  mfe        NUMERIC(18,6),
  fees_usdt  NUMERIC(18,6),
  bars_held  INTEGER,
  notes      JSONB,
  CONSTRAINT pk_bt_trades PRIMARY KEY (entry_ts, run_id, trade_id)
) PARTITION BY RANGE (entry_ts);

DO $$
DECLARE d date := date '2020-01-01'; stop date := date '2030-01-01'; part text;
BEGIN
  WHILE d < stop LOOP
    part := format('bt_trades_%s', to_char(d,'YYYY_MM'));
    EXECUTE format(
      'CREATE TABLE IF NOT EXISTS ml.%I PARTITION OF ml.backtest_trades
         FOR VALUES FROM (%L) TO (%L);',
      part, d::timestamptz, (d + interval '1 month')::timestamptz
    );
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I_run ON ml.%I (run_id);', part||'_run', part);
    d := d + interval '1 month';
  END LOOP;
END $$;

-- ml.agents (NO particionada para permitir unique parcial global)
DROP TABLE IF EXISTS ml.agents CASCADE;
CREATE TABLE ml.agents (
  agent_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol       TEXT        NOT NULL,
  task         TEXT        NOT NULL CHECK (task IN ('direction','regime','smc','execution')),
  version      TEXT,
  components   JSONB,      -- {encoder, heads, ppo_policy, planner_hash...}
  artifact_uri TEXT,       -- ruta a agents/{SYMBOL}_PPO.zip
  train_run_ref TEXT,
  metrics      JSONB,      -- métricas clave para promoción
  status       TEXT NOT NULL DEFAULT 'candidate'
               CHECK (status IN ('candidate','promoted','shadow','archived')),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  promoted_at  TIMESTAMPTZ
);

-- ÚNICO campeón por (symbol, task)
CREATE UNIQUE INDEX IF NOT EXISTS ux_ml_agents_promoted
  ON ml.agents(symbol, task)
  WHERE status = 'promoted';

-- (Opcional) reglas de promoción parametrizables
DROP TABLE IF EXISTS ml.promotion_rules CASCADE;
CREATE TABLE ml.promotion_rules (
  rule_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol       TEXT DEFAULT '*',
  task         TEXT  DEFAULT '*',
  thresholds   JSONB,        -- {sharpe_min, pf_min, max_dd_max, trades_min, stability_min,...}
  effective_from DATE NOT NULL DEFAULT CURRENT_DATE
);
