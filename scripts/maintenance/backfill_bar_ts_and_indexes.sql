-- Backfill rápido de bar_ts aproximado desde created_at (redondeado al minuto)
UPDATE trading."TradePlans"
SET bar_ts = date_trunc('minute', created_at)
WHERE bar_ts IS NULL;

-- Índices para predicciones y señales
CREATE INDEX IF NOT EXISTS idx_pred_ver_ts
  ON trading."AgentPreds" (agent_version_id, timestamp DESC);

-- Índice funcional sobre campo JSONB meta->>'direction_ver_id'
CREATE INDEX IF NOT EXISTS idx_sig_meta_ts
  ON trading."AgentSignals" ((meta->>'direction_ver_id'), timestamp DESC);


