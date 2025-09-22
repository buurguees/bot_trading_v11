-- ============================================================
-- CONSULTAS SQL PARA MONITOREAR ENTRENAMIENTO ML
-- ============================================================

-- 1. PREDICCIONES DE AGENTES (últimas 24h)
-- ============================================================
SELECT 
    task,
    COUNT(*) as total_predicciones,
    AVG(pred_conf) as confianza_promedio,
    MIN(pred_conf) as confianza_min,
    MAX(pred_conf) as confianza_max,
    MAX(created_at) as ultima_prediccion
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY task
ORDER BY task;

-- 2. PREDICCIONES POR HORA (últimas 24h)
-- ============================================================
SELECT 
    DATE_TRUNC('hour', created_at) as hora,
    task,
    COUNT(*) as predicciones,
    AVG(pred_conf) as confianza_promedio
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), task
ORDER BY hora DESC, task
LIMIT 20;

-- 3. TRADE PLANS POR ESTADO (últimas 24h)
-- ============================================================
SELECT 
    status,
    COUNT(*) as cantidad,
    AVG(confidence) as confianza_promedio,
    MAX(created_at) as ultimo_plan
FROM trading.trade_plans 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY status
ORDER BY cantidad DESC;

-- 4. TRADE PLANS POR SÍMBOLO (últimas 24h)
-- ============================================================
SELECT 
    symbol,
    COUNT(*) as planes,
    AVG(confidence) as confianza_promedio,
    COUNT(CASE WHEN status = 'filled' THEN 1 END) as ejecutados
FROM trading.trade_plans 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY symbol
ORDER BY planes DESC;

-- 5. ESTRATEGIAS MINADAS
-- ============================================================
SELECT 
    status,
    COUNT(*) as cantidad,
    MAX(updated_at) as ultima_actualizacion
FROM ml.strategies 
GROUP BY status
ORDER BY cantidad DESC;

-- 6. ESTRATEGIAS RECIENTES
-- ============================================================
SELECT 
    strategy_id,
    symbol,
    timeframe,
    status,
    created_at
FROM ml.strategies 
ORDER BY created_at DESC
LIMIT 10;

-- 7. AGENTES REGISTRADOS
-- ============================================================
SELECT 
    COUNT(*) as total_agentes,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as activos,
    COUNT(CASE WHEN promoted_at IS NOT NULL THEN 1 END) as promovidos,
    MAX(created_at) as ultimo_registro
FROM ml.agents;

-- 8. AGENTES POR TAREA
-- ============================================================
SELECT 
    task,
    COUNT(*) as cantidad,
    AVG(CAST(metrics->>'accuracy' AS FLOAT)) as accuracy_promedio
FROM ml.agents
WHERE metrics->>'accuracy' IS NOT NULL
GROUP BY task
ORDER BY task;

-- 9. CALIDAD DE DATOS - FEATURES
-- ============================================================
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_features,
    MAX(ts) as ultimo_feature,
    MIN(ts) as primer_feature,
    COUNT(*) / EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) * 3600 as features_por_hora
FROM market.features
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;

-- 10. CALIDAD DE DATOS - HISTÓRICO
-- ============================================================
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_velas,
    MAX(ts) as ultima_vela,
    MIN(ts) as primera_vela,
    COUNT(*) / EXTRACT(EPOCH FROM (MAX(ts) - MIN(ts))) * 3600 as velas_por_hora
FROM market.historical_data
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;

-- 11. BACKTEST RUNS
-- ============================================================
SELECT 
    COUNT(*) as total_runs,
    MAX(started_at) as ultimo_run
FROM ml.backtest_runs;

-- 12. BACKTEST TRADES
-- ============================================================
SELECT 
    run_id,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as trades_ganadores,
    AVG(pnl) as pnl_promedio,
    SUM(pnl) as pnl_total
FROM ml.backtest_trades
GROUP BY run_id
ORDER BY run_id DESC
LIMIT 10;

-- 13. PREDICCIONES POR SÍMBOLO (últimas 24h)
-- ============================================================
SELECT 
    symbol,
    task,
    COUNT(*) as predicciones,
    AVG(pred_conf) as confianza_promedio
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY symbol, task
ORDER BY symbol, task;

-- 14. EVOLUCIÓN DE CONFIANZA (últimas 24h)
-- ============================================================
SELECT 
    DATE_TRUNC('hour', created_at) as hora,
    task,
    AVG(pred_conf) as confianza_promedio,
    STDDEV(pred_conf) as desviacion_estandar
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), task
ORDER BY hora DESC, task;

-- 15. TRADE PLANS EJECUTADOS VS PLANIFICADOS
-- ============================================================
SELECT 
    symbol,
    COUNT(*) as total_planes,
    COUNT(CASE WHEN status = 'filled' THEN 1 END) as ejecutados,
    COUNT(CASE WHEN status = 'planned' THEN 1 END) as pendientes,
    COUNT(CASE WHEN status = 'invalid' THEN 1 END) as invalidos,
    ROUND(COUNT(CASE WHEN status = 'filled' THEN 1 END) * 100.0 / COUNT(*), 2) as tasa_ejecucion
FROM trading.trade_plans 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY symbol
ORDER BY tasa_ejecucion DESC;

-- 16. MÉTRICAS DE RENDIMIENTO POR TAREA
-- ============================================================
SELECT 
    task,
    COUNT(*) as total_predicciones,
    AVG(pred_conf) as confianza_promedio,
    COUNT(CASE WHEN pred_conf > 0.7 THEN 1 END) as predicciones_alta_confianza,
    COUNT(CASE WHEN pred_conf < 0.3 THEN 1 END) as predicciones_baja_confianza,
    ROUND(COUNT(CASE WHEN pred_conf > 0.7 THEN 1 END) * 100.0 / COUNT(*), 2) as pct_alta_confianza
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY task
ORDER BY task;

-- 17. ESTRATEGIAS POR RENDIMIENTO
-- ============================================================
SELECT 
    strategy_id,
    symbol,
    timeframe,
    status,
    CAST(metrics_summary->>'total_return' AS FLOAT) as total_return,
    CAST(metrics_summary->>'sharpe_ratio' AS FLOAT) as sharpe_ratio,
    CAST(metrics_summary->>'max_drawdown' AS FLOAT) as max_drawdown,
    created_at
FROM ml.strategies 
WHERE metrics_summary->>'total_return' IS NOT NULL
ORDER BY CAST(metrics_summary->>'total_return' AS FLOAT) DESC
LIMIT 10;

-- 18. RESUMEN DIARIO DE ACTIVIDAD
-- ============================================================
SELECT 
    DATE(created_at) as fecha,
    'agent_preds' as tabla,
    COUNT(*) as registros
FROM ml.agent_preds 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)

UNION ALL

SELECT 
    DATE(created_at) as fecha,
    'trade_plans' as tabla,
    COUNT(*) as registros
FROM trading.trade_plans 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)

ORDER BY fecha DESC, tabla;
