# 📊 MONITOREO DEL SISTEMA DE ENTRENAMIENTO

Este archivo contiene todas las consultas SQL necesarias para monitorear completamente el sistema de entrenamiento desde pgAdmin 4.

## 🎯 **CONSULTAS PRINCIPALES**

### 1. **ESTADO GENERAL DEL SISTEMA**
```sql
-- Vista general del estado del sistema
SELECT 
    'Versiones de Agentes' as componente,
    COUNT(*) as total,
    COUNT(CASE WHEN promoted = true THEN 1 END) as promovidas,
    MAX(created_at) as ultima_actualizacion
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Predicciones' as componente,
    COUNT(*) as total,
    NULL as promovidas,
    MAX(created_at) as ultima_actualizacion
FROM trading.agentpreds
WHERE created_at >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Señales' as componente,
    COUNT(*) as total,
    NULL as promovidas,
    MAX(created_at) as ultima_actualizacion
FROM trading.agentsignals
WHERE created_at >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Planes de Trading' as componente,
    COUNT(*) as total,
    NULL as promovidas,
    MAX(created_at) as ultima_actualizacion
FROM trading.tradeplans
WHERE created_at >= NOW() - INTERVAL '1 hour'

UNION ALL

SELECT 
    'Backtests' as componente,
    COUNT(*) as total,
    NULL as promovidas,
    MAX(run_ts) as ultima_actualizacion
FROM trading.backtests
WHERE run_ts >= NOW() - INTERVAL '1 hour';
```
**Descripción:** Muestra el estado general de todos los componentes del sistema en la última hora.

---

### 2. **EVOLUCIÓN DEL APRENDIZAJE**
```sql
-- Evolución del aprendizaje por símbolo y timeframe
SELECT 
    (params->>'symbol') as symbol,
    (params->>'timeframe') as timeframe,
    COUNT(*) as total_versiones,
    COUNT(CASE WHEN promoted = true THEN 1 END) as versiones_promovidas,
    MAX(created_at) as ultima_version,
    ROUND(AVG(COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8)), 4) as auc_promedio,
    ROUND(AVG(COALESCE((metrics->>'brier')::float8, (params->'metrics'->>'brier')::float8)), 4) as brier_promedio,
    ROUND(AVG(COALESCE((metrics->>'acc')::float8, (params->'metrics'->>'acc')::float8)), 4) as acc_promedio
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY (params->>'symbol'), (params->>'timeframe')
ORDER BY symbol, timeframe;
```
**Descripción:** Muestra cómo evoluciona el aprendizaje de cada modelo en las últimas 24 horas.

---

### 3. **MODELOS PROMOVIDOS ACTUALES**
```sql
-- Modelos actualmente promovidos y su rendimiento
SELECT 
    (params->>'symbol') as symbol,
    (params->>'timeframe') as timeframe,
    id as version_id,
    created_at,
    COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc,
    COALESCE((metrics->>'brier')::float8, (params->'metrics'->>'brier')::float8) as brier,
    COALESCE((metrics->>'acc')::float8, (params->'metrics'->>'acc')::float8) as accuracy,
    COALESCE((params->'metrics'->>'n_train')::int, 0) as n_train,
    COALESCE((params->'metrics'->>'n_test')::int, 0) as n_test
FROM trading.agentversions
WHERE promoted = true
ORDER BY created_at DESC;
```
**Descripción:** Lista todos los modelos que están actualmente en producción.

---

### 4. **ACTIVIDAD DE ENTRENAMIENTO EN TIEMPO REAL**
```sql
-- Actividad de entrenamiento en la última hora
SELECT 
    (params->>'symbol') as symbol,
    (params->>'timeframe') as timeframe,
    created_at,
    promoted,
    COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc,
    COALESCE((metrics->>'brier')::float8, (params->'metrics'->>'brier')::float8) as brier,
    COALESCE((metrics->>'acc')::float8, (params->'metrics'->>'acc')::float8) as accuracy,
    CASE 
        WHEN promoted = true THEN 'PROMOVIDO'
        ELSE 'ENTRENANDO'
    END as estado
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```
**Descripción:** Muestra la actividad de entrenamiento más reciente.

---

### 5. **PREDICCIONES GENERADAS**
```sql
-- Predicciones generadas en la última hora
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_predicciones,
    MIN(timestamp) as desde,
    MAX(timestamp) as hasta,
    ROUND(AVG((payload->>'prob_up')::float8), 4) as prob_up_promedio
FROM trading.agentpreds
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY symbol, timeframe
ORDER BY total_predicciones DESC;
```
**Descripción:** Muestra las predicciones generadas por el sistema.

---

### 6. **SEÑALES GENERADAS**
```sql
-- Señales generadas en la última hora
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_senales,
    COUNT(CASE WHEN side = 1 THEN 1 END) as senales_long,
    COUNT(CASE WHEN side = -1 THEN 1 END) as senales_short,
    ROUND(AVG(strength), 4) as fuerza_promedio
FROM trading.agentsignals
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY symbol, timeframe
ORDER BY total_senales DESC;
```
**Descripción:** Muestra las señales de trading generadas.

---

### 7. **PLANES DE TRADING GENERADOS**
```sql
-- Planes de trading generados en la última hora
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_planes,
    COUNT(CASE WHEN side = 1 THEN 1 END) as planes_long,
    COUNT(CASE WHEN side = -1 THEN 1 END) as planes_short,
    ROUND(AVG(leverage), 2) as leverage_promedio,
    ROUND(AVG(risk_pct), 4) as risk_pct_promedio,
    ROUND(AVG(qty), 4) as qty_promedio
FROM trading.tradeplans
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY symbol, timeframe
ORDER BY total_planes DESC;
```
**Descripción:** Muestra los planes de trading generados.

---

### 8. **RESULTADOS DE BACKTESTING**
```sql
-- Resultados de backtesting en la última hora
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_backtests,
    ROUND(AVG(n_trades), 0) as trades_promedio,
    ROUND(AVG(net_pnl), 2) as pnl_promedio,
    ROUND(AVG(gross_pnl), 2) as gross_pnl_promedio,
    ROUND(AVG(fees), 2) as fees_promedio,
    ROUND(AVG(win_rate), 4) as win_rate_promedio,
    ROUND(AVG(max_dd), 4) as max_drawdown_promedio
FROM trading.backtests
WHERE run_ts >= NOW() - INTERVAL '1 hour'
GROUP BY symbol, timeframe
ORDER BY pnl_promedio DESC;
```
**Descripción:** Muestra los resultados de los backtests realizados.

---

### 9. **MEMORIA DE ESTRATEGIAS**
```sql
-- Memoria de estrategias (rendimiento histórico)
SELECT 
    symbol,
    timeframe,
    signature,
    n_trades,
    ROUND(win_rate, 4) as win_rate,
    ROUND(avg_pnl, 4) as avg_pnl,
    ROUND(sharpe, 4) as sharpe,
    ROUND(avg_hold_bars, 0) as avg_hold_bars,
    last_updated
FROM trading.strategy_memory
WHERE last_updated >= NOW() - INTERVAL '24 hours'
ORDER BY sharpe DESC, n_trades DESC;
```
**Descripción:** Muestra el rendimiento histórico de las estrategias.

---

### 10. **ALERTAS Y MONITOREO DE ERRORES**
```sql
-- Verificar si hay errores o problemas
SELECT 
    'Versiones sin métricas' as tipo_alerta,
    COUNT(*) as cantidad
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '1 hour'
  AND (metrics IS NULL OR metrics = '{}')
  AND (params->'metrics' IS NULL OR params->'metrics' = '{}')

UNION ALL

SELECT 
    'Predicciones sin payload' as tipo_alerta,
    COUNT(*) as cantidad
FROM trading.agentpreds
WHERE created_at >= NOW() - INTERVAL '1 hour'
  AND (payload IS NULL OR payload = '{}')

UNION ALL

SELECT 
    'Señales con fuerza 0' as tipo_alerta,
    COUNT(*) as cantidad
FROM trading.agentsignals
WHERE created_at >= NOW() - INTERVAL '1 hour'
  AND strength = 0;
```
**Descripción:** Detecta posibles problemas en el sistema.

---

## 🔄 **CONSULTAS DE MONITOREO CONTINUO**

### **Para ejecutar cada 5 minutos:**
```sql
-- Monitoreo rápido del estado
SELECT 
    NOW() as timestamp,
    COUNT(*) as versiones_ultima_hora,
    COUNT(CASE WHEN promoted = true THEN 1 END) as promovidas_ultima_hora
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '1 hour';
```

### **Para verificar el progreso del entrenamiento:**
```sql
-- Progreso del entrenamiento actual
SELECT 
    (params->>'symbol') as symbol,
    (params->>'timeframe') as timeframe,
    created_at,
    COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc,
    CASE 
        WHEN COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) > 0.55 THEN 'LISTO'
        WHEN COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) > 0.50 THEN 'ENTRENANDO'
        ELSE 'NECESITA MÁS DATOS'
    END as estado
FROM trading.agentversions
WHERE created_at >= NOW() - INTERVAL '30 minutes'
ORDER BY created_at DESC;
```

---

## 📊 **DASHBOARD COMPLETO**

### **Vista resumen del sistema:**
```sql
-- Dashboard completo del sistema
WITH stats AS (
    SELECT 
        (params->>'symbol') as symbol,
        (params->>'timeframe') as timeframe,
        COUNT(*) as total_versiones,
        COUNT(CASE WHEN promoted = true THEN 1 END) as promovidas,
        MAX(created_at) as ultima_version,
        ROUND(AVG(COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8)), 4) as auc_promedio
    FROM trading.agentversions
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY (params->>'symbol'), (params->>'timeframe')
),
preds AS (
    SELECT 
        symbol,
        timeframe,
        COUNT(*) as total_predicciones
    FROM trading.agentpreds
    WHERE created_at >= NOW() - INTERVAL '1 hour'
    GROUP BY symbol, timeframe
),
plans AS (
    SELECT 
        symbol,
        timeframe,
        COUNT(*) as total_planes
    FROM trading.tradeplans
    WHERE created_at >= NOW() - INTERVAL '1 hour'
    GROUP BY symbol, timeframe
)
SELECT 
    s.symbol,
    s.timeframe,
    s.total_versiones,
    s.promovidas,
    s.ultima_version,
    s.auc_promedio,
    COALESCE(p.total_predicciones, 0) as predicciones_1h,
    COALESCE(pl.total_planes, 0) as planes_1h,
    CASE 
        WHEN s.auc_promedio > 0.55 THEN 'EXCELENTE'
        WHEN s.auc_promedio > 0.52 THEN 'BUENO'
        WHEN s.auc_promedio > 0.50 THEN 'ACEPTABLE'
        ELSE 'MEJORAR'
    END as estado_aprendizaje
FROM stats s
LEFT JOIN preds p ON s.symbol = p.symbol AND s.timeframe = p.timeframe
LEFT JOIN plans pl ON s.symbol = pl.symbol AND s.timeframe = pl.timeframe
ORDER BY s.auc_promedio DESC, s.symbol, s.timeframe;
```

---

## 🎯 **INDICADORES CLAVE DE RENDIMIENTO (KPIs)**

### **Métricas de Calidad del Modelo:**
- **AUC > 0.55**: Excelente rendimiento
- **AUC 0.52-0.55**: Bueno
- **AUC 0.50-0.52**: Aceptable
- **AUC < 0.50**: Necesita mejora

### **Métricas de Actividad:**
- **Versiones por hora**: Debe ser > 0 durante entrenamiento activo
- **Predicciones por hora**: Debe ser > 0 si hay datos nuevos
- **Señales por hora**: Debe ser > 0 si hay predicciones válidas
- **Planes por hora**: Debe ser > 0 si hay señales válidas

### **Métricas de Estabilidad:**
- **Tasa de promoción**: % de versiones que se promueven
- **Consistencia temporal**: Actividad regular sin interrupciones
- **Calidad de datos**: Sin alertas de errores

---

## 📝 **NOTAS DE USO**

1. **Ejecutar consultas principales** cada 15-30 minutos durante el entrenamiento
2. **Monitoreo continuo** cada 5 minutos para detectar problemas
3. **Dashboard completo** una vez por hora para vista general
4. **Alertas** revisar inmediatamente si hay errores
5. **KPIs** evaluar al final de cada sesión de entrenamiento

---

## 💰 **CONSULTAS DE BALANCE Y OBJETIVOS**

### **Configuración de Balance por Símbolo:**
```sql
-- Verificar configuración de balance en training.yaml
-- Esta consulta muestra los balances configurados (ejemplo conceptual)
SELECT 
    'BTCUSDT' as symbol,
    1000.0 as balance_inicial,
    100000.0 as balance_objetivo,
    100.0 as multiplicador_objetivo,
    0.02 as risk_per_trade,
    5 as min_leverage,
    80 as max_leverage,
    'MUY AMBICIOSO' as nivel_objetivo
UNION ALL
SELECT 
    'ETHUSDT' as symbol,
    1000.0 as balance_inicial,
    100000.0 as balance_objetivo,
    100.0 as multiplicador_objetivo,
    0.02 as risk_per_trade,
    3 as min_leverage,
    50 as max_leverage,
    'MUY AMBICIOSO' as nivel_objetivo
UNION ALL
SELECT 
    'ADAUSDT' as symbol,
    5000.0 as balance_inicial,
    10000.0 as balance_objetivo,
    2.0 as multiplicador_objetivo,
    0.015 as risk_per_trade,
    3 as min_leverage,
    30 as max_leverage,
    'MODERADO' as nivel_objetivo
UNION ALL
SELECT 
    'SOLUSDT' as symbol,
    6000.0 as balance_inicial,
    12000.0 as balance_objetivo,
    2.0 as multiplicador_objetivo,
    0.02 as risk_per_trade,
    5 as min_leverage,
    50 as max_leverage,
    'MODERADO' as nivel_objetivo
UNION ALL
SELECT 
    'DOGEUSDT' as symbol,
    3000.0 as balance_inicial,
    6000.0 as balance_objetivo,
    2.0 as multiplicador_objetivo,
    0.01 as risk_per_trade,
    3 as min_leverage,
    20 as max_leverage,
    'MODERADO' as nivel_objetivo
UNION ALL
SELECT 
    'XRPUSDT' as symbol,
    5000.0 as balance_inicial,
    10000.0 as balance_objetivo,
    2.0 as multiplicador_objetivo,
    0.015 as risk_per_trade,
    3 as min_leverage,
    30 as max_leverage,
    'MODERADO' as nivel_objetivo;
```

### **Progreso hacia Objetivos de Balance:**
```sql
-- Simular progreso de balance (necesitarías una tabla de trades reales)
WITH balance_simulation AS (
    SELECT 
        symbol,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN side = 1 THEN 1 END) as long_trades,
        COUNT(CASE WHEN side = -1 THEN 1 END) as short_trades,
        ROUND(AVG(leverage), 2) as avg_leverage,
        ROUND(AVG(risk_pct), 4) as avg_risk_pct
    FROM trading.tradeplans
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY symbol
)
SELECT 
    symbol,
    total_trades,
    long_trades,
    short_trades,
    ROUND((long_trades::numeric / total_trades)::numeric, 3) as long_ratio,
    ROUND((short_trades::numeric / total_trades)::numeric, 3) as short_ratio,
    avg_leverage,
    avg_risk_pct,
    CASE 
        WHEN ROUND((long_trades::numeric / total_trades)::numeric, 1) = 0.5 THEN 'BALANCEADO'
        WHEN ROUND((long_trades::numeric / total_trades)::numeric, 1) > 0.6 THEN 'MUCHOS LONGS'
        WHEN ROUND((long_trades::numeric / total_trades)::numeric, 1) < 0.4 THEN 'MUCHOS SHORTS'
        ELSE 'AJUSTANDO'
    END as balance_status
FROM balance_simulation
ORDER BY total_trades DESC;
```

### **Monitoreo de Apalancamiento Dinámico:**
```sql
-- Análisis de apalancamiento usado en planes recientes
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_planes,
    ROUND(MIN(leverage), 2) as min_leverage,
    ROUND(MAX(leverage), 2) as max_leverage,
    ROUND(AVG(leverage), 2) as avg_leverage,
    ROUND(STDDEV(leverage), 2) as leverage_std,
    ROUND(AVG(risk_pct), 4) as avg_risk_pct,
    CASE 
        WHEN AVG(leverage) > 20 THEN 'ALTO RIESGO'
        WHEN AVG(leverage) > 10 THEN 'RIESGO MEDIO'
        ELSE 'RIESGO BAJO'
    END as risk_level
FROM trading.tradeplans
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY symbol, timeframe
ORDER BY avg_leverage DESC;
```

### **Umbrales de Promoción por Objetivo:**
```sql
-- Verificar umbrales de promoción según objetivos de balance
SELECT 
    symbol,
    balance_ratio,
    min_auc_required,
    max_brier_allowed,
    min_acc_required,
    nivel_objetivo
FROM (
    SELECT 
        'BTCUSDT' as symbol,
        100.0 as balance_ratio,
        0.60 as min_auc_required,
        0.20 as max_brier_allowed,
        0.55 as min_acc_required,
        'MUY AMBICIOSO (100x)' as nivel_objetivo
    UNION ALL
    SELECT 
        'ETHUSDT' as symbol,
        100.0 as balance_ratio,
        0.60 as min_auc_required,
        0.20 as max_brier_allowed,
        0.55 as min_acc_required,
        'MUY AMBICIOSO (100x)' as nivel_objetivo
    UNION ALL
    SELECT 
        'ADAUSDT' as symbol,
        2.0 as balance_ratio,
        0.52 as min_auc_required,
        0.25 as max_brier_allowed,
        0.50 as min_acc_required,
        'MODERADO (2x)' as nivel_objetivo
    UNION ALL
    SELECT 
        'SOLUSDT' as symbol,
        2.0 as balance_ratio,
        0.52 as min_auc_required,
        0.25 as max_brier_allowed,
        0.50 as min_acc_required,
        'MODERADO (2x)' as nivel_objetivo
    UNION ALL
    SELECT 
        'DOGEUSDT' as symbol,
        2.0 as balance_ratio,
        0.52 as min_auc_required,
        0.25 as max_brier_allowed,
        0.50 as min_acc_required,
        'MODERADO (2x)' as nivel_objetivo
    UNION ALL
    SELECT 
        'XRPUSDT' as symbol,
        2.0 as balance_ratio,
        0.52 as min_auc_required,
        0.25 as max_brier_allowed,
        0.50 as min_acc_required,
        'MODERADO (2x)' as nivel_objetivo
) thresholds
ORDER BY balance_ratio DESC;
```

### **Análisis Completo de Rendimiento por Símbolo/Timeframe:**
```sql
-- Análisis completo de rendimiento con balance inicial/objetivo
WITH balance_config AS (
    -- Configuración de balance desde training.yaml
    SELECT 'BTCUSDT' as symbol, 1000.0 as balance_inicial, 100000.0 as balance_objetivo
    UNION ALL SELECT 'ETHUSDT', 1000.0, 100000.0
    UNION ALL SELECT 'ADAUSDT', 5000.0, 10000.0
    UNION ALL SELECT 'SOLUSDT', 6000.0, 12000.0
    UNION ALL SELECT 'DOGEUSDT', 3000.0, 6000.0
    UNION ALL SELECT 'XRPUSDT', 5000.0, 10000.0
),
backtest_stats AS (
    SELECT 
        b.symbol,
        b.timeframe,
        COUNT(*) as total_backtests,
        ROUND(AVG(b.n_trades), 0) as avg_trades,
        ROUND(AVG(b.net_pnl), 2) as avg_pnl_total,
        ROUND(AVG(b.net_pnl) / NULLIF(AVG(b.n_trades), 0), 2) as avg_pnl_per_trade,
        ROUND(AVG(b.net_pnl) / NULLIF(AVG(EXTRACT(EPOCH FROM (b.to_ts - b.from_ts))) / 86400, 0), 2) as avg_pnl_daily,
        ROUND(AVG(b.win_rate), 4) as win_rate,
        ROUND(AVG(b.fees), 2) as avg_fees,
        ROUND(AVG(b.max_dd), 4) as avg_max_drawdown,
        MAX(b.run_ts) as ultimo_backtest
    FROM trading.backtests b
    WHERE b.run_ts >= NOW() - INTERVAL '7 days'
    GROUP BY b.symbol, b.timeframe
),
plan_stats AS (
    SELECT 
        p.symbol,
        p.timeframe,
        COUNT(*) as total_planes,
        COUNT(CASE WHEN p.side = 1 THEN 1 END) as planes_long,
        COUNT(CASE WHEN p.side = -1 THEN 1 END) as planes_short,
        ROUND(AVG(p.leverage), 2) as avg_leverage,
        ROUND(AVG(p.risk_pct), 4) as avg_risk_pct,
        ROUND(AVG(p.qty), 4) as avg_qty
    FROM trading.tradeplans p
    WHERE p.created_at >= NOW() - INTERVAL '7 days'
    GROUP BY p.symbol, p.timeframe
)
SELECT 
    bs.symbol,
    bs.timeframe,
    bc.balance_inicial,
    bc.balance_objetivo,
    ROUND((bc.balance_objetivo / bc.balance_inicial), 1) as multiplicador_objetivo,
    bs.avg_pnl_daily,
    bs.avg_pnl_per_trade,
    bs.avg_trades,
    ps.planes_long,
    ps.planes_short,
    ROUND((ps.planes_long::numeric / NULLIF(ps.total_planes, 0))::numeric, 3) as ratio_longs,
    ROUND((ps.planes_short::numeric / NULLIF(ps.total_planes, 0))::numeric, 3) as ratio_shorts,
    ROUND(bs.win_rate * 100, 2) as winrate_pct,
    ps.avg_leverage,
    ps.avg_risk_pct,
    bs.avg_fees,
    bs.avg_max_drawdown,
    bs.ultimo_backtest,
    CASE 
        WHEN ROUND((bc.balance_objetivo / bc.balance_inicial), 1) >= 50 THEN 'MUY AMBICIOSO'
        WHEN ROUND((bc.balance_objetivo / bc.balance_inicial), 1) >= 10 THEN 'AMBICIOSO'
        WHEN ROUND((bc.balance_objetivo / bc.balance_inicial), 1) >= 2 THEN 'MODERADO'
        ELSE 'CONSERVADOR'
    END as nivel_objetivo,
    CASE 
        WHEN bs.win_rate >= 0.6 THEN 'EXCELENTE'
        WHEN bs.win_rate >= 0.55 THEN 'BUENO'
        WHEN bs.win_rate >= 0.5 THEN 'ACEPTABLE'
        ELSE 'MEJORAR'
    END as estado_winrate
FROM backtest_stats bs
LEFT JOIN plan_stats ps ON bs.symbol = ps.symbol AND bs.timeframe = ps.timeframe
LEFT JOIN balance_config bc ON bs.symbol = bc.symbol
ORDER BY 
    bc.balance_objetivo / bc.balance_inicial DESC,  -- Objetivos más ambiciosos primero
    bs.win_rate DESC,                               -- Mejor win rate primero
    bs.avg_pnl_daily DESC;                          -- Mejor PnL diario primero
```

### **Resumen Ejecutivo de Rendimiento:**
```sql
-- Resumen ejecutivo con métricas clave
WITH balance_config AS (
    SELECT 'BTCUSDT' as symbol, 1000.0 as balance_inicial, 100000.0 as balance_objetivo
    UNION ALL SELECT 'ETHUSDT', 1000.0, 100000.0
    UNION ALL SELECT 'ADAUSDT', 5000.0, 10000.0
    UNION ALL SELECT 'SOLUSDT', 6000.0, 12000.0
    UNION ALL SELECT 'DOGEUSDT', 3000.0, 6000.0
    UNION ALL SELECT 'XRPUSDT', 5000.0, 10000.0
),
performance_summary AS (
    SELECT 
        b.symbol,
        b.timeframe,
        COUNT(*) as backtests_count,
        ROUND(AVG(b.net_pnl) / NULLIF(AVG(EXTRACT(EPOCH FROM (b.to_ts - b.from_ts))) / 86400, 0), 2) as avg_pnl_daily,
        ROUND(AVG(b.win_rate) * 100, 1) as winrate_pct,
        ROUND(AVG(b.n_trades), 0) as avg_trades,
        ROUND(AVG(p.leverage), 1) as avg_leverage,
        COUNT(CASE WHEN p.side = 1 THEN 1 END) as longs,
        COUNT(CASE WHEN p.side = -1 THEN 1 END) as shorts
    FROM trading.backtests b
    LEFT JOIN trading.tradeplans p ON b.symbol = p.symbol AND b.timeframe = p.timeframe
    WHERE b.run_ts >= NOW() - INTERVAL '7 days'
    GROUP BY b.symbol, b.timeframe
)
SELECT 
    ps.symbol,
    ps.timeframe,
    bc.balance_inicial,
    bc.balance_objetivo,
    ROUND((bc.balance_objetivo / bc.balance_inicial), 1) as objetivo_x,
    ps.avg_pnl_daily,
    ps.winrate_pct,
    ps.avg_trades,
    ps.avg_leverage,
    ps.longs,
    ps.shorts,
    ROUND((ps.longs::numeric / NULLIF(ps.longs + ps.shorts, 0) * 100)::numeric, 1) as longs_pct,
    ROUND((ps.shorts::numeric / NULLIF(ps.longs + ps.shorts, 0) * 100)::numeric, 1) as shorts_pct
FROM performance_summary ps
LEFT JOIN balance_config bc ON ps.symbol = bc.symbol
ORDER BY ps.avg_pnl_daily DESC;
```

---

## 🔧 **SOLUCIÓN DE PROBLEMAS**

### **Si no hay actividad:**
- Verificar que el proceso de entrenamiento esté ejecutándose
- Revisar logs del sistema
- Verificar conectividad a la base de datos

### **Si hay errores:**
- Revisar la consulta de alertas
- Verificar logs de la aplicación
- Comprobar configuración de la base de datos

### **Si el rendimiento es bajo:**
- Verificar calidad de los datos de entrada
- Revisar parámetros de entrenamiento
- Considerar ajustar umbrales de promoción

### **Si el balance no progresa:**
- Verificar configuración de balance en symbols.yaml
- Revisar proporción de longs/shorts
- Ajustar apalancamiento dinámico
- Verificar umbrales de promoción por símbolo
