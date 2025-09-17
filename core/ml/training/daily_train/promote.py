import logging

def _cols(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='trading' AND table_name='agentversions'
        """)
        return {r[0] for r in cur.fetchall()}

def _metric_exprs(has_metrics_col: bool):
    # Devuelve expresiones SQL para auc/brier/acc según exista o no la columna metrics
    if has_metrics_col:
        auc   = "COALESCE((metrics->>'auc')::float8,   (params->'metrics'->>'auc')::float8)"
        brier = "COALESCE((metrics->>'brier')::float8, (params->'metrics'->>'brier')::float8)"
        acc   = "COALESCE((metrics->>'acc')::float8,   (params->'metrics'->>'acc')::float8)"
    else:
        # Solo en params->metrics
        auc   = "(params->'metrics'->>'auc')::float8"
        brier = "(params->'metrics'->>'brier')::float8"
        acc   = "(params->'metrics'->>'acc')::float8"
    return auc, brier, acc

def _select_last(conn, symbol, tf, horizon, only_promoted=False):
    cols = _cols(conn)
    has_plain   = {'symbol','timeframe','horizon'}.issubset(cols)
    has_metrics = 'metrics' in cols
    auc, brier, acc = _metric_exprs(has_metrics)

    promo_clause = "AND promoted = true" if only_promoted else ""

    if has_plain:
        # Esquema con columnas planas disponible (más rápido)
        sql = f"""
            SELECT id, {auc} AS auc, {brier} AS brier, {acc} AS acc
            FROM trading.agentversions
            WHERE symbol=%s AND timeframe=%s AND horizon=%s
            {promo_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """
        params = (symbol, tf, horizon)
    else:
        # Solo JSONB en params
        sql = f"""
            SELECT id, {auc} AS auc, {brier} AS brier, {acc} AS acc
            FROM trading.agentversions
            WHERE (params->>'symbol')=%s AND (params->>'timeframe')=%s AND ((params->>'horizon')::int)=%s
            {promo_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """
        params = (symbol, tf, horizon)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()  # (id, auc, brier, acc) | None

def get_promoted(conn, symbol, tf, horizon):
    return _select_last(conn, symbol, tf, horizon, only_promoted=True)

def get_last_trained(conn, symbol, tf, horizon):
    return _select_last(conn, symbol, tf, horizon, only_promoted=False)

def promote_if_better(conn, symbol, tf, horizon, rules: dict) -> int | None:
    """
    Promueve la última versión si supera umbrales y mejora vs la promoted actual.
    Soporta tablas con columnas planas o sólo JSONB en params/metrics.
    """
    last = get_last_trained(conn, symbol, tf, horizon)
    if not last:
        logging.info("[PROMOTE] No hay versiones recientes para %s-%s", symbol, tf)
        return None

    last_id, last_auc, last_brier, last_acc = last
    if last_auc is None or last_brier is None or last_acc is None:
        logging.info("[PROMOTE] Métricas incompletas para %s-%s ver=%s", symbol, tf, last_id)
        return None

    # Umbrales mínimos
    if (last_auc < rules["min_auc"]
        or last_brier > rules["max_brier"]
        or last_acc < rules["min_acc"]):
        logging.info("[PROMOTE] Rechazado por umbrales (auc=%.4f brier=%.4f acc=%.4f)",
                     last_auc, last_brier, last_acc)
        return None

    # ¿Hay promovida actual?
    cur_prom = get_promoted(conn, symbol, tf, horizon)
    better = False
    if cur_prom:
        _, p_auc, p_brier, p_acc = cur_prom
        if   rules["tie_breaker"] == "auc":
            better = (last_auc >= (p_auc or 0) + rules["min_auc_gain"])
        elif rules["tie_breaker"] == "brier":
            better = ((p_brier or 1) - last_brier) >= rules["min_auc_gain"]
        else:  # acc
            better = (last_acc >= (p_acc or 0) + rules["min_auc_gain"])
    else:
        better = True  # no hay promoted → promueve

    if not better:
        logging.info("[PROMOTE] No supera mejora mínima vs promoted actual")
        return None

    # Despromueve anteriores del mismo (symbol, tf, horizon) y promueve la nueva
    cols = _cols(conn)
    has_plain = {'symbol','timeframe','horizon'}.issubset(cols)

    with conn.cursor() as cur:
        if has_plain:
            cur.execute("""
                UPDATE trading.agentversions
                SET promoted = false
                WHERE promoted = true
                  AND symbol=%s AND timeframe=%s AND horizon=%s;
            """, (symbol, tf, horizon))
        else:
            cur.execute("""
                UPDATE trading.agentversions
                SET promoted = false
                WHERE promoted = true
                  AND (params->>'symbol')=%s
                  AND (params->>'timeframe')=%s
                  AND ((params->>'horizon')::int)=%s;
            """, (symbol, tf, horizon))

        cur.execute("UPDATE trading.agentversions SET promoted=true WHERE id=%s;", (last_id,))

    logging.info("[PROMOTE] Promovida ver_id=%s para %s-%s (auc=%.4f brier=%.4f acc=%.4f)",
                 last_id, symbol, tf, last_auc, last_brier, last_acc)
    return last_id
