# core/ml/training/daily_train/runner.py

import os, time, yaml, logging, datetime as dt
import argparse
from dotenv import load_dotenv
from .utils import setup_logging, set_thread_limits, window_from_to, run_cmd, pg_conn, try_lock, release_lock
from .promote import promote_if_better
from .balance_manager import load_symbol_config, get_balance_targets

def _yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _db_url():
    # Carga variables desde config/.env si existe
    load_dotenv("config/.env")
    db = os.getenv("DB_URL")
    if db: return db
    p = _yaml("config/system/paths.yaml")
    return p.get("defaults", {}).get("database", {}).get("url")

def _symbols_tfs():
    y = _yaml("config/trading/symbols.yaml")
    defs = y.get("defaults", {})
    syms = y.get("symbols", {})
    out = []
    for sym, cfg in syms.items():
        tfs = cfg.get("timeframes", defs.get("timeframes", []))
        out.append((sym, tfs))
    return out

def _lookback_days():
    # Usar historial completo desde Septiembre 2024 hasta hoy
    # Calcular días desde Septiembre 1, 2024
    from datetime import datetime, timezone
    sept_2024 = datetime(2024, 9, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_since_sept = (now - sept_2024).days
    return max(days_since_sept, 365)  # Mínimo 365 días

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Ejecuta un solo ciclo y sale")
    ap.add_argument("--skip-backfill", action="store_true", help="Omite el auto-backfill de features (para sistemas con features en tiempo real)")
    args = ap.parse_args()
    # logging a fichero + consola (ya gestionado por utils)
    setup_logging("logs/daily_train.log")
    set_thread_limits(int(os.getenv("BT_MAX_CPU", "2")))

    db_url = _db_url()
    if not db_url:
        raise RuntimeError("DB_URL no definido (usa config/.env o config/system/paths.yaml).")

    pairs = _symbols_tfs()            # viene de config/trading/symbols.yaml
    base_tfs = {"1m","5m"}            # entrenamos/ejecutamos en 1m/5m; snapshots TF altos los añade el builder
    horizon = 1
    every_min = int(os.getenv("BT_RETRAIN_MINUTES", "30"))  # Más frecuente
    win_days  = _lookback_days()      # 365 desde config/ml/training.yaml
    oos_days  = int(os.getenv("BT_OOS_DAYS", "365"))  # Usar todo el historial para OOS
    infer_days= int(os.getenv("BT_INFER_BACKFILL_DAYS", "1"))  # Menos días inferencia
    seed      = int(os.getenv("BT_SEED", "42"))
    
    # Configuración de auto-backfill
    # Prioridad: argumento de línea de comandos > variable de entorno > por defecto
    skip_backfill_env = os.getenv("BT_SKIP_BACKFILL", "false").lower() in ("true", "1", "yes")
    auto_backfill = not (args.skip_backfill or skip_backfill_env)
    
    if auto_backfill:
        logging.info("Auto-backfill de features HABILITADO")
    else:
        logging.info("Auto-backfill de features DESHABILITADO (features en tiempo real)")

    while True:
        cycle = dt.datetime.now(dt.UTC)
        with pg_conn(db_url) as conn:
            for symbol, tfs in pairs:
                for tf in tfs:
                    if tf not in base_tfs:
                        continue
                    lock = f"daily_train:{symbol}:{tf}"
                    if not try_lock(conn, lock):
                        logging.info("Lock activo %s-%s; salto.", symbol, tf)
                        continue
                    try:
                        f_train, t_train = window_from_to(win_days)
                        f_oos,   t_oos   = window_from_to(oos_days)

                        # 0) Verificar cobertura de features (solo si auto-backfill está habilitado)
                        if auto_backfill:
                            need = {
                              "1m":  365*24*60 * 0.95,
                              "5m":  365*24*12 * 0.95,
                              "15m": 365*24*4  * 0.95,
                              "1h":  365*24    * 0.95,
                              "4h":  365*6     * 0.95,
                              "1d":  365       * 0.95,
                            }
                            f_need, t_need = window_from_to(win_days)
                            for tfh in ["1m","5m","15m","1h","4h","1d"]:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        """
                                        SELECT COUNT(1) FROM trading.features
                                        WHERE symbol=%s AND timeframe=%s AND timestamp>= %s AND timestamp< %s
                                        """,
                                        (symbol, tfh, f_need, t_need)
                                    )
                                    cnt = cur.fetchone()[0] or 0
                                if cnt < need[tfh]:
                                    logging.warning("Cobertura insuficiente de features para %s-%s: %d/%d", 
                                                  symbol, tfh, cnt, need[tfh])
                                    run_cmd(["python","-m","core.features.indicator_calculator",
                                             "--symbol",symbol,"--tf",tfh,"--from",f_need,"--to",t_need])
                        else:
                            # Solo verificar y reportar cobertura, sin ejecutar backfill
                            f_need, t_need = window_from_to(win_days)
                            for tfh in ["1m","5m","15m","1h","4h","1d"]:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        """
                                        SELECT COUNT(1) FROM trading.features
                                        WHERE symbol=%s AND timeframe=%s AND timestamp>= %s AND timestamp< %s
                                        """,
                                        (symbol, tfh, f_need, t_need)
                                    )
                                    cnt = cur.fetchone()[0] or 0
                                logging.info("Cobertura de features %s-%s: %d registros", symbol, tfh, cnt)

                        # 1) Entrenar (usa dataset con snapshots 15m/1h/4h/1d automáticamente)
                        rc = run_cmd([
                            "python","-m","core.ml.training.train_direction",
                            "--symbol",symbol,"--tf",tf,"--horizon",str(horizon),
                            "--from",f_train,"--to",t_train,"--seed",str(seed),
                            "--max-bars","0"
                        ])
                        if rc != 0:
                            logging.error("Train falló %s-%s", symbol, tf)
                            continue

                        # 2) Promover si mejora métricas (AUC/Brier/ACC)
                        # Cargar configuración de balance para ajustar umbrales
                        balance_config = load_symbol_config(symbol)
                        initial_balance = balance_config["initial"]
                        target_balance = balance_config["target"]
                        
                        # Umbrales ajustados para objetivos realistas
                        balance_ratio = target_balance / initial_balance
                        if balance_ratio >= 50:  # Objetivo muy ambicioso (50x+)
                            min_auc = 0.52
                            max_brier = 0.26
                            min_acc = 0.50
                        elif balance_ratio >= 10:  # Objetivo ambicioso (10x+)
                            min_auc = 0.51
                            max_brier = 0.27
                            min_acc = 0.50
                        elif balance_ratio >= 2:  # Objetivo moderado (2x+)
                            min_auc = 0.50
                            max_brier = 0.28
                            min_acc = 0.50
                        else:  # Objetivo conservador (1.1x)
                            min_auc = 0.50
                            max_brier = 0.30
                            min_acc = 0.50
                        
                        ver_id = promote_if_better(conn, symbol, tf, horizon, {
                            "min_auc": min_auc, 
                            "max_brier": max_brier, 
                            "min_acc": min_acc,
                            "min_auc_gain": 0.005, 
                            "tie_breaker": "auc"
                        })

                        # 3) Inferir + planes últimos N días y backtest OOS como sanity
                        if ver_id:
                            f_inf, t_inf = window_from_to(infer_days)
                            run_cmd(["python","-m","core.ml.inference.infer_bulk",
                                     "--ver-id",str(ver_id),"--symbol",symbol,"--tf",tf,
                                     "--from",f_inf,"--to",t_inf])
                            run_cmd(["python","-m","core.ml.backtests.build_plans_from_signals",
                                     "--symbol",symbol,"--tf",tf,"--ver-id",str(ver_id),
                                     "--from",f_inf,"--to",t_inf])
                            # Backtest con TODO el historial disponible
                            f_hist, t_hist = window_from_to(win_days)  # Usar win_days (365+ días)
                            run_cmd(["python","-m","core.ml.backtests.backtest_plans",
                                     "--symbol",symbol,"--tf",tf,"--from",f_hist,"--to",t_hist,
                                     "--fees-bps","2","--slip-bps","2","--max-hold-bars","600"])
                    finally:
                        release_lock(conn, lock)

        if args.once:
            break
        sleep_s = max(30, every_min*60 - int((dt.datetime.now(dt.UTC)-cycle).total_seconds()))
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
