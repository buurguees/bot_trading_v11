# core/ml/training/daily_train/runner.py

import os, time, yaml, logging, datetime as dt
import argparse
from dotenv import load_dotenv
from .utils import setup_logging, set_thread_limits, window_from_to, run_cmd, pg_conn, try_lock, release_lock
from .promote import promote_if_better

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
    y = _yaml("config/ml/training.yaml")
    return int(y.get("training", {}).get("modes", {}).get("historical", {}).get("lookback_days", 365))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Ejecuta un solo ciclo y sale")
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
    every_min = int(os.getenv("BT_RETRAIN_MINUTES", "60"))
    win_days  = _lookback_days()      # 365 desde config/ml/training.yaml
    oos_days  = int(os.getenv("BT_OOS_DAYS", "30"))
    infer_days= int(os.getenv("BT_INFER_BACKFILL_DAYS", "2"))
    seed      = int(os.getenv("BT_SEED", "42"))

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

                        # 0) Auto-backfill si falta cobertura
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
                                run_cmd(["python","-m","core.features.indicator_calculator",
                                         "--symbol",symbol,"--tf",tfh,"--from",f_need,"--to",t_need])

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
                        ver_id = promote_if_better(conn, symbol, tf, horizon, {
                            "min_auc":0.52, "max_brier":0.25, "min_acc":0.51,
                            "min_auc_gain":0.01, "tie_breaker":"auc"
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
                            run_cmd(["python","-m","core.ml.backtests.backtest_plans",
                                     "--symbol",symbol,"--tf",tf,"--from",f_oos,"--to",t_oos,
                                     "--fees-bps","2","--slip-bps","2","--max-hold-bars","600"])
                    finally:
                        release_lock(conn, lock)

        if args.once:
            break
        sleep_s = max(30, every_min*60 - int((dt.datetime.now(dt.UTC)-cycle).total_seconds()))
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
