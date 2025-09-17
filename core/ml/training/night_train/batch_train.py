import os, json, math, argparse, pathlib, itertools, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

# Imports del proyecto
from core.ml.datasets.builder import build_dataset, FEATURES
from core.ml.utils.seeds import set_global_seeds

# Cargar variables de entorno
load_dotenv("config/.env")

# --- Utils DB ---------------------------------------------------------------

def get_engine():
    url = os.getenv("DB_URL")
    if not url:
        raise RuntimeError("DB_URL no está definido en .env")
    return create_engine(url, pool_pre_ping=True, future=True)

def ensure_agent(conn, name: str, kind: str = "direction") -> int:
    # Primero intentar obtener el ID existente
    q_check = text("SELECT id FROM trading.Agents WHERE name = :n")
    existing = conn.execute(q_check, {"n": name}).fetchone()
    
    if existing:
        return existing[0]
    
    # Si no existe, insertarlo
    q_insert = text("""
        INSERT INTO trading.Agents(name, kind)
        VALUES (:n, :k)
        RETURNING id
    """)
    return conn.execute(q_insert, {"n": name, "k": kind}).scalar()

def register_version(conn, agent_id: int, version: str,
                     params: Dict, artifact_uri: str,
                     train_start: pd.Timestamp, train_end: pd.Timestamp,
                     metrics: Dict, promoted: bool) -> int:
    q = text("""
        INSERT INTO trading.AgentVersions(
            agent_id, version, params, artifact_uri, train_start, train_end, metrics, promoted
        )
        VALUES (:aid, :ver, :par, :uri, :ts, :te, :met, :pro)
        RETURNING id
    """)
    return conn.execute(q, {
        "aid": agent_id,
        "ver": version,
        "par": json.dumps(params),
        "uri": artifact_uri,
        "ts": train_start,
        "te": train_end,
        "met": json.dumps(metrics),
        "pro": promoted
    }).scalar()

# --- Dataset loader ---------------------------------------------------------

# Devuelve features X y label y (0/1) para (symbol, timeframe, horizon) en un rango de días
def load_dataset(conn, symbol: str, tf: str, days_back: int, horizon: int) -> pd.DataFrame:
    # Usar las features reales del proyecto
    features = [
        "rsi14", "ema20", "ema50", "ema200", "macd", "macd_signal", "macd_hist",
        "atr14", "bb_mid", "bb_upper", "bb_lower", "obv", "supertrend", "st_dir"
    ]
    
    q = text(f"""
        WITH base AS (
          SELECT f.timestamp,
                 {', '.join('f.' + feat for feat in features)},
                 h.close AS close_now,
                 LEAD(h.close, :hz) OVER (ORDER BY f.timestamp) AS close_fwd
          FROM trading.Features f
          JOIN trading.HistoricalData h
            ON h.symbol=f.symbol AND h.timeframe=f.timeframe AND h.timestamp=f.timestamp
          WHERE f.symbol=:s AND f.timeframe=:tf
            AND f.timestamp >= (NOW() - INTERVAL '{days_back} days')
          ORDER BY f.timestamp
        )
        SELECT *, CASE WHEN close_fwd > close_now THEN 1 ELSE 0 END AS y
        FROM base
        -- fuera la última(s) fila(s) sin close_fwd
        WHERE close_fwd IS NOT NULL
        ORDER BY timestamp
    """)
    df = pd.read_sql(q, conn, params={"s": symbol, "tf": tf, "hz": int(horizon)})
    # Asegurar tipo datetime (sin TZ para sklearn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df

# --- Time series CV ---------------------------------------------------------

def iter_time_splits(ts: pd.Series, n_splits: int, embargo_minutes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Genera índices (train_idx, valid_idx) respetando el tiempo + embargo."""
    n = len(ts)
    if n_splits < 2:
        yield (np.arange(0, n), np.array([], dtype=int))
        return
    split_points = np.linspace(0.5, 0.9, n_splits)  # % del historial usado para train
    for sp in split_points:
        cut = int(n * sp)
        if cut <= 0 or cut >= n-1:
            continue
        t_valid_start = ts.iloc[cut]
        embargo = pd.Timedelta(minutes=embargo_minutes)
        # train: todo < (valid_start - embargo)
        train_idx = np.where(ts < (t_valid_start - embargo))[0]
        # valid: [valid_start, valid_start + 5% del historial]
        valid_span = max(1, int(n * 0.05))
        valid_idx = np.arange(cut, min(cut + valid_span, n))
        if len(train_idx) < 100 or len(valid_idx) < 20:
            continue
        yield (train_idx, valid_idx)

# --- Modelo -----------------------------------------------------------------

@dataclass
class TrainCfg:
    symbol: str
    timeframe: str
    horizon: int
    days_back: int
    n_splits: int
    embargo_minutes: int
    model_kind: str
    param_grid: Dict
    max_iter: int
    min_rows: int
    dropna_cols_min_fraction: float
    artifacts_dir: str
    version_tag: str
    promote_if: Dict

def train_one_combo(cfg: TrainCfg, db_url: str) -> Dict:
    # Establecer semillas para reproducibilidad
    set_global_seeds(42)
    
    # Conexión por proceso
    eng = create_engine(db_url, pool_pre_ping=True, future=True)
    
    # Cargar datos
    with eng.connect() as c:
        df = load_dataset(c, cfg.symbol, cfg.timeframe, cfg.days_back, cfg.horizon)

    if len(df) < cfg.min_rows:
        return {"status": "skipped", "reason": f"pocos datos ({len(df)})", "symbol": cfg.symbol, "tf": cfg.timeframe, "h": cfg.horizon}

    # Selección de columnas: quitamos metadata
    drop_cols = {"timestamp", "y", "close_now", "close_fwd", "symbol", "timeframe"}
    feat_cols = [col for col in df.columns if col not in drop_cols]

    # Quitar columnas con demasiados NaN
    keep = []
    for col in feat_cols:
        frac = df[col].notna().mean()
        if frac >= cfg.dropna_cols_min_fraction:
            keep.append(col)
    X = df[keep].astype(float).ffill().bfill().values
    y = df["y"].astype(int).values
    t = df["timestamp"]

    # Pipeline
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=cfg.max_iter))
    ])

    # Grid search manual (pequeño) con walk-forward
    best_params, best_score = None, -np.inf
    scores = []
    for params in itertools.product(*cfg.param_grid.values()):
        cand = dict(zip(cfg.param_grid.keys(), params))
        
        # Manejar class_weight null correctamente
        if cand.get('class_weight') == 'null' or cand.get('class_weight') is None:
            cand['class_weight'] = None
        
        fold_scores = []
        for tr_idx, va_idx in iter_time_splits(t, cfg.n_splits, cfg.embargo_minutes):
            if len(va_idx) == 0:  # por si acaso
                continue
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xva, yva = X[va_idx], y[va_idx]
            model = Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=cfg.max_iter, **cand))])
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xva)[:,1]
            auc = roc_auc_score(yva, p)
            fold_scores.append(auc)
        if fold_scores:
            m = float(np.mean(fold_scores))
            scores.append((cand, m))
            if m > best_score:
                best_score, best_params = m, cand

    # Reentreno final con mejores hiperparámetros
    final_model = Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=cfg.max_iter, **(best_params or {})))])
    final_model.fit(X, y)

    # Métricas finales en la cola (10% últimos)
    tail = max(2000, int(len(y)*0.1))
    Xte, yte = X[-tail:], y[-tail:]
    p = final_model.predict_proba(Xte)[:,1]
    auc = float(roc_auc_score(yte, p))
    brier = float(brier_score_loss(yte, p))
    acc = float(accuracy_score(yte, (p>=0.5).astype(int)))

    # Guardar artefacto
    out_dir = pathlib.Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{cfg.symbol}_{cfg.timeframe}_H{cfg.horizon}_logreg.pkl"
    fpath = out_dir / fname
    joblib.dump(final_model, fpath)

    # Registrar versión en DB
    agent_name = "direction_clf"
    ver = cfg.version_tag
    params_for_db = {
        "symbol": cfg.symbol, "tf": cfg.timeframe, "h": cfg.horizon,
        "model": "LogReg", "params": best_params or {}
    }
    metrics = {"auc": auc, "brier": brier, "acc": acc,
               "n_rows": int(len(df)), "tail": int(tail), "cv_best_auc": float(best_score)}

    train_start, train_end = t.iloc[0], t.iloc[-1]

    with eng.begin() as tx:
        agent_id = ensure_agent(tx, agent_name, "direction")
        ver_id = register_version(
            tx, agent_id, ver, params_for_db, str(fpath),
            train_start, train_end, metrics,
            promoted=(auc >= cfg.promote_if["min_auc"] and brier <= cfg.promote_if["max_brier"])
        )

    return {"status": "ok", "symbol": cfg.symbol, "tf": cfg.timeframe,
                "h": cfg.horizon, "auc": auc, "brier": brier, "acc": acc,
                "cv_best_auc": float(best_score), "artifact": str(fpath)}

# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="ruta YAML del plan")
    ap.add_argument("--workers", type=int, default=None, help="override num_workers")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL no está en el entorno")

    combos = list(itertools.product(cfg["symbols"], cfg["timeframes"], cfg["horizons"]))
    print(f"[INFO] Entrenando {len(combos)} combinaciones…")

    def make_cfg(sym, tf, h):
        return TrainCfg(
            symbol=sym, timeframe=tf, horizon=int(h),
            days_back=int(cfg["days_back"]),
            n_splits=int(cfg["n_splits"]),
            embargo_minutes=int(cfg["embargo_minutes"]),
            model_kind=cfg["model"]["kind"],
            param_grid=cfg["model"]["param_grid"],
            max_iter=int(cfg["model"]["max_iter"]),
            min_rows=int(cfg["min_rows"]),
            dropna_cols_min_fraction=float(cfg["dropna_cols_min_fraction"]),
            artifacts_dir=cfg["artifacts_dir"],
            version_tag=cfg["version_tag"],
            promote_if=cfg["promote_if"],
        )

    workers = int(args.workers or cfg.get("num_workers", 1))
    results = []

    if workers <= 1:
        for (s, tf, h) in combos:
            r = train_one_combo(make_cfg(s, tf, h), db_url)
            print("[DONE]", r)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(train_one_combo, make_cfg(s, tf, h), db_url):(s,tf,h) for (s,tf,h) in combos}
            for fut in as_completed(futs):
                r = fut.result()
                print("[DONE]", r)
                results.append(r)

    # Resumen
    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") != "ok"]
    print(f"\n[RESUMEN] OK: {len(ok)}  |  Skipped: {len(skipped)}")
    if ok:
        m_auc = np.mean([r["auc"] for r in ok])
        print(f"  Media AUC (tail): {m_auc:.3f}")

if __name__ == "__main__":
    main()
