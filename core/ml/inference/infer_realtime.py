import os, pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from core.ml.utils.io import load_pickle
from core.ml.inference.postprocess import prob_to_side, strength_from_prob
from core.ml.training.registry import get_or_create_agent
from core.ml.datasets.builder import build_dataset, FEATURES

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

SYMBOL = os.getenv("ML_SYMBOL", "BTCUSDT")
TF     = os.getenv("ML_TF", "1m")
H      = os.getenv("ML_H", "1")
ARTIFACT = os.getenv("ML_ARTIFACT", f"artifacts/direction/{SYMBOL}_{TF}_H{H}_logreg.pkl")

def upsert_pred(version_id: int, symbol: str, tf: str, ts, horizon: str, payload: dict):
    q = text("""
    INSERT INTO trading.AgentPreds(agent_version_id,symbol,timeframe,timestamp,horizon,payload)
    VALUES (:vid,:s,:tf,:ts,:hz,:pl)
    ON CONFLICT (agent_version_id, symbol, timeframe, timestamp, horizon)
    DO UPDATE SET payload=EXCLUDED.payload
    """).bindparams(
        bindparam("pl", type_=JSONB)   # 👈 JSONB
    )
    with ENGINE.begin() as c:
        c.execute(q, {
            "vid": version_id, "s": symbol, "tf": tf, "ts": ts, "hz": horizon,
            "pl": payload or {}
        })

def upsert_signal(symbol: str, tf: str, ts, side: int, strength: float, sl=None, tp=None, meta=None):
    q = text("""
    INSERT INTO trading.AgentSignals(symbol,timeframe,timestamp,side,strength,sl,tp,meta)
    VALUES (:s,:tf,:ts,:sd,:st,:sl,:tp,:mt)
    ON CONFLICT (symbol, timeframe, timestamp)
    DO UPDATE SET side=EXCLUDED.side, strength=EXCLUDED.strength, sl=EXCLUDED.sl, tp=EXCLUDED.tp, meta=EXCLUDED.meta
    """).bindparams(
        bindparam("mt", type_=JSONB)   # 👈 JSONB
    )
    with ENGINE.begin() as c:
        c.execute(q, {
            "s": symbol, "tf": tf, "ts": ts, "sd": side, "st": strength,
            "sl": sl, "tp": tp,
            "mt": meta or {}
        })

def get_latest_version_id(agent_name: str):
    q = text("""
    SELECT v.id, v.artifact_uri
    FROM trading.AgentVersions v
    JOIN trading.Agents a ON a.id=v.agent_id
    WHERE a.name=:n
    ORDER BY v.id DESC
    LIMIT 1
    """)
    with ENGINE.begin() as c:
        row = c.execute(q, {"n":agent_name}).fetchone()
    return (row[0], row[1]) if row else (None, None)

def main():
    # Carga modelo
    agent_name = "DirectionLogReg"
    ver_id, artifact_uri = get_latest_version_id(agent_name)
    path = ARTIFACT if os.path.exists(ARTIFACT) else artifact_uri
    if not path or not os.path.exists(path):
        print("No encuentro artifact. Entrena primero train_direction.py")
        return
    model = load_pickle(path)

    # Dataset reciente
    df = build_dataset(SYMBOL, TF, use_snapshots=False)
    if df.empty:
        print("Sin datos.")
        return
    # Tomemos últimas 500 filas
    df = df.dropna().tail(500)
    X = df[FEATURES].astype(float)

    # Inferir y upsert
    proba = model.predict_proba(X)[:,1]
    for ts, p in zip(df["timestamp"], proba):
        upsert_pred(ver_id, SYMBOL, TF, ts, H, {"prob_up": float(p)})
        side = prob_to_side(p)
        strength = strength_from_prob(p)
        upsert_signal(SYMBOL, TF, ts, side, strength, sl=None, tp=None, meta={"direction_ver_id": ver_id})

    print(f"Escritas predicciones y señales para {SYMBOL}-{TF} (últimas {len(df)} barras).")

if __name__ == "__main__":
    main()
