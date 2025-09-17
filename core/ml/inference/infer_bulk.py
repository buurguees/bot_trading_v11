import argparse, pickle, json
import numpy as np
import pandas as pd
from sqlalchemy import text
from core.data.database import get_engine

# columnas de features que usa el modelo (ajusta si añadiste más)
FEAT_COLS = [
    "rsi14","ema20","ema50","ema200",
    "macd","macd_signal","macd_hist",
    "atr14","bb_mid","bb_upper","bb_lower",
    "obv","supertrend","st_dir"
]

def parse_utc(s):
    ts = pd.Timestamp(s)
    return ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver-id", type=int, required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--from", dest="dt_from", required=True)
    ap.add_argument("--to",   dest="dt_to", required=True)
    args = ap.parse_args()

    ts_from = parse_utc(args.dt_from)
    ts_to   = parse_utc(args.dt_to)

    eng = get_engine()
    with eng.begin() as c:
        # 1) Traer versión (ruta del pickle)
        row = c.execute(text("""
            SELECT artifact_uri, params
            FROM trading.agentversions
            WHERE id = :vid
        """), {"vid": args.ver_id}).mappings().first()
        assert row, "agent_version_id no encontrado"

        art = row["artifact_uri"]
        with open(art, "rb") as f:
            model = pickle.load(f)

        # 2) Traer features del rango
        df = pd.read_sql(
            """
            SELECT timestamp, symbol, timeframe,
                   rsi14, ema20, ema50, ema200,
                   macd, macd_signal, macd_hist,
                   atr14, bb_mid, bb_upper, bb_lower,
                   obv, supertrend, st_dir
            FROM trading.features
            WHERE symbol = %(s)s AND timeframe = %(tf)s
              AND timestamp >= %(t0)s AND timestamp < %(t1)s
            ORDER BY timestamp ASC
            """,
            con=eng,
            params={"s": args.symbol, "tf": args.tf, "t0": ts_from, "t1": ts_to},
        )

        if df.empty:
            print("Sin features en el rango.")
            return

        # Normalizar timestamp a UTC consciente de zona
        if df["timestamp"].dtype.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Desempaquetar modelo si artefacto es un dict
        model_obj = model
        feat_names = None
        if isinstance(model_obj, dict):
            model_obj = model_obj.get("model") or model_obj.get("clf") or model_obj.get("estimator") or model_obj.get("sk_model")
            feat_names = model.get("feature_names") or model.get("features") or model.get("feat_cols")
        if feat_names is None:
            feat_names = FEAT_COLS

        # Asegurar columnas requeridas y orden
        for col in feat_names:
            if col not in df.columns:
                df[col] = 0.0
        X = df[list(feat_names)].astype(float).to_numpy()

        # Obtener probabilidad
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(X)[:, 1]
        elif hasattr(model_obj, "decision_function"):
            z = model_obj.decision_function(X)
            proba = 1.0 / (1.0 + np.exp(-z))
        else:
            y = model_obj.predict(X)
            proba = np.clip((np.array(y).astype(float) + 1.0) / 2.0, 0.0, 1.0) if set(np.unique(y)).issubset({-1, 1}) else np.clip(np.array(y).astype(float), 0.0, 1.0)

        df["prob_up"] = proba

        # 3) Escribir en AgentPreds (upsert)
        #    y señales “directional” simples (side=sign(prob-0.5))
        preds = df[["timestamp"]].copy()
        preds["agent_version_id"] = args.ver_id
        preds["symbol"] = args.symbol
        preds["timeframe"] = args.tf
        preds["horizon"] = "1"
        preds["prob_up"] = df["prob_up"].astype(float)
        preds["payload"] = [{"prob_up": float(p)} for p in preds["prob_up"]]

        # escribir por lotes
        rows = [(
            int(args.ver_id), args.symbol, args.tf,
            ts.to_pydatetime(), "1", {"prob_up": float(p)}
        ) for ts, p in zip(preds["timestamp"], preds["prob_up"])]

        c.exec_driver_sql(
            """
            INSERT INTO trading.agentpreds
            (agent_version_id, symbol, timeframe, timestamp, horizon, payload)
            VALUES (%(vid)s, %(s)s, %(tf)s, %(ts)s, %(hz)s, %(pl)s::jsonb)
            ON CONFLICT (agent_version_id, symbol, timeframe, timestamp, horizon)
            DO UPDATE SET payload = EXCLUDED.payload
            """,
            [{"vid": r[0], "s": r[1], "tf": r[2], "ts": r[3], "hz": r[4], "pl": json.dumps(r[5])} for r in rows],
        )

        # señales direccionales
        sig = df[["timestamp","prob_up"]].copy()
        sig["symbol"] = args.symbol
        sig["timeframe"] = args.tf
        sig["side"] = np.where(sig["prob_up"] >= 0.5, 1, -1).astype(int)
        sig["strength"] = (sig["prob_up"] - 0.5).abs()
        sig["meta"] = [{"direction_ver_id": int(args.ver_id)} for _ in range(len(sig))]

        c.exec_driver_sql(
            """
            INSERT INTO trading.agentsignals (symbol, timeframe, timestamp, side, strength, meta)
            VALUES (%(s)s, %(tf)s, %(ts)s, %(sd)s, %(st)s, %(mt)s::jsonb)
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET side = EXCLUDED.side,
                          strength = EXCLUDED.strength,
                          meta = EXCLUDED.meta
            """,
            [{"s": args.symbol, "tf": args.tf, "ts": t.to_pydatetime(),
              "sd": int(sd), "st": float(st), "mt": json.dumps(m)}
             for t, sd, st, m in zip(sig["timestamp"], sig["side"], sig["strength"], sig["meta"])],
        )

        print(f"Escritas {len(rows)} predicciones y {len(sig)} señales para {args.symbol}-{args.tf} [{ts_from}..{ts_to})")

if __name__ == "__main__":
    main()
