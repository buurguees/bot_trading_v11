import argparse, pickle, json
import time
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

def calculate_features_quality(X, feature_names):
    """Calcular calidad de las features utilizadas"""
    if X.size == 0:
        return np.array([0.0])
    
    # Calcular métricas de calidad
    n_samples, n_features = X.shape
    
    # 1. Completitud: % de valores no nulos
    completeness = np.sum(~np.isnan(X), axis=1) / n_features
    
    # 2. Estabilidad: desviación estándar de las features (menor = más estable)
    feature_stability = 1.0 / (1.0 + np.std(X, axis=1))
    
    # 3. Rango: normalización del rango de valores
    feature_range = np.ptp(X, axis=1) / (np.ptp(X) + 1e-8)
    
    # 4. Outliers: % de valores dentro de 2 desviaciones estándar
    mean_vals = np.mean(X, axis=1, keepdims=True)
    std_vals = np.std(X, axis=1, keepdims=True)
    within_2std = np.sum(np.abs(X - mean_vals) <= 2 * std_vals, axis=1) / n_features
    
    # Combinar métricas (promedio ponderado)
    quality_score = (
        0.3 * completeness +
        0.3 * feature_stability +
        0.2 * feature_range +
        0.2 * within_2std
    )
    
    return np.clip(quality_score, 0.0, 1.0)

def calculate_market_volatility(df):
    """Calcular volatilidad del mercado basada en datos OHLCV"""
    if len(df) < 2:
        return np.array([0.5])  # Valor neutral
    
    # Calcular volatilidad usando ATR simplificado
    high = df['high'].values if 'high' in df.columns else df['close'].values
    low = df['low'].values if 'low' in df.columns else df['close'].values
    close = df['close'].values
    
    # True Range simplificado
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    
    # ATR de 14 períodos (simplificado)
    atr = pd.Series(tr).rolling(window=min(14, len(tr))).mean().fillna(tr[0]).values
    
    # Normalizar volatilidad (0-1)
    price_range = np.max(close) - np.min(close)
    volatility = atr / (price_range + 1e-8)
    
    return np.clip(volatility, 0.0, 1.0)

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

        # Obtener probabilidad y métricas adicionales
        start_time = time.time()
        
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(X)[:, 1]
            # Calcular confianza del modelo basada en la diferencia entre clases
            if proba.shape[1] > 1:
                class_probs = model_obj.predict_proba(X)
                model_confidence = np.max(class_probs, axis=1) - np.min(class_probs, axis=1)
            else:
                model_confidence = np.abs(proba - 0.5) * 2.0
        elif hasattr(model_obj, "decision_function"):
            z = model_obj.decision_function(X)
            proba = 1.0 / (1.0 + np.exp(-z))
            # Para decision_function, la confianza es la magnitud de la decisión
            model_confidence = np.abs(z) / (np.abs(z).max() + 1e-8)
        else:
            y = model_obj.predict(X)
            proba = np.clip((np.array(y).astype(float) + 1.0) / 2.0, 0.0, 1.0) if set(np.unique(y)).issubset({-1, 1}) else np.clip(np.array(y).astype(float), 0.0, 1.0)
            model_confidence = np.abs(proba - 0.5) * 2.0

        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Calcular métricas de calidad de features
        features_quality = calculate_features_quality(X, feat_names)
        
        # Calcular volatilidad del mercado (simplificado)
        market_volatility = calculate_market_volatility(df)
        
        df["prob_up"] = proba
        df["model_confidence"] = model_confidence
        df["features_quality"] = features_quality
        df["market_volatility"] = market_volatility
        df["processing_time_ms"] = processing_time

        # 3) Escribir en AgentPreds (upsert)
        #    y señales “directional” simples (side=sign(prob-0.5))
        preds = df[["timestamp"]].copy()
        preds["agent_version_id"] = args.ver_id
        preds["symbol"] = args.symbol
        preds["timeframe"] = args.tf
        preds["horizon"] = 1
        preds["prob_up"] = df["prob_up"].astype(float)
        
        # Crear payload detallado con toda la información necesaria
        detailed_payloads = []
        for i, (ts, prob, conf, feat_qual, vol, proc_time) in enumerate(zip(
            preds["timestamp"], 
            df["prob_up"], 
            df["model_confidence"], 
            df["features_quality"], 
            df["market_volatility"],
            df["processing_time_ms"]
        )):
            payload = {
                "prob_up": float(prob),
                "model_confidence": float(conf),
                "features_quality": float(feat_qual),
                "market_volatility": float(vol),
                "processing_time_ms": float(proc_time),
                "timestamp": ts.isoformat(),
                "features_used": feat_names,
                "features_count": len(feat_names),
                "prediction_id": f"{args.ver_id}_{ts.strftime('%Y%m%d_%H%M%S')}_{i}",
                "model_version": args.ver_id,
                "symbol": args.symbol,
                "timeframe": args.tf
            }
            detailed_payloads.append(payload)
        
        preds["payload"] = detailed_payloads

        # escribir por lotes con payload detallado
        rows = [(
            int(args.ver_id), args.symbol, args.tf,
            ts.to_pydatetime(), 1, payload
        ) for ts, payload in zip(preds["timestamp"], preds["payload"])]

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

        print(f"Escritas {len(rows)} predicciones para {args.symbol}-{args.tf} [{ts_from}..{ts_to})")
        print("Nota: Las señales se generarán usando signal_processor.py")

if __name__ == "__main__":
    main()
