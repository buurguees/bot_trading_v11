import os, yaml, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

FEATURES = [
 "rsi14","ema20","ema50","ema200","macd","macd_signal","macd_hist",
 "atr14","bb_mid","bb_upper","bb_lower","obv","supertrend","st_dir"
]

def _read_cfg(path="config/trading/symbols.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def fetch_base(symbol: str, tf: str) -> pd.DataFrame:
    sql = text(f"""
    SELECT h.symbol, h.timeframe, h.timestamp, h.open, h.high, h.low, h.close, h.volume,
           {', '.join('f.'+c for c in FEATURES)}
    FROM trading.historicaldata h
    JOIN trading.features f
      ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.timestamp=h.timestamp
    WHERE h.symbol=:s AND h.timeframe=:tf
    ORDER BY h.timestamp
    """)
    with ENGINE.begin() as c:
        rows = c.execute(sql, {"s": symbol, "tf": tf}).fetchall()
    if not rows:
        return pd.DataFrame()
    cols = ["symbol","timeframe","timestamp","open","high","low","close","volume"] + FEATURES
    return pd.DataFrame(rows, columns=cols)

def add_snapshots(df_base: pd.DataFrame, symbol: str, high_tf: str, suffix: str) -> pd.DataFrame:
    # Trae features del TF alto y las proyecta por merge_asof backward (LEFT) + ffill
    sql = text(f"""
    SELECT timestamp, {', '.join(FEATURES)}
    FROM trading.features
    WHERE symbol=:s AND timeframe=:tf
    ORDER BY timestamp
    """)
    with ENGINE.begin() as c:
        rows = c.execute(sql, {"s": symbol, "tf": high_tf}).fetchall()
    if not rows:
        return df_base
    high = pd.DataFrame(rows, columns=["timestamp"] + FEATURES)
    # Asegurar que timestamp sea datetime y esté ordenado
    high["timestamp"] = pd.to_datetime(high["timestamp"], utc=True)
    high = high.sort_values("timestamp")
    
    base = df_base.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base = base.sort_values("timestamp")
    
    # Renombrar columnas del snapshot antes del merge
    high_suffix = high.copy()
    for col in FEATURES:
        high_suffix = high_suffix.rename(columns={col: f"{col}_{suffix}"})
    
    merged = pd.merge_asof(
        base,
        high_suffix,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    # forward fill sólo columnas del snapshot
    snap_cols = [f"{c}_{suffix}" for c in FEATURES]
    merged[snap_cols] = merged[snap_cols].ffill()
    return merged

def build_dataset(symbol: str, tf_base: str, use_snapshots: bool = True) -> pd.DataFrame:
    df = fetch_base(symbol, tf_base)
    if df.empty:
        return df
    if use_snapshots:
        # Añade 15m, 1h, 4h, 1d si existen
        tf_map = {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
        for tfh, suf in tf_map.items():
            df = add_snapshots(df, symbol, tfh, suf)
    return df
