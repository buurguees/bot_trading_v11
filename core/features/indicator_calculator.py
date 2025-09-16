import os, math, yaml
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv("config/.env")
DB_URL = os.getenv("DB_URL")
ENGINE = create_engine(DB_URL, pool_pre_ping=True)
SYMBOLS_YAML = "config/trading/symbols.yaml"

# Ventanas
RSI_N = 14
EMA_FAST, EMA_SLOW, EMA_LONG = 12, 26, 200
ATR_N = 14
BB_N, BB_K = 20, 2.0
WARMUP = max(EMA_LONG, BB_N) + 50  # barras extra para estabilizar

def read_symbols_and_tfs(path=SYMBOLS_YAML):
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    syms = []
    for s, meta in (cfg.get("symbols") or {}).items():
        syms.append((meta.get("ccxt_symbol") or s, meta.get("timeframes", default_tfs)))
    return syms

def last_feat_ts(symbol, tf):
    q = text("""
        SELECT MAX(timestamp) FROM trading.Features
        WHERE symbol=:s AND timeframe=:tf
    """)
    with ENGINE.begin() as c:
        return c.execute(q, {"s": symbol, "tf": tf}).scalar()

def fetch_candles(symbol, tf, since_ts):
    # si since_ts es None, traer un bloque razonable
    params = {"s": symbol, "tf": tf}
    if since_ts is None:
        q = text(f"""
            SELECT timestamp, open, high, low, close, volume
            FROM trading.HistoricalData
            WHERE symbol=:s AND timeframe=:tf
            ORDER BY timestamp DESC
            LIMIT {5000 + WARMUP}
        """)
    else:
        q = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM trading.HistoricalData
            WHERE symbol=:s AND timeframe=:tf
              AND timestamp >= :since
            ORDER BY timestamp ASC
        """)
        params["since"] = since_ts - timedelta(seconds=bars_to_seconds(tf, WARMUP))
    with ENGINE.begin() as c:
        rows = c.execute(q, params).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 👇 Coerción a float: evita Decimals/objeto que rompen los indicadores
    num_cols = ["open","high","low","close","volume"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    
    return df

def bars_to_seconds(tf, n):
    m = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":14400, "1d":86400}
    return m.get(tf, 60)*n

# ---------- Indicadores ----------
def ema(series, n): return series.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k*std
    lower = mid - k*std
    return mid, upper, lower

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def obv(close, volume):
    # close/volume deben venir ya como float por la coerción anterior
    dir_ = np.sign(close.diff())           # ndarray de floats
    dir_ = pd.Series(dir_, index=close.index).fillna(0.0)
    return (volume * dir_).cumsum()

def supertrend(df, n=10, mult=3.0):
    # Implementación ligera (no 100% idéntica a todas las libs)
    atr_v = atr(df, n)
    hl2 = (df["high"] + df["low"])/2.0
    upper = hl2 + mult * atr_v
    lower = hl2 - mult * atr_v
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=float)

    st.iloc[0] = upper.iloc[0]
    dir_.iloc[0] = 1.0
    for i in range(1, len(df)):
        if df["close"].iloc[i] > st.iloc[i-1]:
            dir_.iloc[i] = 1.0
        elif df["close"].iloc[i] < st.iloc[i-1]:
            dir_.iloc[i] = -1.0
        else:
            dir_.iloc[i] = dir_.iloc[i-1]
        st.iloc[i] = upper.iloc[i] if dir_.iloc[i] > 0 else lower.iloc[i]
    return st, dir_

# ---------- Persistencia ----------
UPSERT = text("""
INSERT INTO trading.Features(
  symbol, timeframe, timestamp,
  rsi14, ema20, ema50, ema200, macd, macd_signal, macd_hist,
  atr14, bb_mid, bb_upper, bb_lower, obv, supertrend, st_dir
) VALUES (
  :symbol, :tf, :ts,
  :rsi14, :ema20, :ema50, :ema200, :macd, :macd_signal, :macd_hist,
  :atr14, :bb_mid, :bb_upper, :bb_lower, :obv, :supertrend, :st_dir
)
ON CONFLICT ON CONSTRAINT unique_features_key DO UPDATE SET
  rsi14 = EXCLUDED.rsi14,
  ema20 = EXCLUDED.ema20,
  ema50 = EXCLUDED.ema50,
  ema200 = EXCLUDED.ema200,
  macd = EXCLUDED.macd,
  macd_signal = EXCLUDED.macd_signal,
  macd_hist = EXCLUDED.macd_hist,
  atr14 = EXCLUDED.atr14,
  bb_mid = EXCLUDED.bb_mid,
  bb_upper = EXCLUDED.bb_upper,
  bb_lower = EXCLUDED.bb_lower,
  obv = EXCLUDED.obv,
  supertrend = EXCLUDED.supertrend,
  st_dir = EXCLUDED.st_dir,
  updated_at = NOW();
""")

def compute_and_save(symbol, tf):
    last_ts = last_feat_ts(symbol, tf)
    df = fetch_candles(symbol, tf, last_ts)
    if df.empty: return 0

    # Indicadores
    df["rsi14"] = rsi(df["close"], RSI_N)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"]= ema(df["close"], 200)
    macd_line, macd_sig, macd_hist_v = macd(df["close"], EMA_FAST, EMA_SLOW, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, macd_sig, macd_hist_v
    df["atr14"] = atr(df, ATR_N)
    bb_mid, bb_up, bb_lo = bollinger(df["close"], BB_N, BB_K)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bb_mid, bb_up, bb_lo
    df["obv"] = obv(df["close"], df["volume"])
    st, st_dir = supertrend(df, 10, 3.0)
    df["supertrend"], df["st_dir"] = st, st_dir

    # Si veníamos de last_ts, recorta warmup sobrante
    if last_ts is not None:
        df = df[df["timestamp"] > last_ts]

    df = df.dropna().copy()
    if df.empty: return 0

    rows = df[[
        "timestamp","rsi14","ema20","ema50","ema200",
        "macd","macd_signal","macd_hist","atr14",
        "bb_mid","bb_upper","bb_lower","obv","supertrend","st_dir"
    ]]

    inserted = 0
    with ENGINE.begin() as c:
        for r in rows.itertuples(index=False):
            c.execute(UPSERT, {
                "symbol": symbol, "tf": tf, "ts": r.timestamp,
                "rsi14": r.rsi14, "ema20": r.ema20, "ema50": r.ema50, "ema200": r.ema200,
                "macd": r.macd, "macd_signal": r.macd_signal, "macd_hist": r.macd_hist,
                "atr14": r.atr14, "bb_mid": r.bb_mid, "bb_upper": r.bb_upper, "bb_lower": r.bb_lower,
                "obv": r.obv, "supertrend": r.supertrend, "st_dir": int(r.st_dir) if not pd.isna(r.st_dir) else None
            })
            inserted += 1
    return inserted

def main():
    total = 0
    for symbol, tfs in read_symbols_and_tfs():
        for tf in tfs:
            ins = compute_and_save(symbol, tf)
            print(f"[{symbol} {tf}] guardadas {ins} filas de features.")
            total += ins
    print(f"Total features upserted: {total}")

if __name__ == "__main__":
    main()
