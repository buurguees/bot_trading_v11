"""
core/ml/feature_engineer.py
---------------------------------
Calcula features técnicos + SMC y escribe en market.features.

- Incremental con solapamiento (LOOKBACK_BARS).
- Solo procesa HASTA LA ÚLTIMA VELA CERRADA (evita la vela en curso).
- Indicadores: RSI, MACD, ATR, EMA20/50/200, OBV, SuperTrend(10,3).
- SMC flags (baseline) para FVG, swings, BOS/CHOCH, OB simples.

Uso puntual (batch):  python -m core.ml.feature_engineer
Para modo continuo usa: core/ml/feature_updater.py
"""

from __future__ import annotations

import os
import math
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import yaml

# ---------- Config y logger ----------
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/trading_db")
CONFIG_PATH = os.path.join("config", "market", "symbols.yaml")

logger = logging.getLogger("FeatureEngineer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

# ---------- Parámetros ----------
RSI_LEN = 14
ATR_LEN = 14
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9
EMA20 = 20
EMA50 = 50
EMA200 = 200
ST_ATR = 10
ST_MULT = 3.0

LOOKBACK_BARS = 250

TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# ---------- Utilidades ----------
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

def load_symbols_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_max_feature_ts(engine, symbol: str, timeframe: str) -> Optional[datetime]:
    sql = text("""
        SELECT MAX(ts) AS max_ts
        FROM market.features
        WHERE symbol = :symbol AND timeframe = :tf
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"symbol": symbol, "tf": timeframe}).mappings().first()
        return row["max_ts"] if row and row["max_ts"] else None

def fetch_ohlcv_df(engine, symbol: str, timeframe: str, since_dt: Optional[datetime]) -> pd.DataFrame:
    # Por política actual: limitar a los últimos 2 años cuando no hay punto de partida
    if since_dt is None:
        since_dt = datetime.now(tz=timezone.utc) - timedelta(days=int(2 * 365))
    sql = text("""
        SELECT ts, open, high, low, close, volume
        FROM market.historical_data
        WHERE symbol = :symbol AND timeframe = :tf AND ts >= :since
        ORDER BY ts ASC
    """)
    params = {"symbol": symbol, "tf": timeframe, "since": since_dt}

    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["ts"])
    return df

def upsert_features(engine, symbol: str, timeframe: str, df_feats: pd.DataFrame, batch: int = 2000) -> int:
    if df_feats.empty:
        return 0

    from sqlalchemy import text as _text
    sql = _text("""
        INSERT INTO market.features
        (symbol, timeframe, ts,
         rsi_14, macd, macd_signal, macd_hist,
         ema_20, ema_50, ema_200, atr_14, obv,
         supertrend, st_direction, smc_flags, extra)
        VALUES
        (:symbol, :tf, :ts,
         :rsi_14, :macd, :macd_signal, :macd_hist,
         :ema_20, :ema_50, :ema_200, :atr_14, :obv,
         :supertrend, :st_direction, :smc_flags, :extra)
        ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
          rsi_14      = EXCLUDED.rsi_14,
          macd        = EXCLUDED.macd,
          macd_signal = EXCLUDED.macd_signal,
          macd_hist   = EXCLUDED.macd_hist,
          ema_20      = EXCLUDED.ema_20,
          ema_50      = EXCLUDED.ema_50,
          ema_200     = EXCLUDED.ema_200,
          atr_14      = EXCLUDED.atr_14,
          obv         = EXCLUDED.obv,
          supertrend  = EXCLUDED.supertrend,
          st_direction= EXCLUDED.st_direction,
          smc_flags   = EXCLUDED.smc_flags,
          extra       = EXCLUDED.extra
    """)

    total = 0
    with engine.begin() as conn:
        records = []
        for _, r in df_feats.iterrows():
            records.append({
                "symbol": symbol,
                "tf": timeframe,
                "ts": r.ts.to_pydatetime().replace(tzinfo=timezone.utc),
                "rsi_14": _safe_num(r.rsi_14),
                "macd": _safe_num(r.macd),
                "macd_signal": _safe_num(r.macd_signal),
                "macd_hist": _safe_num(r.macd_hist),
                "ema_20": _safe_num(r.ema_20),
                "ema_50": _safe_num(r.ema_50),
                "ema_200": _safe_num(r.ema_200),
                "atr_14": _safe_num(r.atr_14),
                "obv": _safe_num(r.obv),
                "supertrend": _safe_num(r.supertrend),
                "st_direction": int(r.st_direction) if not pd.isna(r.st_direction) else None,
                "smc_flags": json.dumps(r.smc_flags) if isinstance(r.smc_flags, dict) else None,
                "extra": json.dumps(r.extra) if isinstance(r.extra, dict) else None,
            })
            if len(records) >= batch:
                conn.execute(sql, records)
                total += len(records)
                records = []
        if records:
            conn.execute(sql, records)
            total += len(records)
    logger.info(f"[{symbol}][{timeframe}] upsert features: {total}")
    return total

def _safe_num(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return float(x)

# ---------- Indicadores ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi(series: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def macd(series: pd.Series, fast: int = EMA_FAST, slow: int = EMA_SLOW, signal: int = MACD_SIGNAL):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def supertrend(df: pd.DataFrame, atr_len: int = ST_ATR, mult: float = ST_MULT) -> Tuple[pd.Series, pd.Series]:
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_ = atr(df, atr_len)

    upperband = hl2 + mult * atr_
    lowerband = hl2 - mult * atr_

    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=float)

    st.iloc[0] = hl2.iloc[0]
    dir_.iloc[0] = 1.0

    for i in range(1, len(df)):
        prev_st = st.iloc[i-1]
        prev_dir = dir_.iloc[i-1]

        fub = upperband.iloc[i]
        flb = lowerband.iloc[i]

        if df["close"].iloc[i] > (prev_st if prev_dir < 0 else flb):
            dir_.iloc[i] = 1.0
        elif df["close"].iloc[i] < (prev_st if prev_dir > 0 else fub):
            dir_.iloc[i] = -1.0
        else:
            dir_.iloc[i] = prev_dir

        st.iloc[i] = flb if dir_.iloc[i] > 0 else fub

    return st, dir_

def obv(df: pd.DataFrame) -> pd.Series:
    change = df["close"].diff().fillna(0)
    sign = np.sign(change)
    return (sign * df["volume"]).fillna(0).cumsum()

# ---------- SMC flags (baseline) ----------
def smc_flags_basic(df: pd.DataFrame) -> List[Dict]:
    n = len(df)
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values

    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)
    for i in range(2, n-2):
        swing_high[i] = hi[i] > hi[i-1] and hi[i] > hi[i-2] and hi[i] > hi[i+1] and hi[i] > hi[i+2]
        swing_low[i]  = lo[i] < lo[i-1] and lo[i] < lo[i-2] and lo[i] < lo[i+1] and lo[i] < lo[i+2]

    fvg_up = np.zeros(n, dtype=bool)
    fvg_dn = np.zeros(n, dtype=bool)
    for i in range(2, n):
        fvg_up[i] = lo[i] > hi[i-2]
        fvg_dn[i] = hi[i] < lo[i-2]

    bos = np.zeros(n, dtype=bool)
    choch = np.zeros(n, dtype=bool)
    last_sh = None
    last_sl = None
    for i in range(n):
        if swing_high[i]:
            last_sh = i
        if swing_low[i]:
            last_sl = i
        if last_sh is not None and cl[i] > hi[last_sh]:
            bos[i] = True
        if last_sl is not None and cl[i] < lo[last_sl]:
            bos[i] = True
        if i >= 1:
            if bos[i] and not bos[i-1]:
                choch[i] = True

    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if bos[i] and cl[i] > cl[i-1]:
            j = i-1
            while j >= 0 and not (cl[j] < df["open"].iloc[j]):
                j -= 1
            if j >= 0:
                ob_bear[j] = True
        if bos[i] and cl[i] < cl[i-1]:
            j = i-1
            while j >= 0 and not (cl[j] > df["open"].iloc[j]):
                j -= 1
            if j >= 0:
                ob_bull[j] = True

    flags = []
    for i in range(n):
        flags.append({
            "fvg_up": bool(fvg_up[i]),
            "fvg_dn": bool(fvg_dn[i]),
            "swing_high": bool(swing_high[i]),
            "swing_low": bool(swing_low[i]),
            "bos": bool(bos[i]),
            "choch": bool(choch[i]),
            "ob_bull": bool(ob_bull[i]),
            "ob_bear": bool(ob_bear[i]),
        })
    return flags

# ---------- Helpers de tiempo ----------
def last_closed_ts_dt(timeframe: str) -> datetime:
    """Devuelve el datetime (UTC) de la última vela COMPLETAMENTE CERRADA para el TF dado."""
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    tf_ms = TF_MS.get(timeframe, 60_000)
    last_closed_ms = (now_ms // tf_ms) * tf_ms - tf_ms
    return datetime.fromtimestamp(last_closed_ms / 1000, tz=timezone.utc)

# ---------- Pipeline principal ----------
def compute_features_for(symbol: str, timeframe: str, engine) -> int:
    """
    Computa y upserta features para un (symbol,timeframe).
    - Procesa hasta la última vela CERRADA.
    - Incremental con solapamiento LOOKBACK_BARS.
    """
    # 1) Determina desde cuándo leer
    last_feat_ts = get_max_feature_ts(engine, symbol, timeframe)
    since_dt = None
    if last_feat_ts:
        tf_ms = TF_MS.get(timeframe, 60_000)
        since_ms = int(last_feat_ts.timestamp() * 1000) - LOOKBACK_BARS * tf_ms
        since_dt = datetime.fromtimestamp(max(0, since_ms)/1000, tz=timezone.utc)

    # 2) Carga OHLCV
    df = fetch_ohlcv_df(engine, symbol, timeframe, since_dt)
    if df.empty:
        logger.info(f"[{symbol}][{timeframe}] sin OHLCV para procesar.")
        return 0

    # 3) Filtra solo HASTA la última vela cerrada
    lc = last_closed_ts_dt(timeframe)
    df = df[df["ts"] <= lc]
    if df.empty:
        logger.info(f"[{symbol}][{timeframe}] no hay velas cerradas nuevas.")
        return 0

    # 4) Calcula indicadores - usar .loc para evitar SettingWithCopyWarning
    df = df.copy()  # Crear copia explícita para evitar warnings
    df.loc[:, "rsi_14"] = rsi(df["close"], RSI_LEN)
    macd_line, signal_line, hist = macd(df["close"], EMA_FAST, EMA_SLOW, MACD_SIGNAL)
    df.loc[:, "macd"] = macd_line
    df.loc[:, "macd_signal"] = signal_line
    df.loc[:, "macd_hist"] = hist

    df.loc[:, "ema_20"] = ema(df["close"], EMA20)
    df.loc[:, "ema_50"] = ema(df["close"], EMA50)
    df.loc[:, "ema_200"] = ema(df["close"], EMA200)

    df.loc[:, "atr_14"] = atr(df, ATR_LEN)
    df.loc[:, "obv"] = obv(df)

    st_line, st_dir = supertrend(df, ST_ATR, ST_MULT)
    df.loc[:, "supertrend"] = st_line
    df.loc[:, "st_direction"] = st_dir

    # 5) Flags SMC + extra
    flags = smc_flags_basic(df)
    df.loc[:, "smc_flags"] = flags
    df.loc[:, "extra"] = [{} for _ in range(len(df))]

    # 6) Si venimos de incremental, corta al rango nuevo (mayor que last_feat_ts)
    if last_feat_ts:
        df = df[df["ts"] > last_feat_ts]
    if df.empty:
        logger.info(f"[{symbol}][{timeframe}] nada nuevo que upsertar.")
        return 0

    # 7) Upsert
    cols = [
        "ts","rsi_14","macd","macd_signal","macd_hist",
        "ema_20","ema_50","ema_200","atr_14","obv",
        "supertrend","st_direction","smc_flags","extra"
    ]
    df_out = df[cols].copy()
    return upsert_features(engine, symbol, timeframe, df_out)

def run() -> None:
    cfg = load_symbols_config(CONFIG_PATH)
    engine = get_engine()
    total_rows = 0

    for s in cfg["symbols"]:
        symbol = s["id"]
        tfs = s.get("timeframes", ["1m","5m","15m","1h","4h","1d"])
        for tf in tfs:
            try:
                logger.info(f"==> Features {symbol} {tf}")
                total_rows += compute_features_for(symbol, tf, engine)
            except Exception as e:
                logger.exception(f"Error en features {symbol} {tf}: {e}")

    logger.info(f"TOTAL filas features upsertadas: {total_rows}")

if __name__ == "__main__":
    run()
