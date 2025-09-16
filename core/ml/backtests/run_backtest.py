import os, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from core.ml.backtests.analyzers import simple_pnl

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

SYMBOL = os.getenv("BT_SYMBOL","BTCUSDT")
TF     = os.getenv("BT_TF","1m")

def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura timestamp = datetime64[ns, UTC], ordenado y sin duplicados."""
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out

def load_signals(symbol: str, tf: str) -> pd.DataFrame:
    q = text("""
    SELECT timestamp, side
    FROM trading.AgentSignals
    WHERE symbol=:s AND timeframe=:tf
    ORDER BY timestamp
    """)
    with ENGINE.begin() as c:
        rows = c.execute(q, {"s":symbol, "tf":tf}).fetchall()
    df = pd.DataFrame(rows, columns=["timestamp","side"]) if rows else pd.DataFrame(columns=["timestamp","side"])
    df = _normalize_ts(df)
    if not df.empty:
        df["side"] = pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(float)
    return df

def load_prices(symbol: str, tf: str) -> pd.DataFrame:
    q = text("""
    SELECT timestamp, close
    FROM trading.HistoricalData
    WHERE symbol=:s AND timeframe=:tf
    ORDER BY timestamp
    """)
    with ENGINE.begin() as c:
        rows = c.execute(q, {"s":symbol, "tf":tf}).fetchall()
    df = pd.DataFrame(rows, columns=["timestamp","close"]) if rows else pd.DataFrame(columns=["timestamp","close"])
    df = _normalize_ts(df)
    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df

def main():
    sig = load_signals(SYMBOL, TF)
    px  = load_prices(SYMBOL, TF)
    if sig.empty or px.empty:
        print("Faltan señales o precios.")
        return
    out = simple_pnl(sig, px)  # ahora con índices alineados
    print("Equity final:", float(out["equity"].iloc[-1]))
    print(out.tail(10))

if __name__ == "__main__":
    main()
