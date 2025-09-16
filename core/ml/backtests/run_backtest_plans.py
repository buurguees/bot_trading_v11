import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

SYMBOL = os.getenv("BT_SYMBOL","BTCUSDT")
TF     = os.getenv("BT_TF","1m")
FEE_BPS = float(os.getenv("FEE_BPS", "8"))  # taker + idas/vueltas

def load_plans(symbol: str, tf: str):
    q = text("""
    SELECT
      id,
      bar_ts,
      side::int        AS side,
      entry_px::float8 AS entry_px,
      sl_px::float8    AS sl_px,
      tp_px::float8    AS tp_px,
      qty::float8      AS qty,
      leverage::float8 AS leverage
    FROM trading.TradePlans
    WHERE symbol = :s AND timeframe = :tf AND status = 'planned'
    ORDER BY bar_ts
    """)
    with ENGINE.begin() as c:
        rows = c.execute(q, {"s":symbol, "tf":tf}).fetchall()
    cols = ["id","bar_ts","side","entry_px","sl_px","tp_px","qty","leverage"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if not df.empty:
        df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    return df

def load_path(symbol: str, tf: str, start_ts):
    q = text("""
    SELECT 
      timestamp, 
      open::float8  AS open, 
      high::float8  AS high, 
      low::float8   AS low, 
      close::float8 AS close
    FROM trading.HistoricalData
    WHERE symbol=:s AND timeframe=:tf AND timestamp > :ts
    ORDER BY timestamp ASC
    """)
    with ENGINE.begin() as c:
        rows = c.execute(q, {"s":symbol, "tf":tf, "ts":start_ts}).fetchall()
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close"]) if rows else pd.DataFrame(columns=["timestamp","open","high","low","close"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def simulate_trade(row, path: pd.DataFrame, fee_bps: float):
    side = int(row.side)
    entry = row.entry_px; sl = row.sl_px; tp = row.tp_px; qty = row.qty
    fee = fee_bps / 10000.0

    exit_ts = None
    exit_px = path["close"].iloc[-1] if not path.empty else entry

    # recorre barras hasta tocar SL o TP
    for _, b in path.iterrows():
        lo = b.low; hi = b.high
        # conservador: si toca ambos en la misma barra, prioriza SL
        if side == 1:
            if lo <= sl:
                exit_ts = b.timestamp; exit_px = sl; break
            if hi >= tp:
                exit_ts = b.timestamp; exit_px = tp; break
        else:  # short
            if hi >= sl:
                exit_ts = b.timestamp; exit_px = sl; break
            if lo <= tp:
                exit_ts = b.timestamp; exit_px = tp; break

    # PnL en cash con comisiones de ida y vuelta ~ (entry + exit)
    pnl_gross = side * (exit_px - entry) * qty
    fees = (entry + exit_px) * qty * fee
    pnl_net = pnl_gross - fees
    r_multiple = (side * (exit_px - entry)) / (abs(entry - sl) + 1e-12)

    return {
        "plan_id": int(row.id),
        "entry_px": entry, "exit_px": exit_px, "side": side,
        "qty": qty, "exit_ts": exit_ts, "pnl_gross": pnl_gross, "fees": fees,
        "pnl_net": pnl_net, "r": r_multiple
    }

def main():
    plans = load_plans(SYMBOL, TF)
    if plans.empty:
        print("No hay TradePlans 'planned'. Ejecuta infer_realtime para generarlos.")
        return

    results = []
    equity = float(os.getenv("BT_EQUITY", "10000"))
    for _, r in plans.iterrows():
        path = load_path(SYMBOL, TF, r.bar_ts)
        if path.empty:
            continue
        res = simulate_trade(r, path, FEE_BPS)
        equity += res["pnl_net"]
        res["equity"] = equity
        results.append(res)

    if not results:
        print("No se pudo simular ningún plan.")
        return

    df = pd.DataFrame(results)
    print("Trades simulados:", len(df))
    print("P&L neto total:", df["pnl_net"].sum())
    print("Equity final:", df["equity"].iloc[-1])
    print(df.tail(10)[["plan_id","exit_ts","pnl_net","r","equity"]])

if __name__ == "__main__":
    main()
