from math import tanh
from sqlalchemy import text
import json

def _bucket(v, cuts):
    for i, c in enumerate(cuts):
        if v < c:
            return i
    return len(cuts)

def build_signature(row):
    # row: dict con features/razones de la barra (lo que guardas en reason)
    rsi = row.get("rsi", 50.0)
    ema_state = row.get("ema_state", "mix")  # ya puedes derivarlo en features_updater
    macd = row.get("macd", 0.0)
    st = row.get("supertrend_dir", 1)
    trend1h = row.get("trend_1h", 0)
    sig = f"r{_bucket(rsi,[30,50,70])}|ema:{ema_state}|macd:{'+' if macd>=0 else '-'}|st:{st}|t1h:{trend1h}"
    feats = {
        "rsi_bucket": _bucket(rsi,[30,50,70]),
        "ema_state": ema_state,
        "macd_sign": 1 if macd>=0 else -1,
        "supertrend_dir": st,
        "trend_1h": trend1h
    }
    return sig, feats

def update_memory(conn, symbol, tf, trades):
    # trades: lista de dicts con keys ['entry_ts','pnl','side','leverage','reason', ...]
    mem_upsert = text("""
      INSERT INTO trading.strategy_memory(symbol, timeframe, signature, features,
             n_trades, win_rate, avg_pnl, sharpe, avg_hold_bars, last_updated)
      VALUES (:s, :tf, :sig, CAST(:feats AS JSONB), 1,
              CASE WHEN :pnl>0 THEN 1.0 ELSE 0.0 END, :pnl, NULL, :hold, now())
      ON CONFLICT (symbol, timeframe, signature) DO UPDATE SET
        n_trades = trading.strategy_memory.n_trades + 1,
        win_rate = ((trading.strategy_memory.win_rate * trading.strategy_memory.n_trades)
                    + CASE WHEN EXCLUDED.avg_pnl>0 THEN 1.0 ELSE 0.0 END)
                   / (trading.strategy_memory.n_trades + 1),
        avg_pnl  = ((trading.strategy_memory.avg_pnl * trading.strategy_memory.n_trades)
                    + EXCLUDED.avg_pnl)
                   / (trading.strategy_memory.n_trades + 1),
        last_updated = now()
      RETURNING id;
    """)
    sample_ins = text("""
      INSERT INTO trading.strategy_samples(memory_id, entry_ts, side, leverage, pnl, reason)
      VALUES (:mid, :et, :sd, :lv, :pl, CAST(:rs AS JSONB))
    """)
    for t in trades:
        sig, feats = build_signature(t.get("reason", {}))
        hold = t.get("hold_bars", 0)
        mid = conn.execute(mem_upsert, {
            "s": symbol, "tf": tf, "sig": sig, "feats": json.dumps(feats),
            "pnl": t["pnl"], "hold": hold
        }).scalar_one()
        conn.execute(sample_ins, {
            "mid": mid, "et": t["entry_ts"], "sd": t["side"],
            "lv": t["leverage"], "pl": t["pnl"],
            "rs": json.dumps(t.get("reason", {}))
        })


