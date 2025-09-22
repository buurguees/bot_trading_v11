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

def update_memory(conn, symbol, tf, trades, mode: str = "backtest"):
    # trades: lista de dicts con keys ['entry_ts','pnl','side','leverage','reason', ...]
    
    # Agrupar trades por firma
    signature_groups = {}
    for t in trades:
        sig, feats = build_signature(t.get("reason", {}))
        if sig not in signature_groups:
            signature_groups[sig] = {
                "features": feats,
                "trades": []
            }
        signature_groups[sig]["trades"].append(t)
    
    mem_upsert = text("""
      INSERT INTO trading.strategy_memory(symbol, timeframe, signature, features,
             n_trades, win_rate, avg_pnl, avg_leverage, sharpe, avg_hold_bars, last_updated, mode)
      VALUES (:s, :tf, :sig, CAST(:feats AS JSONB), :n_trades,
              :win_rate, :avg_pnl, :avg_leverage, NULL, :avg_hold, now(), :mode)
      ON CONFLICT (symbol, timeframe, signature, mode) DO UPDATE SET
        n_trades = trading.strategy_memory.n_trades + EXCLUDED.n_trades,
        win_rate = ((trading.strategy_memory.win_rate * trading.strategy_memory.n_trades)
                    + (EXCLUDED.win_rate * EXCLUDED.n_trades))
                   / (trading.strategy_memory.n_trades + EXCLUDED.n_trades),
        avg_pnl  = ((trading.strategy_memory.avg_pnl * trading.strategy_memory.n_trades)
                    + (EXCLUDED.avg_pnl * EXCLUDED.n_trades))
                   / (trading.strategy_memory.n_trades + EXCLUDED.n_trades),
        avg_leverage = ((trading.strategy_memory.avg_leverage * trading.strategy_memory.n_trades)
                       + (EXCLUDED.avg_leverage * EXCLUDED.n_trades))
                      / (trading.strategy_memory.n_trades + EXCLUDED.n_trades),
        last_updated = now()
      RETURNING id;
    """)
    
    sample_ins = text("""
      INSERT INTO trading.strategy_samples(memory_id, entry_ts, side, leverage, pnl, reason, mode)
      VALUES (:mid, :et, :sd, :lv, :pl, CAST(:rs AS JSONB), :mode)
    """)
    
    # Procesar cada grupo de trades con la misma firma
    for sig, group in signature_groups.items():
        trades_group = group["trades"]
        feats = group["features"]
        
        # Calcular estadísticas del grupo
        n_trades = len(trades_group)
        wins = sum(1 for t in trades_group if t["pnl"] > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0.0
        avg_pnl = sum(t["pnl"] for t in trades_group) / n_trades if n_trades > 0 else 0.0
        avg_leverage = sum(t.get("leverage", 0.0) for t in trades_group) / n_trades if n_trades > 0 else 0.0
        avg_hold = sum(t.get("hold_bars", 0) for t in trades_group) / n_trades if n_trades > 0 else 0.0
        
        # Insertar/actualizar en strategy_memory
        mid = conn.execute(mem_upsert, {
            "s": symbol, "tf": tf, "sig": sig, "feats": json.dumps(feats),
            "n_trades": n_trades, "win_rate": win_rate, "avg_pnl": avg_pnl,
            "avg_leverage": avg_leverage, "avg_hold": avg_hold, "mode": mode
        }).scalar_one()
        
        # Insertar samples individuales
        for t in trades_group:
            conn.execute(sample_ins, {
                "mid": mid, "et": t["entry_ts"], "sd": t["side"],
                "lv": t.get("leverage", 0.0), "pl": t["pnl"],
                "rs": json.dumps(t.get("reason", {})), "mode": mode
            })


