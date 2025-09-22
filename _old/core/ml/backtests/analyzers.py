import pandas as pd

def simple_pnl(signals: pd.DataFrame, prices: pd.DataFrame, fee_bps: float = 2.0) -> pd.DataFrame:
    """
    signals: columns [timestamp, side]  (side ∈ {-1,0,1})
    prices : columns [timestamp, close]
    Estrategia 'close->close', con comisión fija en bps cuando cambia la posición.
    """
    px = prices[["timestamp","close"]].copy().sort_values("timestamp")
    px["ret"] = px["close"].pct_change().fillna(0)

    # Reindexa señales al índice temporal de precios (ffill, arranque 0)
    s = signals[["timestamp","side"]].copy().sort_values("timestamp")
    s = s.set_index("timestamp")["side"].astype(float)
    side_aligned = s.reindex(px["timestamp"]).ffill().fillna(0).values

    px["side"] = side_aligned

    fee = fee_bps / 10000.0
    churn = (px["side"] != pd.Series(side_aligned).shift(1)).astype(int).fillna(0)

    px["pnl"] = px["side"].shift(1).fillna(0) * px["ret"] - churn * fee
    px["equity"] = (1.0 + px["pnl"]).cumprod()

    return px[["timestamp","close","side","ret","pnl","equity"]]
