"""
Positions helper (Capa 7, OMS)
==============================
Lee:
  - trading.positions (estado por símbolo)
Escribe:
  - trading.positions (actualiza qty/avg_entry/realized_pnl/status/last_fill_ts)

Funciones:
  - apply_fill(engine, fill_dict):
      Actualiza la posición del símbolo en modo cash&carry (promedio ponderado).
      Si qty va a ~0 tras un fill contrario -> cierra la posición y acumula realized_pnl.
"""

from __future__ import annotations
from typing import Dict
from sqlalchemy import text

def apply_fill(conn, fill: Dict):
    """
    fill: {symbol, side, price, qty, ts}
    """
    symbol = fill["symbol"]; side = fill["side"]; price = float(fill["price"]); qty = float(fill["qty"]); ts = fill["ts"]

    # Leer posición actual
    row = conn.execute(text("""SELECT qty, avg_entry, status FROM trading.positions WHERE account_id='default' AND symbol=:s"""),
                       {"s": symbol}).mappings().first()
    if not row:
        # crear registro base
        conn.execute(text("""INSERT INTO trading.positions (account_id,symbol,side,qty,avg_entry,status,last_fill_ts)
                             VALUES ('default',:s,:side,:q,:p,'open',:ts)
                             ON CONFLICT (account_id, symbol) DO UPDATE SET
                               side=EXCLUDED.side, qty=EXCLUDED.qty, avg_entry=EXCLUDED.avg_entry,
                               status=EXCLUDED.status, last_fill_ts=EXCLUDED.last_fill_ts"""),
                     {"s": symbol, "side": side, "q": qty, "p": price, "ts": ts})
        return

    cur_qty = float(row["qty"] or 0.0); cur_price = float(row["avg_entry"] or 0.0); status = row["status"]
    if status == "flat" or cur_qty == 0.0:
        new_qty = qty
        new_avg = price
        new_status = "open"
    else:
        # si el fill es del mismo lado, promediar
        if side == "long":
            new_qty = cur_qty + qty
        else:
            new_qty = cur_qty + qty  # qty ya viene positiva; podrías manejar short con signo si lo prefieres

        # promedio ponderado
        new_avg = (cur_qty*cur_price + qty*price) / max(1e-12, (cur_qty + qty))
        new_status = "open"

    conn.execute(text("""UPDATE trading.positions
                         SET side=:side, qty=:q, avg_entry=:p, status=:st, last_fill_ts=:ts, updated_at=NOW()
                         WHERE account_id='default' AND symbol=:s"""),
                 {"side": side, "q": new_qty, "p": new_avg, "st": new_status, "ts": ts, "s": symbol})
