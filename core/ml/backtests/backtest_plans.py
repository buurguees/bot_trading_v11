# core/ml/backtests/backtest_plans.py
import argparse
from datetime import datetime, timedelta, timezone
import math
import os
import json
import yaml
import pandas as pd
import numpy as np
from sqlalchemy import text
from core.data.database import get_engine

# -------------------------
# Utilidades / Config
# -------------------------
def load_yaml(path_list):
    for p in path_list:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}

def tf_to_minutes(tf: str) -> int:
    tf = tf.lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 1440
    raise ValueError(f"Timeframe desconocido: {tf}")

def first_touch_exit(row, side, tp, sl, liq):
    """
    Decide salida intrabar:
    - Long: si low<=sl -> SL; elif high>=tp -> TP; elif low<=liq -> LIQ
    - Short: si high>=sl -> SL; elif low<=tp -> TP; elif high>=liq -> LIQ
    Priorizamos el caso peor (SL antes que TP si coinciden).
    """
    if side == 1:
        if row["low"]  <= sl:  return ("SL",  sl)
        if row["high"] >= tp:  return ("TP",  tp)
        if row["low"]  <= liq: return ("LIQ", liq)
        return (None, None)
    else:
        if row["high"] >= sl:  return ("SL",  sl)
        if row["low"]  <= tp:  return ("TP",  tp)
        if row["high"] >= liq: return ("LIQ", liq)
        return (None, None)

def approx_liq(entry_px, side, lev, mmr=0.005):
    """
    Aprox de precio de liquidación para futuros lineales USDT-M (aislado).
    Conservadora y simple:
      Long : entry * (1 - 1/lev + mmr)
      Short: entry * (1 + 1/lev - mmr)
    """
    if side == 1:
        return entry_px * (1 - 1.0/lev + mmr)
    else:
        return entry_px * (1 + 1.0/lev - mmr)

def apply_slippage(px, side, slip_bps):
    if slip_bps <= 0: return px
    factor = 1.0 + (slip_bps / 10_000.0)
    # Entry/exit “peor precio”:
    return px * (factor if side == 1 else (2 - factor))  # para side=-1 => divide por factor

def taker_fee(notional, fee_bps):
    return notional * (fee_bps / 10_000.0)

# -------------------------
# Core
# -------------------------
def load_ohlcv(conn, symbol, tf, start_ts, end_ts):
    q = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM trading.HistoricalData
        WHERE symbol=:s AND timeframe=:tf
          AND timestamp >= :a AND timestamp <= :b
        ORDER BY timestamp ASC
    """)
    df = pd.read_sql(q, conn, params={"s":symbol, "tf":tf, "a":start_ts, "b":end_ts})
    return df

def load_plans(conn, symbol, tf, start_ts, end_ts):
    q = text("""
        SELECT id, created_at, bar_ts, symbol, timeframe, side, entry_px, sl_px, tp_px, qty, leverage, reason
        FROM trading.TradePlans
        WHERE symbol=:s AND timeframe=:tf
          AND created_at BETWEEN :a AND :b
          AND status IN ('planned','simulated')
        ORDER BY created_at ASC
    """)
    df = pd.read_sql(q, conn, params={"s":symbol, "tf":tf, "a":start_ts, "b":end_ts})
    # asegurar bar_ts: si viene nulo, usar created_at truncado a minuto
    if "bar_ts" in df.columns:
        df["bar_ts"] = pd.to_datetime(df["bar_ts"]).fillna(pd.to_datetime(df["created_at"]).dt.floor("min"))
    else:
        df["bar_ts"] = pd.to_datetime(df["created_at"]).dt.floor("min")
    return df

def simulate_symbol_tf(conn, symbol, tf, from_ts, to_ts, fees_bps, slip_bps, mmr, max_hold_bars,
                       funding_bps_8h: float = 0.0):
    plans = load_plans(conn, symbol, tf, from_ts, to_ts)
    if plans.empty:
        return None, []

    # cargar OHLCV suficiente (desde 1 barra después del bar_ts mínimo)
    start_px = plans["bar_ts"].min() + pd.Timedelta(minutes=tf_to_minutes(tf))
    px = load_ohlcv(conn, symbol, tf, start_px, to_ts + timedelta(minutes= tf_to_minutes(tf)*max(2, max_hold_bars)))
    if px.empty:
        return None, []

    px["timestamp"] = pd.to_datetime(px["timestamp"])
    px.set_index("timestamp", inplace=True)

    trades = []
    gross, fees, net = 0.0, 0.0, 0.0
    eq = 0.0
    eq_series = []
    liq_count = 0

    for _, p in plans.iterrows():
        side = int(p["side"])
        entry_bar = pd.to_datetime(p["bar_ts"]) + pd.Timedelta(minutes=tf_to_minutes(tf))  # entramos en la siguiente barra
        # la serie de barras a evaluar (hasta max_hold_bars)
        future = px.loc[entry_bar:].iloc[:max_hold_bars]
        if future.empty: 
            continue

        entry_px = float(p["entry_px"])
        tp = float(p["tp_px"])
        sl = float(p["sl_px"])
        qty = float(p["qty"])
        lev = float(p["leverage"])
        liq = approx_liq(entry_px, side, lev, mmr)

        # aplicar slippage a la entrada “en contra”
        entry_px_eff = apply_slippage(entry_px, side, slip_bps)
        open_time = future.index[0]

        exit_side = None
        exit_px_eff = None
        exit_ts = None

        # Buscar primer toque
        for ts, row in future.iterrows():
            exit_side, px_exit = first_touch_exit(row, side, tp, sl, liq)
            if exit_side is not None:
                # slippage al cierre, “en contra”
                exit_px_eff = apply_slippage(px_exit, -side, slip_bps)
                exit_ts = ts
                break

        # Si no hubo toque, cerramos a close de la última barra considerada
        if exit_side is None:
            ts = future.index[-1]
            px_exit = float(future.loc[ts, "close"])
            exit_px_eff = apply_slippage(px_exit, -side, slip_bps)
            exit_ts = ts
            exit_side = "TIME"

        # PnL lineal (USDT-M)
        pnl = (exit_px_eff - entry_px_eff) * side * qty
        fee_open  = taker_fee(entry_px_eff * qty, fees_bps)
        fee_close = taker_fee(exit_px_eff  * qty, fees_bps)
        fee_total = fee_open + fee_close

        # Funding (aprox): tasa bps por 8h prorrateada por horas en posición sobre nocional medio
        holding_hours = max(0.0, (exit_ts - open_time).total_seconds() / 3600.0)
        funding_rate_per_hour = (funding_bps_8h / 8.0) / 10_000.0 if funding_bps_8h else 0.0
        notional_mean = ((entry_px_eff + exit_px_eff) / 2.0) * abs(qty)
        funding_fee = notional_mean * funding_rate_per_hour * holding_hours

        pnl_net = pnl - fee_total - funding_fee

        gross += pnl
        fees  += fee_total
        net   += pnl_net
        eq += pnl_net
        eq_series.append(eq)
        if exit_side == "LIQ":
            liq_count += 1

        trades.append({
            "plan_id": int(p["id"]),
            "symbol": symbol,
            "timeframe": tf,
            "entry_ts": open_time.to_pydatetime().replace(tzinfo=timezone.utc),
            "exit_ts": exit_ts.to_pydatetime().replace(tzinfo=timezone.utc),
            "side": side,
            "entry_px": entry_px_eff,
            "exit_px": exit_px_eff,
            "qty": qty,
            "leverage": lev,
            "fee": fee_total,
            "pnl": pnl_net,
            "funding": funding_fee,
            "exit": exit_side,
            "reason": p["reason"] if isinstance(p["reason"], dict) else json.loads(p["reason"]) if p["reason"] else {}
        })

    if not trades:
        return None, []

    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / len(trades) if trades else 0.0
    # max drawdown de la equity walk
    if eq_series:
        curve = np.array([0.0] + eq_series)  # empieza en 0
        peak = np.maximum.accumulate(curve)
        dd = np.min(curve - peak)
        max_dd = float(abs(dd))
    else:
        max_dd = 0.0

    # Sharpe simple por trade sobre incrementos (sin anualizar)
    if trades:
        increments = np.diff([0.0] + eq_series)
        mu = float(np.mean(increments))
        sigma = float(np.std(increments))
        sharpe = (mu / sigma) if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    summary = {
        "symbol": symbol,
        "timeframe": tf,
        "from_ts": from_ts,
        "to_ts": to_ts,
        "n_trades": len(trades),
        "gross_pnl": gross,
        "fees": fees,
        "net_pnl": net,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "comment": (
            f"fees_bps={fees_bps}, slip_bps={slip_bps}, mmr={mmr}, max_hold_bars={max_hold_bars}, "
            f"funding_bps_8h={funding_bps_8h}, sharpe={sharpe:.4f}, liq_pct={(liq_count/len(trades)) if trades else 0.0:.4f}"
        )
    }
    return summary, trades

def write_results(conn, summary, trades):
    ins_bt = text("""
        INSERT INTO trading.Backtests(symbol,timeframe,from_ts,to_ts,n_trades,gross_pnl,fees,net_pnl,win_rate,max_dd,comment)
        VALUES (:s,:tf,:a,:b,:n,:gp,:f,:np,:wr,:dd,:c)
        RETURNING id
    """)
    bt_id = conn.execute(ins_bt, {
        "s":summary["symbol"], "tf":summary["timeframe"],
        "a":summary["from_ts"], "b":summary["to_ts"],
        "n":summary["n_trades"], "gp":summary["gross_pnl"], "f":summary["fees"],
        "np":summary["net_pnl"], "wr":summary["win_rate"], "dd":summary["max_dd"],
        "c":summary["comment"]
    }).scalar()

    ins_tr = text("""
        INSERT INTO trading.BacktestTrades(
            backtest_id, plan_id, symbol, timeframe, entry_ts, exit_ts, side, entry_px, exit_px, qty, leverage, fee, pnl, reason
        ) VALUES (
            :bt, :pid, :s, :tf, :et, :xt, :sd, :ep, :xp, :q, :lv, :fe, :pl, :rs::jsonb
        )
    """)
    for t in trades:
        conn.execute(ins_tr, {
            "bt": bt_id, "pid": t["plan_id"], "s": t["symbol"], "tf": t["timeframe"],
            "et": t["entry_ts"], "xt": t["exit_ts"], "sd": t["side"],
            "ep": t["entry_px"], "xp": t["exit_px"], "q": t["qty"], "lv": t["leverage"],
            "fe": t["fee"], "pl": t["pnl"], "rs": json.dumps(t["reason"])
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf",     default="1m")
    ap.add_argument("--from",   dest="from_ts", default=None, help="YYYY-MM-DD or ISO")
    ap.add_argument("--to",     dest="to_ts",   default=None, help="YYYY-MM-DD or ISO")
    ap.add_argument("--max-hold-bars", type=int, default=500)
    ap.add_argument("--fees-bps",  type=float, default=None)
    ap.add_argument("--slip-bps",  type=float, default=None)
    ap.add_argument("--mmr",       type=float, default=None, help="maintenance margin rate aprox")
    ap.add_argument("--funding-bps-8h", type=float, default=None, help="funding en bps por 8h (aprox)")
    args = ap.parse_args()

    # rango temporal por defecto: último día
    now = datetime.now(timezone.utc)
    from_ts = pd.to_datetime(args.from_ts) if args.from_ts else (now - timedelta(days=1))
    to_ts   = pd.to_datetime(args.to_ts)   if args.to_ts   else now

    # cargar risk.yaml (si existe)
    risk_cfg = load_yaml([
        os.path.join("config","risk.yaml"),
        os.path.join("core","config","risk.yaml"),
        "risk.yaml"
    ])
    # defaults razonables si no hay yaml
    fees_bps = args.fees_bps if args.fees_bps is not None else float(risk_cfg.get("fees_bps", 2.0))
    slip_bps = args.slip_bps if args.slip_bps is not None else float(risk_cfg.get("slippage_bps", 2.0))
    mmr      = args.mmr      if args.mmr      is not None else float(risk_cfg.get("maint_margin_rate", 0.005))
    funding8 = args.funding_bps_8h if args.funding_bps_8h is not None else float(risk_cfg.get("funding_bps_8h", 0.0))

    engine = get_engine()
    with engine.begin() as conn:
        summary, trades = simulate_symbol_tf(
            conn, args.symbol, args.tf, from_ts, to_ts,
            fees_bps, slip_bps, mmr, args.max_hold_bars, funding_bps_8h=funding8
        )
        if not trades:
            print("No hay trades para backtest en el rango indicado.")
            return
        write_results(conn, summary, trades)
        print(f"[Backtest] {summary['symbol']}-{summary['timeframe']} "
              f"n={summary['n_trades']} net={summary['net_pnl']:.4f} "
              f"wr={summary['win_rate']:.3f} dd={summary['max_dd']:.4f}")

if __name__ == "__main__":
    main()
