# core/ml/backtests/backtest_plans.py
import argparse
from datetime import datetime, timedelta, timezone
import math
import os
import json
import hashlib
import yaml
import pandas as pd
import numpy as np
from sqlalchemy import text
from core.data.database import get_engine
from .strategy_memory import update_memory
from ..training.daily_train.balance_manager import load_symbol_config

# -------------------------
# Utilidades / Config
# -------------------------
def load_yaml(path_list):
    for p in path_list:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}

def first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

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
        FROM trading.historicaldata
        WHERE symbol=:s AND timeframe=:tf
          AND timestamp >= :a AND timestamp <= :b
        ORDER BY timestamp ASC
    """)
    df = pd.read_sql(q, conn, params={"s":symbol, "tf":tf, "a":start_ts, "b":end_ts})
    return df

def load_plans(conn, symbol, tf, start_ts, end_ts):
    sql = """
    SELECT id, symbol, timeframe, bar_ts, side,
           entry_px, sl_px, tp_px, qty, leverage, status, reason
    FROM trading.tradeplans
    WHERE symbol = %(symbol)s
      AND timeframe = %(tf)s
      AND bar_ts >= %(from)s AND bar_ts < %(to)s
      AND status IN ('planned','openable')
    ORDER BY bar_ts
    """
    rng = {"from": start_ts, "to": end_ts}
    df = pd.read_sql(sql, conn, params=dict(symbol=symbol, tf=tf, **rng))
    if df.empty:
        return df
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    df = df.sort_values(["bar_ts", "id"]).drop_duplicates(
        subset=["symbol", "timeframe", "bar_ts", "side"], keep="last"
    )
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

    # Balance de referencia para sizing por trade (20%-80%)
    sym_cfg = load_symbol_config(symbol)
    balance_ref = float(sym_cfg.get("initial", 1000.0))

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

        # --- Sizing: usar entre 20% y 80% del balance de referencia (en margen)
        # margen deseado del plan = notional / leverage
        notional_desired = entry_px_eff * abs(qty)
        margin_desired = notional_desired / max(lev, 1e-9)
        cap_min = 0.20 * balance_ref
        cap_max = 0.80 * balance_ref

        if margin_desired <= 0:
            # si el plan trae qty 0 o inválida, saltar
            continue

        if margin_desired < cap_min:
            scale = cap_min / margin_desired
            qty = abs(qty) * scale * (1 if qty >= 0 else -1)
        elif margin_desired > cap_max:
            scale = cap_max / margin_desired
            qty = abs(qty) * scale * (1 if qty >= 0 else -1)

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
            "reason": p["reason"] if isinstance(p["reason"], dict) else json.loads(p["reason"]) if p["reason"] else {},
            "entry_balance": balance_ref,
        })

        # Actualizar balance de referencia tras el trade (equity compuesta)
        balance_ref += pnl_net

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

    # Balances (usar training.yaml por símbolo)
    sym_cfg = load_symbol_config(symbol)
    initial_balance = float(sym_cfg.get("initial", 1000.0))
    target_balance = float(sym_cfg.get("target", max(1.0, initial_balance)))
    final_balance = initial_balance + float(net)
    progress_balance_pct = (final_balance / target_balance) * 100.0 if target_balance > 0 else 0.0

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
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "progress_balance_pct": progress_balance_pct,
        "fees_bps": fees_bps,
        "slip_bps": slip_bps,
        "max_hold_bars": max_hold_bars,
        "metrics": {},
        "comment": (
            f"fees_bps={fees_bps}, slip_bps={slip_bps}, mmr={mmr}, max_hold_bars={max_hold_bars}"
        )
    }
    return summary, trades

def write_results(conn, summary, trades):
    """
    Guarda el resumen del backtest y sus trades.
    SQLAlchemy 2.x: usar sqlalchemy.text() + parámetros :nombre.
    Se adapta al esquema existente (Backtests/BacktestTrades).
    """

    # --- INSERT resumen acorde a scripts/maintenance/create_backtest_tables.sql ---
    ins_bt = text("""
        INSERT INTO trading.backtests
        (symbol, timeframe, from_ts, to_ts,
         n_trades, gross_pnl, fees, net_pnl, win_rate, max_dd,
         initial_balance, final_balance, progress_balance_pct,
         comment, params)
        VALUES (:s, :tf, :a, :b,
                :n, :gp, :f, :np, :wr, :dd,
                :ib, :fb, :pp,
                :c, CAST(:params AS JSONB))
        RETURNING id
    """)

    f_ts = summary["from_ts"]
    t_ts = summary["to_ts"]
    if hasattr(f_ts, "to_pydatetime"): f_ts = f_ts.to_pydatetime()
    if hasattr(t_ts, "to_pydatetime"): t_ts = t_ts.to_pydatetime()

    # construir payload reproducible
    try:
        ver_ids = [t.get("reason", {}).get("direction_ver_id") for t in trades]
        ver_id = max(set([v for v in ver_ids if v is not None]), key=ver_ids.count) if any(ver_ids) else None
    except Exception:
        ver_id = None

    risk_paths = [
        os.path.join("config","risk.yaml"),
        os.path.join("core","config","risk.yaml"),
        "risk.yaml"
    ]
    risk_path = first_existing_path(risk_paths)
    risk_sha = None
    if risk_path:
        try:
            with open(risk_path, "rb") as rf:
                risk_sha = hashlib.sha256(rf.read()).hexdigest()
        except Exception:
            risk_sha = None

    params_payload = {
        "fees_bps": summary.get("fees_bps"),
        "slip_bps": summary.get("slip_bps"),
        "max_hold_bars": summary.get("max_hold_bars"),
        "engine": "vectorized",
        "ver_id": ver_id,
        "risk_yaml_sha": risk_sha,
        "build_script": "build_plans_from_signals@N/A"
    }

    bt_id = conn.execute(ins_bt, {
        "s": summary["symbol"],
        "tf": summary["timeframe"],
        "a": f_ts,
        "b": t_ts,
        "n": summary["n_trades"],
        "gp": summary["gross_pnl"],
        "f": summary["fees"],
        "np": summary["net_pnl"],
        "wr": summary.get("win_rate", 0.0),
        "dd": summary.get("max_dd", 0.0),
        "ib": summary.get("initial_balance"),
        "fb": summary.get("final_balance"),
        "pp": summary.get("progress_balance_pct"),
        "c": summary.get("comment", ""),
        "params": json.dumps(params_payload),
    }).scalar_one()

    # --- INSERT trades ---
    ins_tr = text("""
        INSERT INTO trading.backtesttrades(
            backtest_id, plan_id, symbol, timeframe,
            entry_ts, exit_ts, side, entry_px, exit_px,
            qty, leverage, fee, pnl, reason,
            entry_balance, final_balance
        ) VALUES (
            :bt, :pid, :s, :tf,
            :et, :xt, :sd, :ep, :xp,
            :q, :lv, :fe, :pl, CAST(:rs AS JSONB),
            :eb, :fb
        )
    """)

    for t in trades:
        et = t["entry_ts"]; xt = t["exit_ts"]
        if hasattr(et, "to_pydatetime"): et = et.to_pydatetime()
        if hasattr(xt, "to_pydatetime"): xt = xt.to_pydatetime()

        eb = t.get("entry_balance")
        fb = None
        try:
            if eb is not None and t.get("pnl") is not None and t.get("fee") is not None:
                fb = float(eb) + float(t["pnl"]) - float(t["fee"])
        except Exception:
            fb = None

        conn.execute(ins_tr, {
            "bt": bt_id,
            "pid": t["plan_id"],
            "s":   t["symbol"],
            "tf":  t["timeframe"],
            "et":  et,
            "xt":  xt,
            "sd":  t["side"],
            "ep":  t["entry_px"],
            "xp":  t["exit_px"],
            "q":   t["qty"],
            "lv":  t["leverage"],
            "fe":  t["fee"],
            "pl":  t["pnl"],
            "rs":  json.dumps(t.get("reason", {})),
            "eb":  eb,
            "fb":  fb,
        })

    # actualizar memoria de estrategia con modo backtest
    update_memory(conn, summary["symbol"], summary["timeframe"], trades, mode="backtest")

    try:
        conn.commit()
    except Exception:
        pass

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
        # -- asegurar índices (minúsculas)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_features_sym_tf_ts
            ON trading.features (symbol, timeframe, timestamp DESC);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_bar
            ON trading.tradeplans (symbol, timeframe, bar_ts DESC);
        """))

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
