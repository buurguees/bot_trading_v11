"""
Agent SMC (heurístico)
----------------------
Lee: market.features
Escribe: ml.agent_preds con task='smc' (labels: 'bull'|'bear'|'neutral')

Reglas:
- 'bull' si supertrend_dir>0 o (close>EMA50 y RSI>50) o smc_bull=1
- 'bear' si supertrend_dir<0 o (close<EMA50 y RSI<50) o smc_bear=1
- 'neutral' en lo demás
"""

from __future__ import annotations
import logging
from core.agents._common import load_symbols_and_query_tf, fetch_feature_row, upsert_pred

logger = logging.getLogger("AgentSMC")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout); logger.addHandler(h)

def run_once() -> int:
    symbols, qtf = load_symbols_and_query_tf(task="smc", default_tf="5m")
    wrote = 0
    for sym in symbols:
        f = fetch_feature_row(sym, qtf)
        if not f: 
            continue
        st = int(f.get("supertrend_dir") or 0)
        close = float(f.get("close") or 0.0)
        ema50 = float(f.get("ema_50") or 0.0)
        rsi = float(f.get("rsi_14") or 50.0)
        bull_flag = int(f.get("smc_bull") or 0)
        bear_flag = int(f.get("smc_bear") or 0)

        bull = (st > 0) or ((close > ema50) and (rsi > 50)) or (bull_flag == 1)
        bear = (st < 0) or ((close < ema50) and (rsi < 50)) or (bear_flag == 1)

        if bull and not bear:
            label, conf = "bull", 0.7
        elif bear and not bull:
            label, conf = "bear", 0.7
        else:
            label, conf = "neutral", 0.55

        probs = {"bull": 0.7 if label=="bull" else 0.15, "bear": 0.7 if label=="bear" else 0.15, "neutral": 0.7 if label=="neutral" else 0.15}
        upsert_pred(f["ts"], sym, qtf, "smc", label, conf, probs)
        wrote += 1
        logger.info(f"[{sym} {qtf}] smc={label} conf={conf:.2f}")
    return wrote

if __name__ == "__main__":
    run_once()
