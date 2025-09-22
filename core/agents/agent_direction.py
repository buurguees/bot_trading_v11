"""
Agent Direction (heurístico)
----------------------------
Lee: market.features (última vela cerrada del TF de consulta)
Escribe: ml.agent_preds con task='direction' (labels: 'long'|'short'|'flat')

Reglas simples:
- LONG si EMA21>EMA50, RSI>55, MACD_hist>0  (más señales -> más confianza)
- SHORT si EMA21<EMA50, RSI<45, MACD_hist<0
- FLAT en caso contrario
"""

from __future__ import annotations
import logging
from core.agents._common import load_symbols_and_query_tf, fetch_feature_row, upsert_pred

logger = logging.getLogger("AgentDirection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout); logger.addHandler(h)

def _score_long(f):
    s = 0
    s += 1 if (f.get("ema_21") or 0) > (f.get("ema_50") or 0) else 0
    s += 1 if (f.get("rsi_14") or 0) > 55 else 0
    s += 1 if (f.get("macd_hist") or 0) > 0 else 0
    return s

def _score_short(f):
    s = 0
    s += 1 if (f.get("ema_21") or 0) < (f.get("ema_50") or 0) else 0
    s += 1 if (f.get("rsi_14") or 0) < 45 else 0
    s += 1 if (f.get("macd_hist") or 0) < 0 else 0
    return s

def run_once() -> int:
    symbols, qtf = load_symbols_and_query_tf(task="direction", default_tf="1m")
    wrote = 0
    for sym in symbols:
        f = fetch_feature_row(sym, qtf)
        if not f: 
            continue
        long_s = _score_long(f)
        short_s = _score_short(f)
        if long_s > short_s and long_s >= 2:
            label, conf = "long", long_s/3.0
        elif short_s > long_s and short_s >= 2:
            label, conf = "short", short_s/3.0
        else:
            label, conf = "flat", 0.50
        probs = {"long": long_s/3.0, "short": short_s/3.0, "flat": 1.0 - max(long_s, short_s)/3.0}
        upsert_pred(f["ts"], sym, qtf, "direction", label, conf, probs)
        wrote += 1
        logger.info(f"[{sym} {qtf}] direction={label} conf={conf:.2f}")
    return wrote

if __name__ == "__main__":
    run_once()
