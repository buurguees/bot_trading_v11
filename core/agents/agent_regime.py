"""
Agent Regime (heurístico)
-------------------------
Lee: market.features
Escribe: ml.agent_preds con task='regime' (labels: 'trend'|'range')

Reglas:
- 'trend' si |EMA50-EMA200|/close > 0.004 o si supertrend_dir≠0
- 'range' en caso contrario
"""

from __future__ import annotations
import logging
from core.agents._common import load_symbols_and_query_tf, fetch_feature_row, upsert_pred

logger = logging.getLogger("AgentRegime")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout); logger.addHandler(h)

def run_once() -> int:
    symbols, qtf = load_symbols_and_query_tf(task="regime", default_tf="5m")
    wrote = 0
    for sym in symbols:
        f = fetch_feature_row(sym, qtf)
        if not f:
            continue
        close = float(f.get("close") or 0.0)
        e50 = float(f.get("ema_50") or 0.0)
        e200 = float(f.get("ema_200") or 0.0)
        st = int(f.get("supertrend_dir") or 0)
        slope = abs(e50 - e200) / close if close > 0 else 0.0
        label = "trend" if (slope > 0.004 or st != 0) else "range"
        conf = min(0.95, 0.6 + (slope*50.0)) if label == "trend" else 0.6
        upsert_pred(f["ts"], sym, qtf, "regime", label, conf, {"trend": conf, "range": 1.0-conf})
        wrote += 1
        logger.info(f"[{sym} {qtf}] regime={label} conf={conf:.2f}")
    return wrote

if __name__ == "__main__":
    run_once()
