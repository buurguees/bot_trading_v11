"""
core/features/features_updater.py

Bucle de actualización de FEATURES:
- Recorre todos los (símbolo, timeframe) definidos en config/trading/symbols.yaml
- Llama al calculador incremental (compute_and_save) para rellenar solo las barras nuevas
- Duerme hasta el próximo cierre del TF más corto y repite
Ctrl+C para detener. Usa --once para una sola pasada.
"""

import os
import time
import math
import json
import random
import argparse
import yaml

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from datetime import datetime, timezone

from dotenv import load_dotenv

# Reutilizamos tu calculador incremental
# (asumimos que core/features/indicator_calculator.py está tal como lo dejaste)
from core.features.indicator_calculator import compute_and_save

load_dotenv("config/.env")

CONFIG_PATH = "config/trading/symbols.yaml"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
POLL_GRACE_MS = int(os.getenv("POLL_GRACE_MS", "5000"))   # ms de “gracia” tras cierre
JITTER_MAX_MS = int(os.getenv("JITTER_MAX_MS", "1500"))   # ms aleatorios para repartir carga

def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} - features_updater - {msg}", flush=True)

def timeframe_to_ms(tf: str) -> int:
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    if tf not in mapping:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return mapping[tf]

def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def next_bar_sleep_seconds(min_tf_ms: int, grace_ms: int = POLL_GRACE_MS, jitter_ms: int = JITTER_MAX_MS) -> float:
    t = now_ms()
    next_bar = ((t // min_tf_ms) + 1) * min_tf_ms
    wait_ms = max(0, (next_bar + grace_ms) - t) + random.randint(0, jitter_ms)
    return wait_ms / 1000.0

def read_symbols_and_tfs(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    symbols_cfg = cfg.get("symbols", {}) or {}
    out = []
    for sym, meta in symbols_cfg.items():
        ccxt_symbol = meta.get("ccxt_symbol") or sym
        tfs = meta.get("timeframes", default_tfs)
        out.append((ccxt_symbol, tfs))
    return out

def min_tf_ms_from_config(path=CONFIG_PATH) -> int:
    tfs = set()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    for _, meta in (cfg.get("symbols") or {}).items():
        tfs.update(meta.get("timeframes", default_tfs))
    if not tfs:
        tfs = set(default_tfs)
    return min(timeframe_to_ms(tf) for tf in tfs)

def one_pass_all() -> int:
    total = 0
    for symbol, tfs in read_symbols_and_tfs():
        for tf in tfs:
            try:
                ins = compute_and_save(symbol, tf)  # incremental (usa MAX(timestamp) en Features)
                log(f"[{symbol} {tf}] upsert features: {ins}")
                total += (ins or 0)
            except Exception as e:
                log(f"[{symbol} {tf}] ERROR: {e}")
    return total

def run_loop():
    min_tf_ms = min_tf_ms_from_config()
    log(f"Min timeframe: {min_tf_ms/1000:.0f}s. Entrando en bucle… (Ctrl+C para salir)")
    while True:
        try:
            inserted = one_pass_all()
            log(f"Pasada completada. Filas upserted: {inserted}")
        except Exception as e:
            log(f"ERROR en pasada: {e}")
        sleep_s = next_bar_sleep_seconds(min_tf_ms)
        log(f"Durmiendo {sleep_s:.1f}s hasta el próximo cierre…")
        time.sleep(sleep_s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Ejecuta solo una pasada y termina.")
    args = parser.parse_args()
    if args.once:
        inserted = one_pass_all()
        log(f"Pasada única completa. Filas upserted: {inserted}")
    else:
        run_loop()

if __name__ == "__main__":
    main()
