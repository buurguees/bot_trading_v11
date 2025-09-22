"""
core/data/realtime_updater.py
python -m core.data.realtime_updater
----------------------------------
Actualizador en "casi" tiempo real:
- Rellena gaps desde el último ts en BD hasta NOW.
- Luego hace polling periódico para traer nuevas velas.
- Misma lógica de inserción que historical_downloader (idempotente).

Este módulo está pensado para ejecutarse como servicio (systemd/pm2) o en un screen/tmux.

Requisitos:
- pip install ccxt pyyaml python-dotenv sqlalchemy psycopg2-binary
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List

import ccxt
import yaml
from dotenv import load_dotenv

from core.data.database import MarketDB, TF_MS

load_dotenv(os.path.join("config", ".env"))

logger = logging.getLogger("RealtimeUpdater")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)

CONFIG_PATH = os.path.join("config", "market", "symbols.yaml")
CCXT_LIMIT = 1000


def load_symbols_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_exchange() -> ccxt.bitget:
    api_key = os.getenv("BITGET_API_KEY") or ""
    secret = os.getenv("BITGET_API_SECRET") or ""
    password = os.getenv("BITGET_API_PASSPHRASE") or os.getenv("BITGET_API_PASSWORD") or ""

    ex = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
        "apiKey": api_key,
        "secret": secret,
        "password": password,
    })
    return ex


def backfill_from_last(
    ex: ccxt.Exchange,
    db: MarketDB,
    ccxt_symbol: str,
    db_symbol: str,
    timeframe: str,
    overlap_bars: int = 2,
) -> int:
    """
    Desde el último ts, baja hasta ahora (incluye solapamiento de N velas para asegurar continuidad).
    Solo procesa velas CERRADAS para evitar usar velas en curso.
    """
    tf_ms = TF_MS[timeframe]
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    last_dt = db.get_max_ts(db_symbol, timeframe)
    if last_dt is None:
        logger.info(f"[{db_symbol}][{timeframe}] sin histórico; salta backfill.")
        return 0

    # Calcular cuántas velas deberían haber desde la última hasta ahora
    last_ms = int(last_dt.timestamp() * 1000)
    expected_velas = (now_ms - last_ms) // tf_ms
    
    # Si no hay velas nuevas esperadas, no hacer nada
    if expected_velas <= 0:
        logger.info(f"[{db_symbol}][{timeframe}] no hay velas nuevas esperadas (última: {last_dt})")
        return 0

    since_ms = last_ms - overlap_bars * tf_ms
    since_ms = max(since_ms, 0)

    total = 0
    cursor = since_ms

    while cursor < now_ms:
        try:
            batch = ex.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=timeframe,
                since=cursor,
                limit=CCXT_LIMIT,
            )
        except ccxt.NetworkError as e:
            logger.warning(f"Red: reintentando en 5s… ({e})")
            time.sleep(5)
            continue
        except ccxt.BaseError as e:
            logger.error(f"CCXT error: {e}")
            break

        if not batch:
            break

        # Filtrar solo velas cerradas (excluir la vela en curso)
        closed_batch = [v for v in batch if v[0] < now_ms]
        if not closed_batch:
            break

        rows = MarketDB.ccxt_ohlcv_to_rows(closed_batch, db_symbol, timeframe)
        total += db.upsert_ohlcv_batch(rows)

        cursor = batch[-1][0] + tf_ms
        time.sleep(ex.rateLimit / 1000.0)

        if len(batch) < CCXT_LIMIT:
            if cursor >= now_ms:
                break

    logger.info(f"[{db_symbol}][{timeframe}] backfill completado. filas: {total}")
    return total


def poll_loop(interval_sec: int = 20) -> None:
    """
    Bucle de polling:
    - Para cada símbolo/TF, hace un backfill pequeño + inserta la última vela cerrada.
    - Repite cada interval_sec.
    """
    cfg = load_symbols_config(CONFIG_PATH)
    db = MarketDB()
    ex = make_exchange()
    ex.load_markets()

    symbols = cfg["symbols"]

    while True:
        start = time.time()
        for s in symbols:
            ccxt_symbol = s["ccxt_symbol"]
            db_symbol = s["id"]
            tfs = s.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])

            for tf in tfs:
                try:
                    backfill_from_last(ex, db, ccxt_symbol, db_symbol, tf, overlap_bars=2)
                except Exception as e:
                    logger.exception(f"Error en backfill {db_symbol} {tf}: {e}")

        elapsed = time.time() - start
        sleep_time = max(1.0, interval_sec - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    # Ejecuta: python -m core.data.realtime_updater
    poll_loop(interval_sec=20)
