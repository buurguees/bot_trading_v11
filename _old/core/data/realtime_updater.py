"""
core/data/realtime_updater.py

Modo 'daemon' simple: actualiza velas de Futuros para todos los símbolos/TFs
desde el último timestamp en DB hasta 'ahora', y se queda en bucle
esperando al siguiente cierre de vela del timeframe más corto.

Ctrl+C para detener.
"""

import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import math
import random

# --------------------------
# Configuración básica
# --------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("realtime_updater")

load_dotenv(dotenv_path="config/.env")
DB_URL = os.getenv("DB_URL")

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE")

CONFIG_PATH = "config/trading/symbols.yaml"
ENGINE = create_engine(DB_URL, pool_pre_ping=True)

# Límite CCXT y concurrencia
REQUEST_LIMIT = int(os.getenv("REQUEST_LIMIT", "1000"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "6"))

# Si no hay datos previos, cuánto retroceder (minutos)
FALLBACK_LOOKBACK_MIN = int(os.getenv("FALLBACK_LOOKBACK_MIN", "1440"))  # 1 día

# Espera extra tras el cierre de vela para evitar velas en formación (ms)
POLL_GRACE_MS = int(os.getenv("POLL_GRACE_MS", "5000"))

# Jitter aleatorio para evitar sincronización perfecta (ms)
JITTER_MAX_MS = int(os.getenv("JITTER_MAX_MS", "1500"))

# --------------------------
# Utilidades tiempo/TF
# --------------------------
def timeframe_to_ms(tf: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if tf not in mapping:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return mapping[tf]

def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def next_bar_sleep_seconds(min_tf_ms: int, grace_ms: int = POLL_GRACE_MS) -> float:
    """Calcula segundos hasta el próximo cierre del TF más corto, + gracia + jitter."""
    t = now_ms()
    next_bar = ((t // min_tf_ms) + 1) * min_tf_ms
    wait_ms = max(0, (next_bar + grace_ms) - t)
    # pequeño jitter para repartir carga
    wait_ms += random.randint(0, JITTER_MAX_MS)
    return wait_ms / 1000.0

# --------------------------
# Utilidades DB
# --------------------------
def get_last_ts_ms(symbol: str, timeframe: str) -> Optional[int]:
    q = text("""
        SELECT EXTRACT(EPOCH FROM MAX(timestamp)) * 1000 AS ts_ms
        FROM trading.historicaldata
        WHERE symbol=:symbol AND timeframe=:tf
    """)
    with ENGINE.begin() as conn:
        row = conn.execute(q, {"symbol": symbol, "tf": timeframe}).scalar()
        return int(row) if row else None

def save_batch_to_db(rows: List[Tuple[int, float, float, float, float, float]],
                     symbol: str, timeframe: str) -> int:
    if not rows:
        return 0
    insert_q = text("""
        INSERT INTO trading.historicaldata (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :tf, to_timestamp(:tsms / 1000.0), :o, :h, :l, :c, :v)
        ON CONFLICT ON CONSTRAINT unique_ohlcv DO NOTHING
    """)
    inserted = 0
    try:
        with ENGINE.begin() as conn:
            for tsms, o, h, l, c, v in rows:
                conn.execute(insert_q, {
                    "symbol": symbol, "tf": timeframe, "tsms": tsms,
                    "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)
                })
                inserted += 1
        return inserted
    except SQLAlchemyError as e:
        logger.error(f"[DB] Error guardando {symbol} {timeframe}: {e}")
        return inserted

# --------------------------
# Exchange (reutilizado en bucle)
# --------------------------
exchange = ccxt.bitget({
    "apiKey": BITGET_API_KEY,
    "secret": BITGET_SECRET_KEY,
    "password": BITGET_PASSPHRASE,
    "enableRateLimit": True,
})

async def fetch_ohlcv(symbol: str, timeframe: str, since_ms_: int, limit: int = REQUEST_LIMIT) -> Optional[List[List]]:
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe, since_ms_, limit)
        return data or None
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"[{symbol} {timeframe}] Rate limit: {e}. Esperando 60s…")
        await asyncio.sleep(60)
        return await fetch_ohlcv(symbol, timeframe, since_ms_, limit)
    except ccxt.NetworkError as e:
        logger.warning(f"[{symbol} {timeframe}] Network error: {e}. Reintentando en 10s…")
        await asyncio.sleep(10)
        return await fetch_ohlcv(symbol, timeframe, since_ms_, limit)
    except Exception as e:
        logger.error(f"[{symbol} {timeframe}] Error inesperado: {e}")
        return None

def filter_closed_candles(rows: List[List], timeframe: str) -> List[List]:
    if not rows:
        return rows
    cutoff = now_ms() - timeframe_to_ms(timeframe)
    return [c for c in rows if c[0] <= cutoff]

# --------------------------
# Catch-up por símbolo/TF
# --------------------------
async def catch_up_symbol_tf(symbol: str, timeframe: str,
                             fallback_minutes: int = FALLBACK_LOOKBACK_MIN) -> int:
    current_ms = now_ms()
    last_ts = get_last_ts_ms(symbol, timeframe)
    if last_ts:
        since_ms_ = last_ts + 1
        logger.info(f"[{symbol} {timeframe}] Reanudando desde {last_ts} (ms).")
    else:
        since_ms_ = current_ms - fallback_minutes * 60_000
        logger.info(f"[{symbol} {timeframe}] Sin datos previos. Retrocediendo {fallback_minutes} min.")

    total_saved = 0
    step_guard = timeframe_to_ms(timeframe)

    while since_ms_ < current_ms:
        batch = await fetch_ohlcv(symbol, timeframe, since_ms_, REQUEST_LIMIT)
        if not batch:
            # evita loop infinito
            since_ms_ += step_guard
            await asyncio.sleep(0.5)
            continue

        batch = filter_closed_candles(batch, timeframe)
        if not batch:
            break

        saved = save_batch_to_db(
            [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in batch],
            symbol, timeframe
        )
        total_saved += saved
        since_ms_ = batch[-1][0] + 1
        await asyncio.sleep(0.2)

    logger.info(f"[{symbol} {timeframe}] Catch-up insertadas ~{total_saved} velas.")
    return total_saved

# --------------------------
# Una pasada completa (todos símbolos/TFs)
# --------------------------
async def one_pass_all(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    symbols_cfg: Dict = cfg.get("symbols", {})

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total = 0

    async def worker(ccxt_symbol: str, tfs: List[str]):
        nonlocal total
        async with sem:
            for tf in tfs:
                total += await catch_up_symbol_tf(ccxt_symbol, tf)

    tasks = []
    for sym, stg in symbols_cfg.items():
        tfs = stg.get("timeframes", default_tfs)
        ccxt_symbol = stg.get("ccxt_symbol") or sym
        tasks.append(worker(ccxt_symbol, tfs))

    await asyncio.gather(*tasks)
    return total

def get_min_tf_ms_from_config(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    symbols_cfg: Dict = cfg.get("symbols", {})

    tfs = set()
    for _, stg in symbols_cfg.items():
        tfs.update(stg.get("timeframes", default_tfs))
    if not tfs:
        tfs = set(default_tfs)
    return min(timeframe_to_ms(tf) for tf in tfs)

# --------------------------
# Bucle infinito
# --------------------------
async def run_forever(config_path: str = CONFIG_PATH):
    await exchange.load_markets()
    min_tf_ms = get_min_tf_ms_from_config(config_path)
    logger.info(f"Min timeframe en config = {min_tf_ms/1000:.0f}s. Entrando en bucle… Ctrl+C para detener.")

    while True:
        try:
            inserted = await one_pass_all(config_path)
            logger.info(f"Pasada completada. Velas insertadas ~{inserted}.")
        except Exception as e:
            logger.exception(f"Error en pasada: {e}")

        sleep_s = next_bar_sleep_seconds(min_tf_ms)
        logger.info(f"Esperando {sleep_s:.1f}s hasta el próximo cierre del TF más corto…")
        await asyncio.sleep(sleep_s)

async def main():
    try:
        await run_forever(CONFIG_PATH)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.info("Cancelado por el usuario.")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass
        ENGINE.dispose()
        logger.info("Cerrado limpio.")

if __name__ == "__main__":
    asyncio.run(main())
