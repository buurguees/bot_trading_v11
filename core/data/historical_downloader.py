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

# --------------------------
# Configuración básica
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("historical_downloader")

load_dotenv(dotenv_path="config/.env")
DB_URL = os.getenv("DB_URL")
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE")

CONFIG_PATH = "config/trading/symbols.yaml"
ENGINE = create_engine(DB_URL)

# Límite CCXT
REQUEST_LIMIT = 1000
# Concurrencia simultánea de símbolos/timeframes
MAX_CONCURRENT = 4
# Días por defecto a retroceder si no hay datos previos
DEFAULT_SINCE_DAYS = int(os.getenv("SINCE_DAYS", "365"))

# --------------------------
# Utilidades DB
# --------------------------
def get_last_ts_ms(symbol: str, timeframe: str) -> Optional[int]:
    """
    Devuelve el último timestamp (ms) presente en DB para (symbol, timeframe).
    Si no hay registros, None.
    """
    q = text("""
        SELECT EXTRACT(EPOCH FROM MAX(timestamp)) * 1000 AS ts_ms
        FROM trading.HistoricalData
        WHERE symbol=:symbol AND timeframe=:tf
    """)
    with ENGINE.begin() as conn:
        row = conn.execute(q, {"symbol": symbol, "tf": timeframe}).scalar()
        return int(row) if row else None

def save_batch_to_db(rows: List[Tuple[int, float, float, float, float, float]], symbol: str, timeframe: str) -> int:
    """
    Inserta un lote de OHLCV en DB. rows: [ [ts_ms, o, h, l, c, v], ... ]
    Devuelve cuántas filas se han insertado (aprox; ON CONFLICT ignora duplicados).
    """
    if not rows:
        return 0

    insert_q = text("""
        INSERT INTO trading.HistoricalData (symbol, timeframe, timestamp, open, high, low, close, volume)
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
# Exchange
# --------------------------
exchange = ccxt.bitget({
    "apiKey": BITGET_API_KEY,
    "secret": BITGET_SECRET_KEY,
    "password": BITGET_PASSPHRASE,
    "enableRateLimit": True,
})

async def fetch_ohlcv(symbol: str, timeframe: str, since_ms: int, limit: int = REQUEST_LIMIT) -> Optional[List[List]]:
    """
    Descarga un bloque OHLCV. Devuelve lista de velas [[ms, o, h, l, c, v], ...]
    """
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
        if not data:
            return None
        return data
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"[{symbol} {timeframe}] Rate limit: {e}. Esperando 60s…")
        await asyncio.sleep(60)
        return await fetch_ohlcv(symbol, timeframe, since_ms, limit)
    except ccxt.NetworkError as e:
        logger.warning(f"[{symbol} {timeframe}] Network error: {e}. Reintentando en 10s…")
        await asyncio.sleep(10)
        return await fetch_ohlcv(symbol, timeframe, since_ms, limit)
    except Exception as e:
        logger.error(f"[{symbol} {timeframe}] Error inesperado: {e}")
        return None

def timeframe_to_ms(tf: str) -> int:
    """Tamaño de vela en milisegundos (aprox) para avanzar en bucle si la API devuelve vela repetida."""
    mapping = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
    }
    return mapping.get(tf, 60_000)

async def backfill_symbol_tf(symbol: str, timeframe: str, since_days: int):
    """
    Backfill para un símbolo/TF.
    - Si hay datos, continúa desde el último ts + 1ms.
    - Si no hay, arranca desde now - since_days.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    last_ts_ms = get_last_ts_ms(symbol, timeframe)

    if last_ts_ms:
        since_ms = last_ts_ms + 1
        logger.info(f"[{symbol} {timeframe}] Reanudando desde {last_ts_ms} (ms).")
    else:
        since_ms = now_ms - since_days * 24 * 60 * 60 * 1000
        logger.info(f"[{symbol} {timeframe}] Backfill {since_days} días atrás.")

    total_saved = 0
    step_guard = timeframe_to_ms(timeframe)

    while since_ms < now_ms:
        batch = await fetch_ohlcv(symbol, timeframe, since_ms, REQUEST_LIMIT)
        if not batch:
            # Avanza al menos 1 vela para evitar bucle infinito si el exchange devuelve vacío
            since_ms += step_guard
            await asyncio.sleep(0.5)
            continue

        # Guardar en DB
        saved = save_batch_to_db(
            [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in batch],
            symbol, timeframe
        )
        total_saved += saved

        # Avanzar
        since_ms = batch[-1][0] + 1
        # Pequeño respiro para no saturar
        await asyncio.sleep(0.2)

    logger.info(f"[{symbol} {timeframe}] Guardadas ~{total_saved} velas nuevas.")

def read_symbols_and_tfs(config_path: str = CONFIG_PATH):
    """Lee símbolos y timeframes desde el archivo de configuración"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    symbols_cfg: Dict = cfg.get("symbols", {})
    
    result = []
    for sym, stg in symbols_cfg.items():
        tfs = stg.get("timeframes", default_tfs)
        ccxt_symbol = stg.get("ccxt_symbol") or sym
        result.append((ccxt_symbol, tfs))
    
    return result

async def run_all(config_path: str = CONFIG_PATH, since_days: int = DEFAULT_SINCE_DAYS):
    # Carga mercados una sola vez (necesario para Bitget y símbolos swap)
    await exchange.load_markets()

    symbols_tfs = read_symbols_and_tfs(config_path)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    async def worker(sym: str, tfs: List[str]):
        async with sem:
            for tf in tfs:
                await backfill_symbol_tf(sym, tf, since_days)

    for sym, tfs in symbols_tfs:
        tasks.append(worker(sym, tfs))

    await asyncio.gather(*tasks)
    logger.info("Descarga histórica completada para todos los símbolos/timeframes.")

async def main():
    try:
        await run_all()
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
