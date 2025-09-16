"""
core/data/realtime_updater.py

Actualiza la base de datos con velas recientes de Futuros (Bitget)
para TODOS los símbolos y timeframes definidos en config/trading/symbols.yaml.

✔ Lee DB_URL y claves Bitget de config/.env
✔ Para cada (symbol, timeframe):
    - Busca el último timestamp existente en trading.HistoricalData
    - Descarga desde (último_ts+1ms) hasta "ahora"
    - Inserta con ON CONFLICT DO NOTHING (requiere constraint 'unique_ohlcv')
✔ Evita velas incompletas (filtra la última si está en formación)
✔ Concurrencia controlada (asyncio.Semaphore)

Requisitos previos en DB (ejecutar una vez):
  ALTER TABLE trading.HistoricalData
    ADD CONSTRAINT IF NOT EXISTS unique_ohlcv UNIQUE(symbol,timeframe,timestamp);
  CREATE INDEX IF NOT EXISTS idx_hist_symbol_tf_ts
    ON trading.HistoricalData(symbol,timeframe,timestamp DESC);
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

# Límite CCXT
REQUEST_LIMIT = int(os.getenv("REQUEST_LIMIT", "1000"))
# Concurrencia simultánea de símbolos/timeframes
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "6"))
# Si NO hay datos previos en DB, retroceder (minutos) para primer arranque puntual
FALLBACK_LOOKBACK_MIN = int(os.getenv("FALLBACK_LOOKBACK_MIN", "1440"))  # 1 día

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
    return mapping.get(tf, 60_000)

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
    Requiere constraint UNIQUE(symbol,timeframe,timestamp) llamada 'unique_ohlcv'.
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

def filter_closed_candles(rows: List[List], timeframe: str) -> List[List]:
    """
    Filtra la última vela si pudiera estar en formación.
    Regla conservadora: elimina cualquier vela con ts >= now_ms - tf_ms.
    """
    if not rows:
        return rows
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    tf_ms = timeframe_to_ms(timeframe)
    cutoff = now_ms - tf_ms
    filtered = [c for c in rows if c[0] <= cutoff]
    return filtered

# --------------------------
# Catch-up principal
# --------------------------
async def catch_up_symbol_tf(symbol: str, timeframe: str, fallback_minutes: int = FALLBACK_LOOKBACK_MIN) -> int:
    """
    Recolecta datos desde el último timestamp de la DB hasta ahora (velas cerradas).
    Si no hay datos previos para ese (symbol, timeframe), retrocede fallback_minutes.
    Devuelve el número aproximado de velas guardadas.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    last_ts_ms = get_last_ts_ms(symbol, timeframe)

    if last_ts_ms:
        since_ms = last_ts_ms + 1
        logger.info(f"[{symbol} {timeframe}] Reanudando desde {last_ts_ms} (ms).")
    else:
        since_ms = now_ms - fallback_minutes * 60_000
        logger.info(f"[{symbol} {timeframe}] Sin datos previos. Retrocediendo {fallback_minutes} min.")

    total_saved = 0
    step_guard = timeframe_to_ms(timeframe)

    while since_ms < now_ms:
        batch = await fetch_ohlcv(symbol, timeframe, since_ms, REQUEST_LIMIT)
        if not batch:
            # Avanza 1 vela para evitar bucle si el exchange devuelve vacío
            since_ms += step_guard
            await asyncio.sleep(0.5)
            continue

        # Asegura insertar solo velas CERRADAS
        batch = filter_closed_candles(batch, timeframe)
        if not batch:
            # Si todo lo devuelto es "vela actual", esperamos y salimos
            break

        saved = save_batch_to_db(
            [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in batch],
            symbol, timeframe
        )
        total_saved += saved

        # Avanzar desde la última vela descargada
        since_ms = batch[-1][0] + 1

        # Respiro pequeño para no saturar
        await asyncio.sleep(0.2)

    logger.info(f"[{symbol} {timeframe}] Catch-up insertadas ~{total_saved} velas.")
    return total_saved

# --------------------------
# Runner para TODOS los símbolos/TFs del YAML
# --------------------------
async def run_catch_up_all(config_path: str = CONFIG_PATH):
    # Cargar mercados una sola vez
    await exchange.load_markets()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    default_tfs = cfg.get("defaults", {}).get("timeframes", ["1m"])
    symbols_cfg: Dict = cfg.get("symbols", {})

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    async def worker(ccxt_symbol: str, tfs: List[str]):
        async with sem:
            for tf in tfs:
                await catch_up_symbol_tf(ccxt_symbol, tf)

    for sym, stg in symbols_cfg.items():
        tfs = stg.get("timeframes", default_tfs)
        # Importante: usar símbolo CCXT correcto para Futuros USDT-M en Bitget (ej. "BTC/USDT:USDT")
        ccxt_symbol = stg.get("ccxt_symbol") or sym
        tasks.append(worker(ccxt_symbol, tfs))

    await asyncio.gather(*tasks)
    logger.info("Catch-up completado para todos los símbolos/timeframes.")

async def main():
    try:
        await run_catch_up_all()
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
