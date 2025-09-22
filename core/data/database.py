"""
core/data/database.py
---------------------------------
Capa de acceso a datos para OHLCV.

Responsabilidades:
- Crear y mantener el engine de SQLAlchemy (sync).
- Upsert en batch sobre market.historical_data.
- Utilidades para leer el último timestamp disponible por (symbol, timeframe).
- Conversión robusta de OHLCV → filas para BD.

Requisitos:
- pip install sqlalchemy psycopg2-binary python-dotenv
- Definir DB_URL en config/.env (p.ej. postgresql+psycopg2://user:pass@host:5432/trading_db)
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple, Optional, Dict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Carga .env (si existe)
load_dotenv(os.path.join("config", ".env"))

DEFAULT_DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:160501@192.168.10.109:5432/trading_db")

logger = logging.getLogger("MarketDB")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)

# Map TF → ms
TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


@dataclass
class OHLCVRow:
    symbol: str
    timeframe: str
    ts: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDB:
    """Acceso a BD para escribir/leer OHLCV en market.historical_data."""

    def __init__(self, db_url: Optional[str] = None, pool_size: int = 10, max_overflow: int = 20) -> None:
        self.db_url = db_url or DEFAULT_DB_URL
        self.engine: Engine = create_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True,
        )
        logger.info(f"DB engine inicializado: {self.db_url}")

    # ---------- Lecturas ----------

    def get_max_ts(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Devuelve el máximo ts (UTC) almacenado para (symbol, timeframe); None si no hay datos."""
        sql = text(
            """
            SELECT MAX(ts) AS max_ts
            FROM market.historical_data
            WHERE symbol = :symbol AND timeframe = :timeframe
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe}).mappings().first()
            return row["max_ts"] if row and row["max_ts"] else None

    def get_min_ts(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Devuelve el mínimo ts (UTC) almacenado para (symbol, timeframe); None si no hay datos."""
        sql = text(
            """
            SELECT MIN(ts) AS min_ts
            FROM market.historical_data
            WHERE symbol = :symbol AND timeframe = :timeframe
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe}).mappings().first()
            return row["min_ts"] if row and row["min_ts"] else None

    def get_next_ts(self, symbol: str, timeframe: str, after_ts: datetime) -> Optional[datetime]:
        """Devuelve el siguiente ts después de after_ts para (symbol, timeframe); None si no hay más datos."""
        sql = text(
            """
            SELECT ts AS next_ts
            FROM market.historical_data
            WHERE symbol = :symbol AND timeframe = :timeframe AND ts > :after_ts
            ORDER BY ts ASC
            LIMIT 1
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe, "after_ts": after_ts}).mappings().first()
            return row["next_ts"] if row and row["next_ts"] else None

    def count_records_in_period(self, symbol: str, timeframe: str, start_ts: datetime, end_ts: datetime) -> int:
        """Cuenta registros en un período específico para (symbol, timeframe)."""
        sql = text(
            """
            SELECT COUNT(*) AS count
            FROM market.historical_data
            WHERE symbol = :symbol AND timeframe = :timeframe 
            AND ts >= :start_ts AND ts <= :end_ts
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(sql, {
                "symbol": symbol, 
                "timeframe": timeframe, 
                "start_ts": start_ts, 
                "end_ts": end_ts
            }).mappings().first()
            return row["count"] if row else 0

    # ---------- Escrituras ----------

    def upsert_ohlcv_batch(self, rows: Iterable[OHLCVRow], batch_size: int = 2000) -> int:
        """
        Inserta/actualiza OHLCV en lotes (ON CONFLICT).
        Retorna el número de filas insertadas/actualizadas.
        """
        rows_list = list(rows)
        if not rows_list:
            return 0

        sql = text(
            """
            INSERT INTO market.historical_data
                (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES
                (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume)
            ON CONFLICT (symbol, timeframe, ts) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """
        )

        total = 0
        with self.engine.begin() as conn:
            for i in range(0, len(rows_list), batch_size):
                chunk = rows_list[i : i + batch_size]
                # Mapea a dicts para SQLAlchemy
                payload = [
                    {
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "ts": r.ts,
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                    }
                    for r in chunk
                ]
                conn.execute(sql, payload)
                total += len(chunk)

        logger.info(f"Upsert OHLCV -> filas afectadas: {total}")
        return total

    # ---------- Helpers ----------

    @staticmethod
    def ccxt_ohlcv_to_rows(
        ccxt_ohlcv: List[List[float]],
        symbol: str,
        timeframe: str,
    ) -> List[OHLCVRow]:
        """
        Convierte listas CCXT [[ms, o, h, l, c, v], ...] a OHLCVRow (UTC).
        Ignora filas con NaN/None.
        """
        out: List[OHLCVRow] = []
        for rec in ccxt_ohlcv:
            try:
                ts_ms, o, h, l, c, v = rec[:6]
                if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), (o, h, l, c, v))):
                    continue
                ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                out.append(
                    OHLCVRow(
                        symbol=symbol,
                        timeframe=timeframe,
                        ts=ts_dt,
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v),
                    )
                )
            except Exception as e:
                logger.warning(f"Fila CCXT inválida (omitida): {rec} | err={e}")
        return out
