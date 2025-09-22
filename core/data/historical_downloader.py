"""
core/data/historical_downloader.py
----------------------------------
Descarga ~2 años de histórico por símbolo/timeframe desde Bitget (CCXT)
y escribe en market.historical_data de forma idempotente.

Requisitos:
- pip install ccxt pyyaml python-dotenv sqlalchemy psycopg2-binary
- DB_URL en config/.env
- Símbolos/TFs en config/market/symbols.yaml
"""

from __future__ import annotations

import os
import time
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Iterable

import ccxt
import yaml
from dotenv import load_dotenv

# Import robusto para ejecución directa del archivo
try:
    from core.data.database import MarketDB
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    # Añade la raíz del proyecto al sys.path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from core.data.database import MarketDB

load_dotenv(os.path.join("config", ".env"))

logger = logging.getLogger("HistoricalDownloader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)

CONFIG_PATH = os.path.join("config", "market", "symbols.yaml")

# CCXT devuelve máx 1000 velas por llamada en la mayoría de mercados
CCXT_LIMIT = 1000

# Map TF → ms
TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def load_symbols_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_exchange() -> ccxt.bitget:
    api_key = os.getenv("BITGET_API_KEY") or ""
    secret = os.getenv("BITGET_API_SECRET") or ""
    password = os.getenv("BITGET_API_PASSPHRASE") or os.getenv("BITGET_API_PASSWORD") or ""

    ex = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # por defecto swap USDT
        "apiKey": api_key,
        "secret": secret,
        "password": password,  # requerido por bitget
    })
    return ex


def fetch_ohlcv_all(
    ex: ccxt.Exchange,
    db: MarketDB,
    ccxt_symbol: str,
    db_symbol: str,
    timeframe: str,
    years: int,
) -> int:
    """
    Descarga histórico (hasta 'years' atrás) respetando lo ya almacenado.
    Inserta en BD en lotes; retorna cantidad total de velas procesadas.
    """
    tf_ms = TF_MS[timeframe]
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    since_ms = now_ms - int(years * 365.25 * 24 * 60 * 60 * 1000)

    # Si ya hay datos, empezamos desde el último ts + 1*TF
    last_dt = db.get_max_ts(db_symbol, timeframe)
    if last_dt:
        last_ms = int(last_dt.timestamp() * 1000)
        since_ms = min(since_ms, last_ms)  # para no retroceder demasiado
        since_ms = last_ms + tf_ms

    logger.info(f"[{db_symbol}][{timeframe}] desde: {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)}")

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

        rows = MarketDB.ccxt_ohlcv_to_rows(batch, db_symbol, timeframe)
        inserted = db.upsert_ohlcv_batch(rows)
        total += inserted

        # Avanza el cursor
        last_batch_ms = batch[-1][0]
        cursor = last_batch_ms + tf_ms

        # Respeta rate limit
        time.sleep(ex.rateLimit / 1000.0)

        # Salida de seguridad para evitar bucles infinitos
        if len(batch) < CCXT_LIMIT:
            # Último tramo
            if cursor >= now_ms:
                break

    logger.info(f"[{db_symbol}][{timeframe}] total velas upsert: {total}")
    return total


def analyze_data_gaps(db: MarketDB, symbol: str, timeframe: str, days_back: int = 30) -> Dict:
    """
    Analiza gaps en los datos históricos de un símbolo/timeframe.
    
    Returns:
        Dict con información sobre gaps encontrados
    """
    tf_ms = TF_MS[timeframe]
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    # Obtener datos existentes
    since_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
    last_dt = db.get_max_ts(symbol, timeframe)
    first_dt = db.get_min_ts(symbol, timeframe)
    
    if not last_dt or not first_dt:
        return {
            "has_data": False,
            "gaps": [],
            "total_expected": 0,
            "total_actual": 0,
            "completeness_pct": 0.0
        }
    
    # Calcular registros esperados
    expected_records = days_back * 24 * 60 // (tf_ms // 1000 // 60)
    
    # Obtener gaps
    gaps = []
    cursor_ms = since_ms
    
    while cursor_ms < now_ms:
        # Buscar el siguiente registro
        next_dt = db.get_next_ts(symbol, timeframe, datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc))
        
        if next_dt:
            next_ms = int(next_dt.timestamp() * 1000)
            expected_next = cursor_ms + tf_ms
            
            if next_ms > expected_next + tf_ms:  # Gap detectado
                gap_start = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc)
                gap_end = datetime.fromtimestamp((next_ms - tf_ms) / 1000, tz=timezone.utc)
                gap_duration = (next_ms - expected_next) // tf_ms
                
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration_bars": gap_duration,
                    "duration_hours": gap_duration * (tf_ms / 1000 / 60 / 60)
                })
            
            cursor_ms = next_ms + tf_ms
        else:
            break
    
    # Contar registros actuales en el período
    actual_records = db.count_records_in_period(symbol, timeframe, since_dt, datetime.now(tz=timezone.utc))
    completeness_pct = (actual_records / expected_records * 100) if expected_records > 0 else 0.0
    
    return {
        "has_data": True,
        "gaps": gaps,
        "total_expected": expected_records,
        "total_actual": actual_records,
        "completeness_pct": completeness_pct,
        "first_date": first_dt,
        "last_date": last_dt
    }


def repair_data_gaps(ex: ccxt.Exchange, db: MarketDB, symbol: str, timeframe: str, gaps: List[Dict]) -> int:
    """
    Repara gaps específicos descargando datos faltantes.
    
    Returns:
        Número de registros descargados
    """
    if not gaps:
        return 0
    
    total_downloaded = 0
    
    for gap in gaps:
        logger.info(f"Reparando gap en {symbol} {timeframe}: {gap['start']} → {gap['end']}")
        
        start_ms = int(gap['start'].timestamp() * 1000)
        end_ms = int(gap['end'].timestamp() * 1000)
        
        cursor_ms = start_ms
        tf_ms = TF_MS[timeframe]
        
        while cursor_ms <= end_ms:
            try:
                batch = ex.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=cursor_ms,
                    limit=min(1000, (end_ms - cursor_ms) // tf_ms + 1)
                )
                
                if not batch:
                    break
                
                # Filtrar solo datos del gap
                gap_batch = [v for v in batch if start_ms <= v[0] <= end_ms]
                if gap_batch:
                    rows = MarketDB.ccxt_ohlcv_to_rows(gap_batch, symbol, timeframe)
                    downloaded = db.upsert_ohlcv_batch(rows)
                    total_downloaded += downloaded
                    logger.info(f"  Descargados {downloaded} registros")
                
                cursor_ms = batch[-1][0] + tf_ms
                time.sleep(ex.rateLimit / 1000.0)
                
            except ccxt.NetworkError as e:
                logger.warning(f"Error de red: {e}, reintentando...")
                time.sleep(5)
                continue
            except ccxt.BaseError as e:
                logger.error(f"Error CCXT: {e}")
                break
    
    return total_downloaded


def analyze_and_repair(symbol: str = None, timeframe: str = None, days_back: int = 30, 
                      min_completeness: float = 95.0, repair_gaps: bool = True) -> Dict:
    """
    Analiza y repara datos faltantes para símbolos/timeframes específicos o todos.
    
    Args:
        symbol: Símbolo específico (None = todos)
        timeframe: Timeframe específico (None = todos)
        days_back: Días hacia atrás para analizar
        min_completeness: Completitud mínima requerida (%)
        repair_gaps: Si True, descarga datos faltantes
    
    Returns:
        Dict con resumen de análisis y reparación
    """
    cfg = load_symbols_config(CONFIG_PATH)
    db = MarketDB()
    ex = make_exchange()
    ex.load_markets()
    
    results = {
        "analyzed": [],
        "repaired": [],
        "total_downloaded": 0,
        "errors": []
    }
    
    symbols_to_check = [s for s in cfg["symbols"] if symbol is None or s["id"] == symbol]
    
    for s in symbols_to_check:
        db_symbol = s["id"]
        ccxt_symbol = s["ccxt_symbol"]
        tfs = s.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
        
        if timeframe:
            tfs = [timeframe] if timeframe in tfs else []
        
        for tf in tfs:
            try:
                logger.info(f"Analizando {db_symbol} {tf}...")
                
                # Analizar gaps
                analysis = analyze_data_gaps(db, db_symbol, tf, days_back)
                results["analyzed"].append({
                    "symbol": db_symbol,
                    "timeframe": tf,
                    "completeness": analysis["completeness_pct"],
                    "gaps_count": len(analysis["gaps"]),
                    "has_data": analysis["has_data"]
                })
                
                # Si la completitud es baja o hay gaps, reparar
                if (analysis["completeness_pct"] < min_completeness or analysis["gaps"]) and repair_gaps:
                    logger.info(f"Reparando {db_symbol} {tf} (completitud: {analysis['completeness_pct']:.1f}%)")
                    
                    downloaded = repair_data_gaps(ex, db, db_symbol, tf, analysis["gaps"])
                    
                    results["repaired"].append({
                        "symbol": db_symbol,
                        "timeframe": tf,
                        "downloaded": downloaded,
                        "gaps_repaired": len(analysis["gaps"])
                    })
                    
                    results["total_downloaded"] += downloaded
                
            except Exception as e:
                error_msg = f"Error en {db_symbol} {tf}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
    
    return results


def run(years: int = None) -> None:
    cfg = load_symbols_config(CONFIG_PATH)
    years_cfg = cfg.get("default_years_history", 2)
    years_to_use = years if years is not None else years_cfg
    db = MarketDB()
    ex = make_exchange()
    ex.load_markets()

    for s in cfg["symbols"]:
        ccxt_symbol = s["ccxt_symbol"]   # ej. 'BTC/USDT:USDT' (swap) o 'BTC/USDT' (spot)
        db_symbol = s["id"]              # ej. 'BTCUSDT'
        tfs = s.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])

        for tf in tfs:
            logger.info(f"==> Descargando {db_symbol} {tf}")
            fetch_ohlcv_all(ex, db, ccxt_symbol, db_symbol, tf, years_to_use)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical Data Downloader")
    parser.add_argument("--mode", choices=["download", "analyze", "repair"], default="download",
                       help="Modo de operación: download (descarga completa), analyze (solo análisis), repair (análisis y reparación)")
    parser.add_argument("--symbol", type=str, help="Símbolo específico (ej: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe específico (ej: 1m)")
    parser.add_argument("--days", type=int, default=30, help="Días hacia atrás para analizar (default: 30)")
    parser.add_argument("--min-completeness", type=float, default=95.0, help="Completitud mínima requerida %% (default: 95.0)")
    parser.add_argument("--years", type=int, default=2, help="Años de datos para descarga completa (default: 2)")
    parser.add_argument("--no-repair", action="store_true", help="Solo analizar, no reparar gaps")
    
    args = parser.parse_args()
    
    if args.mode == "download":
        print("=== MODO: DESCARGA COMPLETA ===")
        run(years=args.years)
    elif args.mode == "analyze":
        print("=== MODO: ANÁLISIS SOLO ===")
        results = analyze_and_repair(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days_back=args.days,
            min_completeness=args.min_completeness,
            repair_gaps=False
        )
        print(f"\\n📊 RESUMEN DEL ANÁLISIS:")
        print(f"   Analizados: {len(results['analyzed'])} símbolos/timeframes")
        print(f"   Con problemas: {len([a for a in results['analyzed'] if a['completeness'] < args.min_completeness or a['gaps_count'] > 0])}")
        print(f"   Errores: {len(results['errors'])}")
    elif args.mode == "repair":
        print("=== MODO: ANÁLISIS Y REPARACIÓN ===")
        results = analyze_and_repair(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days_back=args.days,
            min_completeness=args.min_completeness,
            repair_gaps=not args.no_repair
        )
        print(f"\\n📊 RESUMEN DE REPARACIÓN:")
        print(f"   Analizados: {len(results['analyzed'])} símbolos/timeframes")
        print(f"   Reparados: {len(results['repaired'])} símbolos/timeframes")
        print(f"   Total descargado: {results['total_downloaded']} registros")
        print(f"   Errores: {len(results['errors'])}")
        
        if results['errors']:
            print("\\n❌ ERRORES:")
            for error in results['errors']:
                print(f"   {error}")
