import os
import yaml
import pandas as pd
import numpy as np
import pickle
import gzip
import hashlib
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
from contextlib import contextmanager
from functools import lru_cache
import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, Table, Index
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings de pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

load_dotenv("config/.env")

# Configuración de cache
CACHE_DIR = Path("cache/datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 3600  # 1 hora en segundos
CACHE_MAX_SIZE = 100  # Máximo 100 archivos en cache

# Configuración de base de datos optimizada
DB_CONFIG = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'echo': False
}

# Crear engine con pool de conexiones
ENGINE = create_engine(
    os.getenv("DB_URL"),
    poolclass=QueuePool,
    **DB_CONFIG
)

# Features disponibles
FEATURES = [
    "rsi14", "ema20", "ema50", "ema200", "macd", "macd_signal", "macd_hist",
    "atr14", "bb_mid", "bb_upper", "bb_lower", "obv", "supertrend", "st_dir"
]

# Configuración de paginación
DEFAULT_PAGE_SIZE = 50000
MAX_PAGE_SIZE = 100000

# Configuración de compresión
COMPRESSION_LEVEL = 6  # Balance entre velocidad y compresión

class DatasetCache:
    """Sistema de cache inteligente para datasets"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, symbol: str, tf: str, params: Dict) -> str:
        """Generar clave de cache basada en parámetros"""
        key_data = f"{symbol}_{tf}_{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Obtener ruta del archivo de cache"""
        return self.cache_dir / f"{cache_key}.pkl.gz"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Verificar si el cache es válido"""
        if not cache_path.exists():
            return False
        
        # Verificar TTL
        mtime = cache_path.stat().st_mtime
        if time.time() - mtime > self.ttl:
            return False
        
        return True
    
    def get(self, symbol: str, tf: str, params: Dict) -> Optional[pd.DataFrame]:
        """Obtener dataset del cache"""
        cache_key = self._get_cache_key(symbol, tf, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit para {symbol}-{tf}")
            return data
        except Exception as e:
            logger.warning(f"Error cargando cache {cache_path}: {e}")
            return None
    
    def set(self, symbol: str, tf: str, params: Dict, data: pd.DataFrame):
        """Guardar dataset en cache"""
        cache_key = self._get_cache_key(symbol, tf, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with gzip.open(cache_path, 'wb', compresslevel=COMPRESSION_LEVEL) as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cache guardado para {symbol}-{tf}")
        except Exception as e:
            logger.warning(f"Error guardando cache {cache_path}: {e}")
    
    def cleanup(self):
        """Limpiar cache viejo"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            if len(cache_files) <= CACHE_MAX_SIZE:
                return
            
            # Ordenar por fecha de modificación (más antiguos primero)
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Eliminar archivos más antiguos
            files_to_remove = cache_files[:-CACHE_MAX_SIZE]
            for file_path in files_to_remove:
                file_path.unlink()
                logger.debug(f"Cache eliminado: {file_path}")
                
        except Exception as e:
            logger.warning(f"Error limpiando cache: {e}")

class DataValidator:
    """Validador de calidad de datos"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Dict[str, Any]:
        """Validar datos OHLCV"""
        issues = []
        
        # Verificar columnas requeridas
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Columnas faltantes: {missing_cols}")
        
        # Verificar valores nulos
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Valores nulos: {null_counts.to_dict()}")
        
        # Verificar rangos de precios
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("Precios de cierre no positivos")
            
            if (df['high'] < df['low']).any():
                issues.append("High < Low en algunos registros")
            
            if (df['high'] < df['close']).any():
                issues.append("High < Close en algunos registros")
            
            if (df['low'] > df['close']).any():
                issues.append("Low > Close en algunos registros")
        
        # Verificar volúmenes
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                issues.append("Volúmenes negativos")
        
        # Detectar gaps temporales
        if 'timestamp' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                large_gaps = time_diffs > median_diff * 10
                if large_gaps.any():
                    issues.append(f"Gaps temporales detectados: {large_gaps.sum()} registros")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_rows': len(df),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
    
    @staticmethod
    def validate_features(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Validar features"""
        issues = []
        
        # Verificar features disponibles
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            issues.append(f"Features faltantes: {missing_features}")
        
        # Verificar valores nulos en features
        if available_features:
            null_counts = df[available_features].isnull().sum()
            high_null_features = null_counts[null_counts > len(df) * 0.1]
            if not high_null_features.empty:
                issues.append(f"Features con muchos nulos: {high_null_features.to_dict()}")
        
        # Verificar rangos de features
        for feature in available_features:
            if df[feature].dtype in ['float64', 'int64']:
                if df[feature].isnull().all():
                    issues.append(f"Feature {feature} completamente nula")
                elif df[feature].nunique() == 1:
                    issues.append(f"Feature {feature} constante")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'available_features': available_features,
            'missing_features': missing_features
        }

# Instancia global del cache
dataset_cache = DatasetCache()

class OptimizedQueryBuilder:
    """Constructor de queries optimizadas"""
    
    def __init__(self, engine):
        self.engine = engine
        self._prepared_queries = {}
    
    def _get_base_query(self, features: List[str]) -> str:
        """Query base optimizada con índices compuestos"""
        return f"""
        SELECT h.symbol, h.timeframe, h.timestamp, h.open, h.high, h.low, h.close, h.volume,
               {', '.join('f.' + c for c in features)}
        FROM trading.historicaldata h
        JOIN trading.features f
          ON f.symbol = h.symbol 
         AND f.timeframe = h.timeframe 
         AND f.timestamp = h.timestamp
        WHERE h.symbol = :symbol AND h.timeframe = :tf
        ORDER BY h.timestamp
        """
    
    def _get_paginated_query(self, features: List[str], limit: int, offset: int) -> str:
        """Query paginada para datasets grandes"""
        return f"""
        SELECT h.symbol, h.timeframe, h.timestamp, h.open, h.high, h.low, h.close, h.volume,
               {', '.join('f.' + c for c in features)}
        FROM trading.historicaldata h
        JOIN trading.features f
          ON f.symbol = h.symbol 
         AND f.timeframe = h.timeframe 
         AND f.timestamp = h.timestamp
        WHERE h.symbol = :symbol AND h.timeframe = :tf
        ORDER BY h.timestamp
        LIMIT :limit OFFSET :offset
        """
    
    def _get_count_query(self) -> str:
        """Query para contar registros"""
        return """
        SELECT COUNT(*)
        FROM trading.historicaldata h
        JOIN trading.features f
          ON f.symbol = h.symbol 
         AND f.timeframe = h.timeframe 
         AND f.timestamp = h.timestamp
        WHERE h.symbol = :symbol AND h.timeframe = :tf
        """
    
    def _get_features_query(self, symbol: str, tf: str) -> str:
        """Query para features de timeframe específico"""
        return f"""
        SELECT timestamp, {', '.join(FEATURES)}
        FROM trading.features
        WHERE symbol = :symbol AND timeframe = :tf
        ORDER BY timestamp
        """
    
    def ensure_indexes(self):
        """Crear índices compuestos si no existen"""
        try:
            with self.engine.begin() as conn:
                # Índice para historicaldata
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_historicaldata_symbol_tf_ts 
                    ON trading.historicaldata (symbol, timeframe, timestamp)
                """))
                
                # Índice para features
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_symbol_tf_ts 
                    ON trading.features (symbol, timeframe, timestamp)
                """))
                
                logger.info("Índices compuestos creados/verificados")
        except Exception as e:
            logger.warning(f"Error creando índices: {e}")

class LazyDatasetLoader:
    """Cargador lazy para datasets grandes"""
    
    def __init__(self, engine, query_builder: OptimizedQueryBuilder):
        self.engine = engine
        self.query_builder = query_builder
    
    def load_chunked(self, symbol: str, tf: str, features: List[str], 
                    chunk_size: int = DEFAULT_PAGE_SIZE) -> Iterator[pd.DataFrame]:
        """Cargar dataset en chunks"""
        # Obtener total de registros
        count_query = self.query_builder._get_count_query()
        with self.engine.begin() as conn:
            total_rows = conn.execute(text(count_query), 
                                    {"symbol": symbol, "tf": tf}).scalar()
        
        if total_rows == 0:
            return
        
        logger.info(f"Cargando {total_rows} registros en chunks de {chunk_size}")
        
        # Cargar en chunks
        offset = 0
        while offset < total_rows:
            query = self.query_builder._get_paginated_query(features, chunk_size, offset)
            
            with self.engine.begin() as conn:
                chunk_df = pd.read_sql(
                    text(query), 
                    conn, 
                    params={"symbol": symbol, "tf": tf, "limit": chunk_size, "offset": offset}
                )
            
            if chunk_df.empty:
                break
            
            # Procesar chunk
            chunk_df = self._process_chunk(chunk_df)
            yield chunk_df
            
            offset += chunk_size
            logger.debug(f"Chunk cargado: {offset}/{total_rows} registros")
    
    def _process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesar chunk de datos"""
        # Convertir timestamp a datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Validar datos
        validator = DataValidator()
        validation_result = validator.validate_ohlcv(df)
        
        if not validation_result['valid']:
            logger.warning(f"Problemas de calidad en chunk: {validation_result['issues']}")
        
        return df

@lru_cache(maxsize=128)
def _read_cfg(path: str = "config/trading/symbols.yaml") -> Dict:
    """Leer configuración con cache"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def fetch_base_optimized(symbol: str, tf: str, 
                        use_cache: bool = True,
                        chunk_size: int = DEFAULT_PAGE_SIZE,
                        features: List[str] = None) -> pd.DataFrame:
    """Fetch optimizado con cache y paginación"""
    if features is None:
        features = FEATURES
    
    params = {
        'symbol': symbol,
        'timeframe': tf,
        'chunk_size': chunk_size,
        'features': tuple(features)
    }
    
    # Verificar cache
    if use_cache:
        cached_data = dataset_cache.get(symbol, tf, params)
        if cached_data is not None:
            return cached_data
    
    # Crear query builder y asegurar índices
    query_builder = OptimizedQueryBuilder(ENGINE)
    query_builder.ensure_indexes()
    
    # Cargar datos
    if chunk_size >= 100000:  # Usar lazy loading para datasets grandes
        loader = LazyDatasetLoader(ENGINE, query_builder)
        chunks = list(loader.load_chunked(symbol, tf, features, chunk_size))
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Carga directa para datasets pequeños
        query = query_builder._get_base_query(features)
        with ENGINE.begin() as conn:
            df = pd.read_sql(text(query), conn, params={"symbol": symbol, "tf": tf})
    
    if df.empty:
        return df
    
    # Procesar datos
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    
    # Validar datos
    validator = DataValidator()
    validation_result = validator.validate_ohlcv(df)
    feature_validation = validator.validate_features(df, features)
    
    if not validation_result['valid']:
        logger.warning(f"Problemas de calidad en {symbol}-{tf}: {validation_result['issues']}")
    
    if not feature_validation['valid']:
        logger.warning(f"Problemas en features {symbol}-{tf}: {feature_validation['issues']}")
    
    # Guardar en cache
    if use_cache:
        dataset_cache.set(symbol, tf, params, df)
    
    return df

def fetch_base(symbol: str, tf: str) -> pd.DataFrame:
    """Función de compatibilidad"""
    return fetch_base_optimized(symbol, tf, use_cache=True)

class SnapshotManager:
    """Gestor optimizado de snapshots multi-timeframe"""
    
    def __init__(self, engine, cache_dir: Path = CACHE_DIR):
        self.engine = engine
        self.cache_dir = cache_dir
        self.snapshot_cache = {}
    
    def _get_snapshot_cache_key(self, symbol: str, tf: str) -> str:
        """Clave de cache para snapshot"""
        return f"snapshot_{symbol}_{tf}"
    
    def _load_snapshot_from_db(self, symbol: str, tf: str) -> pd.DataFrame:
        """Cargar snapshot desde base de datos"""
        query = f"""
    SELECT timestamp, {', '.join(FEATURES)}
    FROM trading.features
        WHERE symbol = :symbol AND timeframe = :tf
    ORDER BY timestamp
        """
        
        with self.engine.begin() as conn:
            df = pd.read_sql(text(query), conn, params={"symbol": symbol, "tf": tf})
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp')
        
        return df
    
    def get_snapshot(self, symbol: str, tf: str, use_cache: bool = True) -> pd.DataFrame:
        """Obtener snapshot con cache"""
        cache_key = self._get_snapshot_cache_key(symbol, tf)
        
        if use_cache and cache_key in self.snapshot_cache:
            return self.snapshot_cache[cache_key].copy()
        
        # Cargar desde DB
        df = self._load_snapshot_from_db(symbol, tf)
        
        # Guardar en cache
        if use_cache:
            self.snapshot_cache[cache_key] = df.copy()
        
        return df
    
    def add_snapshots_optimized(self, df_base: pd.DataFrame, symbol: str, 
                               high_tfs: List[str]) -> pd.DataFrame:
        """Agregar snapshots de manera optimizada"""
        if df_base.empty:
            return df_base
        
        result_df = df_base.copy()
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], utc=True)
        result_df = result_df.sort_values('timestamp')
        
        for tf in high_tfs:
            snapshot_df = self.get_snapshot(symbol, tf)
            
            if snapshot_df.empty:
                logger.warning(f"No hay datos para snapshot {symbol}-{tf}")
                continue
            
            # Renombrar columnas del snapshot
            snapshot_renamed = snapshot_df.copy()
            for col in FEATURES:
                snapshot_renamed = snapshot_renamed.rename(columns={col: f"{col}_{tf}"})
            
            # Merge optimizado
            result_df = pd.merge_asof(
                result_df,
                snapshot_renamed,
                on="timestamp",
                direction="backward",
                allow_exact_matches=True
            )
            
            # Forward fill solo columnas del snapshot
            snapshot_cols = [f"{c}_{tf}" for c in FEATURES]
            result_df[snapshot_cols] = result_df[snapshot_cols].ffill()
            
            logger.debug(f"Snapshot {tf} agregado: {len(snapshot_df)} registros")
        
        return result_df

def add_snapshots_optimized(df_base: pd.DataFrame, symbol: str, 
                           high_tfs: List[str] = None) -> pd.DataFrame:
    """Agregar snapshots de manera optimizada"""
    if high_tfs is None:
        high_tfs = ["15m", "1h", "4h", "1d"]
    
    if df_base.empty:
        return df_base
    
    snapshot_manager = SnapshotManager(ENGINE)
    return snapshot_manager.add_snapshots_optimized(df_base, symbol, high_tfs)

def add_snapshots(df_base: pd.DataFrame, symbol: str, high_tf: str, suffix: str) -> pd.DataFrame:
    """Función de compatibilidad"""
    return add_snapshots_optimized(df_base, symbol, [high_tf])

def build_dataset_optimized(symbol: str, tf_base: str, 
                           use_snapshots: bool = True,
                           use_cache: bool = True,
                           chunk_size: int = DEFAULT_PAGE_SIZE,
                           features: List[str] = None) -> pd.DataFrame:
    """Construir dataset de manera optimizada"""
    if features is None:
        features = FEATURES
    
    params = {
        'symbol': symbol,
        'timeframe': tf_base,
        'use_snapshots': use_snapshots,
        'chunk_size': chunk_size,
        'features': tuple(features)
    }
    
    # Cargar datos base
    df = fetch_base_optimized(symbol, tf_base, use_cache, chunk_size, features)
    
    if df.empty:
        return df
    
    # Agregar snapshots si se solicita
    if use_snapshots:
        high_tfs = ["15m", "1h", "4h", "1d"]
        df = add_snapshots_optimized(df, symbol, high_tfs)
    
    return df

def build_dataset_streaming(symbol: str, tf_base: str, 
                           chunk_size: int = DEFAULT_PAGE_SIZE,
                           features: List[str] = None) -> Iterator[pd.DataFrame]:
    """Construir dataset con streaming para datasets muy grandes"""
    if features is None:
        features = FEATURES
    
    query_builder = OptimizedQueryBuilder(ENGINE)
    query_builder.ensure_indexes()
    
    loader = LazyDatasetLoader(ENGINE, query_builder)
    
    # Cargar chunks base
    for chunk in loader.load_chunked(symbol, tf_base, features, chunk_size):
        # Agregar snapshots al chunk
        chunk_with_snapshots = add_snapshots_optimized(chunk, symbol, ["15m", "1h", "4h", "1d"])
        yield chunk_with_snapshots

def cleanup_cache():
    """Limpiar cache viejo"""
    dataset_cache.cleanup()
    logger.info("Cache limpiado")

def get_cache_stats() -> Dict[str, Any]:
    """Obtener estadísticas del cache"""
    cache_files = list(CACHE_DIR.glob("*.pkl.gz"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        'total_files': len(cache_files),
        'total_size_mb': total_size / (1024 * 1024),
        'cache_dir': str(CACHE_DIR),
        'ttl_hours': CACHE_TTL / 3600
    }

def initialize_optimized_builder():
    """Inicializar builder optimizado"""
    logger.info("Inicializando builder optimizado...")
    
    # Crear directorios necesarios
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Crear índices en base de datos
    query_builder = OptimizedQueryBuilder(ENGINE)
    query_builder.ensure_indexes()
    
    # Limpiar cache viejo
    cleanup_cache()
    
    logger.info("Builder optimizado inicializado")

def build_dataset(symbol: str, tf_base: str, use_snapshots: bool = True) -> pd.DataFrame:
    """Función de compatibilidad"""
    return build_dataset_optimized(symbol, tf_base, use_snapshots=use_snapshots)

# Inicializar al importar
if __name__ != "__main__":
    initialize_optimized_builder()
