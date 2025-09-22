#!/usr/bin/env python3
"""
Sistema de inferencia en tiempo real de alta performance
"""

import os
import sys
import json
import asyncio
import time
import gc
import signal
import logging
import threading
import queue
import psutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Imports del proyecto
from core.ml.datasets.builder import FEATURES, build_dataset_optimized
from core.ml.utils.seeds import set_global_seeds

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/infer_realtime.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv("config/.env")

# Configuración de performance
PERFORMANCE_CONFIG = {
    'max_latency_ms': 5000,  # 5 segundos máximo
    'health_check_interval': 60,  # 1 minuto
    'cache_ttl_seconds': 300,  # 5 minutos
    'max_memory_mb': 2048,  # 2GB máximo
    'batch_size_initial': 10,
    'batch_size_max': 100,
    'workers_initial': 2,
    'workers_max': 8,
    'model_pool_size': 50,
    'feature_cache_size': 1000
}

@dataclass
class ModelInfo:
    """Información de un modelo cargado"""
    symbol: str
    timeframe: str
    horizon: int
    version: str
    model: Any
    scaler: Any
    feature_names: List[str]
    loaded_at: datetime
    last_used: datetime
    predictions_count: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0

@dataclass
class FeatureCache:
    """Cache de features calculadas"""
    symbol: str
    timeframe: str
    timestamp: datetime
    features: Dict[str, float]
    calculated_at: datetime

@dataclass
class PredictionResult:
    """Resultado de una predicción"""
    symbol: str
    timeframe: str
    horizon: int
    timestamp: datetime
    prediction: float
    confidence: float
    latency_ms: float
    model_version: str
    features_used: int

@dataclass
class PerformanceMetrics:
    """Métricas de performance del sistema"""
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    models_loaded: int = 0
    last_health_check: datetime = field(default_factory=datetime.now)

# Crear engine de base de datos
ENGINE = create_engine(
    os.getenv("DB_URL"),
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

class ModelPool:
    """Pool de modelos pre-cargados en memoria"""
    
    def __init__(self, max_size: int = 50):
        self.models: Dict[str, ModelInfo] = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.access_times: Dict[str, datetime] = {}
    
    def _get_model_key(self, symbol: str, timeframe: str, horizon: int) -> str:
        """Generar clave única para el modelo"""
        return f"{symbol}_{timeframe}_H{horizon}"
    
    def _get_latest_model_version(self, symbol: str, timeframe: str, horizon: int) -> Optional[str]:
        """Obtener la versión más reciente del modelo promovido"""
        try:
            query = text("""
                SELECT version, artifact_uri
                FROM trading.agentversions av
                JOIN trading.agents a ON av.agent_id = a.id
                WHERE a.name = 'DirectionLogReg' 
                  AND a.type = 'direction'
                  AND (av.params->>'symbol') = :symbol
                  AND (av.params->>'timeframe') = :timeframe
                  AND (av.params->>'horizon')::int = :horizon
                  AND av.promoted = true
                ORDER BY av.created_at DESC
                LIMIT 1
            """)
            
            with ENGINE.begin() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon
                }).fetchone()
                
                if result:
                    return result[0]  # version
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo versión del modelo {symbol}-{timeframe}: {e}")
            return None
    
    def _load_model_from_artifact(self, artifact_uri: str) -> Optional[Tuple[Any, Any, List[str]]]:
        """Cargar modelo desde archivo de artefacto"""
        try:
            if not os.path.exists(artifact_uri):
                logger.error(f"Artefacto no encontrado: {artifact_uri}")
                return None
            
            with open(artifact_uri, 'rb') as f:
                model_data = pickle.load(f)
            
            # Manejar diferentes formatos de modelo
            if isinstance(model_data, dict):
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                feature_names = model_data.get('feature_names', FEATURES)
            else:
                # Modelo directo (compatibilidad)
                model = model_data
                scaler = None
                feature_names = FEATURES
            
            return model, scaler, feature_names
            
        except Exception as e:
            logger.error(f"Error cargando modelo desde {artifact_uri}: {e}")
            return None
    
    def get_model(self, symbol: str, timeframe: str, horizon: int) -> Optional[ModelInfo]:
        """Obtener modelo del pool"""
        model_key = self._get_model_key(symbol, timeframe, horizon)
        
        with self.lock:
            if model_key in self.models:
                model_info = self.models[model_key]
                model_info.last_used = datetime.now()
                self.access_times[model_key] = datetime.now()
                return model_info
            
            # Cargar modelo si no está en el pool
            return self._load_model(symbol, timeframe, horizon)
    
    def _load_model(self, symbol: str, timeframe: str, horizon: int) -> Optional[ModelInfo]:
        """Cargar modelo en el pool"""
        model_key = self._get_model_key(symbol, timeframe, horizon)
        
        try:
            # Obtener versión más reciente
            version = self._get_latest_model_version(symbol, timeframe, horizon)
            if not version:
                logger.warning(f"No hay modelo promovido para {symbol}-{timeframe}-H{horizon}")
                return None
            
            # Obtener ruta del artefacto
            query = text("""
                SELECT artifact_uri
                FROM trading.agentversions av
                JOIN trading.agents a ON av.agent_id = a.id
                WHERE a.name = 'DirectionLogReg' 
                  AND a.type = 'direction'
                  AND (av.params->>'symbol') = :symbol
                  AND (av.params->>'timeframe') = :timeframe
                  AND (av.params->>'horizon')::int = :horizon
                  AND av.version = :version
                  AND av.promoted = true
            """)
            
            with ENGINE.begin() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon,
                    "version": version
                }).fetchone()
                
                if not result:
                    logger.error(f"No se encontró artefacto para {symbol}-{timeframe}-H{horizon}")
                    return None
                
                artifact_uri = result[0]
            
            # Cargar modelo
            model_data = self._load_model_from_artifact(artifact_uri)
            if not model_data:
                return None
            
            model, scaler, feature_names = model_data
            
            # Crear ModelInfo
            model_info = ModelInfo(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                version=version,
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                loaded_at=datetime.now(),
                last_used=datetime.now()
            )
            
            # Agregar al pool (con límite de tamaño)
            self._add_to_pool(model_key, model_info)
            
            logger.info(f"Modelo cargado: {symbol}-{timeframe}-H{horizon} v{version}")
            return model_info
            
        except Exception as e:
            logger.error(f"Error cargando modelo {symbol}-{timeframe}-H{horizon}: {e}")
            return None
    
    def _add_to_pool(self, model_key: str, model_info: ModelInfo):
        """Agregar modelo al pool respetando límite de tamaño"""
        with self.lock:
            # Si el pool está lleno, eliminar el menos usado
            if len(self.models) >= self.max_size:
                self._evict_least_used()
            
            self.models[model_key] = model_info
            self.access_times[model_key] = datetime.now()
    
    def _evict_least_used(self):
        """Eliminar el modelo menos usado del pool"""
        if not self.access_times:
            return
        
        # Encontrar el modelo con menor tiempo de acceso
        least_used_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Eliminar del pool
        if least_used_key in self.models:
            del self.models[least_used_key]
            del self.access_times[least_used_key]
            logger.debug(f"Modelo evictado del pool: {least_used_key}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pool de modelos"""
        with self.lock:
            return {
                'total_models': len(self.models),
                'max_size': self.max_size,
                'models': {
                    key: {
                        'symbol': info.symbol,
                        'timeframe': info.timeframe,
                        'horizon': info.horizon,
                        'version': info.version,
                        'predictions_count': info.predictions_count,
                        'avg_latency_ms': info.avg_latency_ms,
                        'error_count': info.error_count,
                        'loaded_at': info.loaded_at.isoformat(),
                        'last_used': info.last_used.isoformat()
                    }
                    for key, info in self.models.items()
                }
            }

class FeatureCacheManager:
    """Gestor de cache de features calculadas"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, FeatureCache] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, symbol: str, timeframe: str, timestamp: datetime) -> str:
        """Generar clave de cache para features"""
        # Redondear timestamp a minuto para cache
        rounded_ts = timestamp.replace(second=0, microsecond=0)
        return f"{symbol}_{timeframe}_{rounded_ts.isoformat()}"
    
    def get_features(self, symbol: str, timeframe: str, timestamp: datetime) -> Optional[Dict[str, float]]:
        """Obtener features del cache"""
        cache_key = self._get_cache_key(symbol, timeframe, timestamp)
        
        with self.lock:
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                
                # Verificar TTL
                if (datetime.now() - cache_entry.calculated_at).total_seconds() < self.ttl_seconds:
                    self.hit_count += 1
                    return cache_entry.features
                else:
                    # Cache expirado
                    del self.cache[cache_key]
            
            self.miss_count += 1
            return None
    
    def set_features(self, symbol: str, timeframe: str, timestamp: datetime, features: Dict[str, float]):
        """Guardar features en cache"""
        cache_key = self._get_cache_key(symbol, timeframe, timestamp)
        
        with self.lock:
            # Si el cache está lleno, eliminar entradas expiradas
            if len(self.cache) >= self.max_size:
                self._cleanup_expired()
            
            # Si aún está lleno, eliminar la más antigua
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].calculated_at)
                del self.cache[oldest_key]
            
            # Agregar nueva entrada
            self.cache[cache_key] = FeatureCache(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                features=features.copy(),
                calculated_at=datetime.now()
            )
    
    def _cleanup_expired(self):
        """Limpiar entradas expiradas del cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if (now - entry.calculated_at).total_seconds() >= self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_entries': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate_percent': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }

class RealtimeInferenceEngine:
    """Motor de inferencia en tiempo real de alta performance"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or PERFORMANCE_CONFIG
        self.model_pool = ModelPool(self.config['model_pool_size'])
        self.feature_cache = FeatureCacheManager(
            self.config['feature_cache_size'],
            self.config['cache_ttl_seconds']
        )
        self.metrics = PerformanceMetrics()
        self.running = False
        self.health_check_task = None
        self.prediction_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers = []
        self.lock = threading.RLock()
        
        # Configurar engine de base de datos
        self.engine = ENGINE
        
        # Configurar manejadores de señales
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Configurar manejadores de señales para shutdown graceful"""
        def signal_handler(signum, frame):
            logger.info(f"Señal {signum} recibida, iniciando shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Iniciar el motor de inferencia"""
        logger.info("🚀 Iniciando motor de inferencia en tiempo real...")
        
        self.running = True
        
        # Iniciar tareas asíncronas
        tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._prediction_processor()),
            asyncio.create_task(self._result_processor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tareas canceladas")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Cerrar el motor de inferencia"""
        logger.info("🛑 Cerrando motor de inferencia...")
        
        self.running = False
        
        # Cancelar tareas
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        for task in tasks:
            task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("✅ Motor de inferencia cerrado")
    
    async def _health_check_loop(self):
        """Loop de verificación de salud del sistema"""
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config['health_check_interval'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en health check: {e}")
                await asyncio.sleep(10)
    
    async def _perform_health_check(self):
        """Realizar verificación de salud del sistema"""
        try:
            # Verificar memoria
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
            
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.cpu_usage_percent = cpu_percent
            self.metrics.last_health_check = datetime.now()
            
            # Alertas de memoria
            if memory_mb > self.config['max_memory_mb']:
                logger.warning(f"⚠️  Uso de memoria alto: {memory_mb:.1f}MB")
                # Forzar garbage collection
                gc.collect()
            
            # Alertas de CPU
            if cpu_percent > 90:
                logger.warning(f"⚠️  Uso de CPU alto: {cpu_percent:.1f}%")
            
            # Verificar latencia promedio
            if self.metrics.avg_latency_ms > self.config['max_latency_ms']:
                logger.warning(f"⚠️  Latencia alta: {self.metrics.avg_latency_ms:.1f}ms")
            
            logger.debug(f"Health check: Mem={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, Lat={self.metrics.avg_latency_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
    
    async def _prediction_processor(self):
        """Procesador de predicciones"""
        while self.running:
            try:
                # Obtener solicitud de predicción
                prediction_request = await asyncio.wait_for(
                    self.prediction_queue.get(), 
                    timeout=1.0
                )
                
                # Procesar predicción
                await self._process_prediction(prediction_request)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error procesando predicción: {e}")
    
    async def _process_prediction(self, request: Dict[str, Any]):
        """Procesar una solicitud de predicción"""
        start_time = time.time()
        
        try:
            symbol = request['symbol']
            timeframe = request['timeframe']
            horizon = request['horizon']
            timestamp = request['timestamp']
            
            # Obtener modelo del pool
            model_info = self.model_pool.get_model(symbol, timeframe, horizon)
            if not model_info:
                raise Exception(f"No se pudo cargar modelo para {symbol}-{timeframe}-H{horizon}")
            
            # Obtener features (con cache)
            features = await self._get_features(symbol, timeframe, timestamp)
            if not features:
                raise Exception(f"No se pudieron obtener features para {symbol}-{timeframe}")
            
            # Preparar features para el modelo
            feature_array = self._prepare_features(features, model_info.feature_names)
            
            # Realizar predicción
            prediction, confidence, prediction_metadata = self._make_prediction(model_info, feature_array)
            
            # Calcular latencia
            latency_ms = (time.time() - start_time) * 1000
            
            # Actualizar métricas del modelo
            model_info.predictions_count += 1
            model_info.avg_latency_ms = (
                (model_info.avg_latency_ms * (model_info.predictions_count - 1) + latency_ms) 
                / model_info.predictions_count
            )
            
            # Crear resultado con metadatos detallados
            result = PredictionResult(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                timestamp=timestamp,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                model_version=model_info.version,
                features_used=len(features)
            )
            
            # Añadir metadatos detallados al resultado
            result.prediction_metadata = prediction_metadata
            
            # Enviar resultado
            await self.result_queue.put(result)
            
            # Actualizar métricas globales
            self._update_global_metrics(latency_ms, True)
            
        except Exception as e:
            logger.error(f"Error en predicción {request}: {e}")
            
            # Actualizar métricas de error
            if 'model_info' in locals():
                model_info.error_count += 1
            
            self._update_global_metrics((time.time() - start_time) * 1000, False)
    
    async def _get_features(self, symbol: str, timeframe: str, timestamp: datetime) -> Optional[Dict[str, float]]:
        """Obtener features con cache"""
        # Verificar cache primero
        cached_features = self.feature_cache.get_features(symbol, timeframe, timestamp)
        if cached_features:
            return cached_features
        
        # Calcular features si no están en cache
        try:
            # Obtener datos recientes para calcular features
            df = await self._get_recent_data(symbol, timeframe, timestamp)
            if df.empty:
                return None
            
            # Calcular features (simplificado para demo)
            features = self._calculate_features(df)
            
            # Guardar en cache
            self.feature_cache.set_features(symbol, timeframe, timestamp, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculando features para {symbol}-{timeframe}: {e}")
            return None
    
    async def _get_recent_data(self, symbol: str, timeframe: str, timestamp: datetime) -> pd.DataFrame:
        """Obtener datos recientes para calcular features"""
        try:
            query = text("""
                SELECT h.timestamp, h.open, h.high, h.low, h.close, h.volume,
                       f.rsi14, f.ema20, f.ema50, f.ema200, f.macd, f.macd_signal, f.macd_hist,
                       f.atr14, f.bb_mid, f.bb_upper, f.bb_lower, f.obv, f.supertrend, f.st_dir
                FROM trading.historicaldata h
                JOIN trading.features f ON f.symbol = h.symbol 
                    AND f.timeframe = h.timeframe 
                    AND f.timestamp = h.timestamp
                WHERE h.symbol = :symbol AND h.timeframe = :timeframe
                  AND h.timestamp <= :timestamp
                ORDER BY h.timestamp DESC
                LIMIT 100
            """)
            
            with self.engine.begin() as conn:
                df = pd.read_sql(
                    query, 
                    conn, 
                    params={"symbol": symbol, "timeframe": timeframe, "timestamp": timestamp}
                )
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos recientes: {e}")
            return pd.DataFrame()
    
    def _calculate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular features a partir de datos OHLCV"""
        if df.empty:
            return {}
        
        # Obtener la última fila (más reciente)
        latest = df.iloc[-1]
        
        features = {}
        for feature in FEATURES:
            if feature in latest:
                features[feature] = float(latest[feature])
            else:
                features[feature] = 0.0
        
        return features
    
    def _calculate_features_quality(self, feature_array: np.ndarray) -> float:
        """Calcular calidad de las features utilizadas"""
        if feature_array.size == 0:
            return 0.0
        
        # Calcular métricas de calidad
        n_samples, n_features = feature_array.shape
        
        # 1. Completitud: % de valores no nulos
        completeness = np.sum(~np.isnan(feature_array), axis=1) / n_features
        
        # 2. Estabilidad: desviación estándar de las features (menor = más estable)
        feature_stability = 1.0 / (1.0 + np.std(feature_array, axis=1))
        
        # 3. Rango: normalización del rango de valores
        feature_range = np.ptp(feature_array, axis=1) / (np.ptp(feature_array) + 1e-8)
        
        # 4. Outliers: % de valores dentro de 2 desviaciones estándar
        mean_vals = np.mean(feature_array, axis=1, keepdims=True)
        std_vals = np.std(feature_array, axis=1, keepdims=True)
        within_2std = np.sum(np.abs(feature_array - mean_vals) <= 2 * std_vals, axis=1) / n_features
        
        # Combinar métricas (promedio ponderado)
        quality_score = (
            0.3 * completeness +
            0.3 * feature_stability +
            0.2 * feature_range +
            0.2 * within_2std
        )
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def _prepare_features(self, features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
        """Preparar features para el modelo"""
        # Crear array en el orden correcto
        feature_array = np.array([features.get(name, 0.0) for name in feature_names])
        
        # Manejar valores nulos
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_array.reshape(1, -1)
    
    def _make_prediction(self, model_info: ModelInfo, feature_array: np.ndarray) -> Tuple[float, float, dict]:
        """Realizar predicción con el modelo y métricas detalladas"""
        try:
            model = model_info.model
            scaler = model_info.scaler
            start_time = time.time()
            
            # Escalar features si hay scaler
            if scaler is not None:
                feature_array = scaler.transform(feature_array)
            
            # Realizar predicción
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)
                prediction = proba[0][1]  # Probabilidad de clase positiva
                
                # Calcular confianza del modelo basada en la diferencia entre clases
                if proba.shape[1] > 1:
                    model_confidence = np.max(proba[0]) - np.min(proba[0])
                else:
                    model_confidence = abs(prediction - 0.5) * 2.0
                    
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(feature_array)
                prediction = 1 / (1 + np.exp(-decision[0]))  # Sigmoid
                model_confidence = abs(decision[0]) / (abs(decision[0]).max() + 1e-8)
            else:
                prediction = float(model.predict(feature_array)[0])
                model_confidence = abs(prediction - 0.5) * 2.0
            
            # Calcular confianza final
            confidence = abs(prediction - 0.5) * 2  # Confianza (0-1)
            
            # Calcular métricas adicionales
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Calcular calidad de features
            features_quality = self._calculate_features_quality(feature_array)
            
            # Crear metadatos detallados
            metadata = {
                "prob_up": float(prediction),
                "model_confidence": float(model_confidence),
                "features_quality": float(features_quality),
                "processing_time_ms": float(processing_time),
                "features_used": model_info.feature_names,
                "features_count": len(model_info.feature_names),
                "model_version": model_info.version,
                "is_fast_prediction": processing_time < 1000,
                "prediction_quality": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
            }
            
            return prediction, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error en predicción del modelo: {e}")
            return 0.5, 0.0, {"error": str(e)}  # Predicción neutral con error
    
    async def _result_processor(self):
        """Procesador de resultados"""
        while self.running:
            try:
                # Obtener resultado
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=1.0
                )
                
                # Procesar resultado (guardar en DB, enviar notificación, etc.)
                await self._handle_prediction_result(result)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error procesando resultado: {e}")
    
    async def _handle_prediction_result(self, result: PredictionResult):
        """Manejar resultado de predicción"""
        try:
            # Guardar en base de datos
            await self._save_prediction_to_db(result)
            
            # Log de resultado
            logger.info(
                f"Predicción: {result.symbol}-{result.timeframe} "
                f"pred={result.prediction:.3f} conf={result.confidence:.3f} "
                f"lat={result.latency_ms:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Error manejando resultado: {e}")
    
    async def _save_prediction_to_db(self, result: PredictionResult):
        """Guardar predicción en base de datos"""
        try:
            query = text("""
                INSERT INTO trading.agentpreds 
                (agent_version_id, symbol, timeframe, timestamp, horizon, payload, created_at)
                VALUES (
                    (SELECT av.id FROM trading.agentversions av
                     JOIN trading.agents a ON av.agent_id = a.id
                     WHERE a.name = 'DirectionLogReg' 
                       AND a.type = 'direction'
                       AND (av.params->>'symbol') = :symbol
                       AND (av.params->>'timeframe') = :timeframe
                       AND (av.params->>'horizon')::int = :horizon
                       AND av.version = :version
                       AND av.promoted = true
                     LIMIT 1),
                    :symbol, :timeframe, :timestamp, :horizon, :payload, :created_at
                )
            """)
            
            # Crear payload detallado con toda la información
            payload = {
                'prob_up': result.prediction,
                'confidence': result.confidence,
                'latency_ms': result.latency_ms,
                'features_used': result.features_used,
                'timestamp': result.timestamp.isoformat(),
                'model_version': result.model_version,
                'prediction_quality': getattr(result, 'prediction_metadata', {}).get('prediction_quality', 'unknown'),
                'is_fast_prediction': getattr(result, 'prediction_metadata', {}).get('is_fast_prediction', False)
            }
            
            # Añadir metadatos detallados si están disponibles
            if hasattr(result, 'prediction_metadata'):
                payload.update(result.prediction_metadata)
            
            with self.engine.begin() as conn:
                conn.execute(query, {
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
                    "timestamp": result.timestamp,
                    "horizon": result.horizon,
                    "version": result.model_version,
                    "payload": json.dumps(payload),
                    "created_at": datetime.now()
                })
                
        except Exception as e:
            logger.error(f"Error guardando predicción en DB: {e}")
    
    async def _metrics_collector(self):
        """Recolector de métricas del sistema"""
        while self.running:
            try:
                # Actualizar métricas
                self._update_system_metrics()
                
                # Log de métricas cada 5 minutos
                if self.metrics.total_predictions % 100 == 0 and self.metrics.total_predictions > 0:
                    logger.info(f"Métricas: pred={self.metrics.total_predictions} "
                              f"lat={self.metrics.avg_latency_ms:.1f}ms "
                              f"err={self.metrics.error_rate:.1f}% "
                              f"mem={self.metrics.memory_usage_mb:.1f}MB")
                
                await asyncio.sleep(30)  # Cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en collector de métricas: {e}")
    
    def _update_global_metrics(self, latency_ms: float, success: bool):
        """Actualizar métricas globales"""
        with self.lock:
            self.metrics.total_predictions += 1
            
            # Actualizar latencia promedio
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (self.metrics.total_predictions - 1) + latency_ms)
                / self.metrics.total_predictions
            )
            
            # Actualizar latencia máxima
            if latency_ms > self.metrics.max_latency_ms:
                self.metrics.max_latency_ms = latency_ms
            
            # Actualizar tasa de error
            if not success:
                error_count = int(self.metrics.error_rate * self.metrics.total_predictions / 100) + 1
                self.metrics.error_rate = (error_count / self.metrics.total_predictions) * 100
    
    def _update_system_metrics(self):
        """Actualizar métricas del sistema"""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics.cpu_usage_percent = process.cpu_percent()
            self.metrics.models_loaded = len(self.model_pool.models)
            
            # Calcular hit rate del cache
            cache_stats = self.feature_cache.get_cache_stats()
            self.metrics.cache_hit_rate = cache_stats['hit_rate_percent']
            
        except Exception as e:
            logger.error(f"Error actualizando métricas del sistema: {e}")
    
    async def predict(self, symbol: str, timeframe: str, horizon: int, timestamp: datetime = None):
        """Solicitar predicción"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        request = {
            'symbol': symbol,
            'timeframe': timeframe,
            'horizon': horizon,
            'timestamp': timestamp
        }
        
        await self.prediction_queue.put(request)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema"""
        with self.lock:
            return {
                'performance': {
                    'total_predictions': self.metrics.total_predictions,
                    'avg_latency_ms': self.metrics.avg_latency_ms,
                    'max_latency_ms': self.metrics.max_latency_ms,
                    'error_rate': self.metrics.error_rate,
                    'cache_hit_rate': self.metrics.cache_hit_rate
                },
                'system': {
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent,
                    'models_loaded': self.metrics.models_loaded,
                    'last_health_check': self.metrics.last_health_check.isoformat()
                },
                'model_pool': self.model_pool.get_pool_stats(),
                'feature_cache': self.feature_cache.get_cache_stats()
            }

# Funciones de utilidad
async def start_realtime_inference(config: Dict[str, Any] = None):
    """Iniciar el sistema de inferencia en tiempo real"""
    engine = RealtimeInferenceEngine(config)
    await engine.start()

def create_inference_engine(config: Dict[str, Any] = None) -> RealtimeInferenceEngine:
    """Crear instancia del motor de inferencia"""
    return RealtimeInferenceEngine(config)

# Función principal
async def main():
    """Función principal"""
    logger.info("🌙 INICIANDO SISTEMA DE INFERENCIA EN TIEMPO REAL")
    logger.info("=" * 60)
    
    # Configuración personalizada
    config = {
        'max_latency_ms': 3000,  # 3 segundos
        'health_check_interval': 30,  # 30 segundos
        'cache_ttl_seconds': 180,  # 3 minutos
        'max_memory_mb': 1024,  # 1GB
        'model_pool_size': 20,
        'feature_cache_size': 500
    }
    
    # Crear y iniciar motor
    engine = create_inference_engine(config)
    
    # Ejemplo de uso
    try:
        # Iniciar motor en background
        inference_task = asyncio.create_task(engine.start())
        
        # Esperar un poco para que se inicialice
        await asyncio.sleep(2)
        
        # Solicitar algunas predicciones de ejemplo
        await engine.predict("BTCUSDT", "1m", 1)
        await engine.predict("ETHUSDT", "5m", 3)
        await engine.predict("ADAUSDT", "1h", 5)
        
        # Esperar un poco para procesar
        await asyncio.sleep(5)
        
        # Mostrar métricas
        metrics = engine.get_metrics()
        logger.info(f"📊 MÉTRICAS: {json.dumps(metrics, indent=2, default=str)}")
        
        # Cancelar tarea
        inference_task.cancel()
        await inference_task
        
    except KeyboardInterrupt:
        logger.info("Interrupción recibida")
    except Exception as e:
        logger.error(f"Error en main: {e}")
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())