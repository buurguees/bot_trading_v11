#!/usr/bin/env python3
"""
Procesador de Señales de Trading
Convierte predicciones de agentpreds a señales de trading en agentsignals
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Importar funciones de postprocesamiento
from core.ml.inference.postprocess import (
    prob_to_side, strength_from_prob, calculate_confidence,
    calculate_signal_quality, should_generate_signal, extract_signal_metadata
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/signal_processor.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv("config/.env")

@dataclass
class SignalConfig:
    """Configuración del procesador de señales"""
    batch_size: int = 1000
    lookback_minutes: int = 5
    min_confidence: float = 0.6
    min_strength: float = 0.1
    side_threshold: float = 0.5
    filters: List[str] = None
    strength_calculation: str = "abs(prob - 0.5)"
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = ["duplicate_filter", "time_filter", "confidence_filter"]

@dataclass
class PredictionData:
    """Datos de una predicción"""
    id: int
    agent_version_id: int
    symbol: str
    timeframe: str
    timestamp: datetime
    horizon: int
    payload: Dict[str, Any]
    created_at: datetime

@dataclass
class SignalData:
    """Datos de una señal generada"""
    symbol: str
    timeframe: str
    timestamp: datetime
    side: int  # 1 para long, -1 para short
    strength: float  # 0-1
    meta: Dict[str, Any]
    agent_version_id: int

class SignalProcessor:
    """Procesador principal de señales de trading"""
    
    def __init__(self, config_path: str = "config/signals/signal_processing.yaml"):
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.processed_predictions = set()  # Para evitar duplicados
        
    def _load_config(self, config_path: str) -> SignalConfig:
        """Cargar configuración desde archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            signal_config = config_data.get('signal_processing', {})
            
            return SignalConfig(
                batch_size=signal_config.get('batch_size', 1000),
                lookback_minutes=signal_config.get('lookback_minutes', 5),
                min_confidence=signal_config.get('min_confidence', 0.6),
                min_strength=signal_config.get('min_strength', 0.1),
                side_threshold=signal_config.get('side_threshold', 0.5),
                filters=signal_config.get('filters', ["duplicate_filter", "time_filter"]),
                strength_calculation=signal_config.get('strength_calculation', "abs(prob - 0.5)")
            )
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return SignalConfig()  # Configuración por defecto
    
    def _create_engine(self):
        """Crear engine de base de datos"""
        db_url = os.getenv("DB_URL")
        if not db_url:
            raise ValueError("DB_URL no está definido en .env")
        
        return create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    
    def get_recent_predictions(
        self,
        symbol: str = None,
        timeframe: str = None,
        lookback_minutes: int = None,
        limit: int = None
    ) -> List[PredictionData]:
        """Obtener predicciones recientes de agentpreds"""
        if lookback_minutes is None:
            lookback_minutes = self.config.lookback_minutes
        if limit is None:
            limit = self.config.batch_size
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        
        query = text("""
            SELECT id, agent_version_id, symbol, timeframe, timestamp, horizon, payload, created_at
            FROM trading.agentpreds
            WHERE created_at >= :cutoff_time
            AND id NOT IN (SELECT unnest(ARRAY[:processed_ids]))
        """)
        
        params = {
            "cutoff_time": cutoff_time,
            "processed_ids": list(self.processed_predictions) if self.processed_predictions else [0]
        }
        
        if symbol:
            query = text(str(query) + " AND symbol = :symbol")
            params["symbol"] = symbol
        
        if timeframe:
            query = text(str(query) + " AND timeframe = :timeframe")
            params["timeframe"] = timeframe
        
        query = text(str(query) + " ORDER BY created_at DESC LIMIT :limit")
        params["limit"] = limit
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                
                predictions = []
                for row in result:
                    predictions.append(PredictionData(
                        id=row[0],
                        agent_version_id=row[1],
                        symbol=row[2],
                        timeframe=row[3],
                        timestamp=row[4],
                        horizon=row[5],
                        payload=row[6] if isinstance(row[6], dict) else json.loads(row[6]),
                        created_at=row[7]
                    ))
                
                logger.info(f"Obtenidas {len(predictions)} predicciones recientes")
                return predictions
                
        except Exception as e:
            logger.error(f"Error obteniendo predicciones: {e}")
            return []
    
    def convert_prediction_to_signal(
        self,
        prediction: PredictionData,
        rules: Dict[str, Any] = None
    ) -> Optional[SignalData]:
        """Convertir una predicción a señal de trading usando postprocess.py"""
        try:
            # Extraer datos del payload
            prob_up = prediction.payload.get('prob_up', 0.5)
            model_confidence = prediction.payload.get('model_confidence')
            features_quality = prediction.payload.get('features_quality')
            market_volatility = prediction.payload.get('market_volatility')
            processing_time_ms = prediction.payload.get('processing_time_ms')
            features_used = prediction.payload.get('features_used', [])
            model_version = prediction.payload.get('model_version')
            
            # Usar postprocess.py para calcular métricas de calidad
            quality_metrics = calculate_signal_quality(
                prob_up, model_confidence, features_quality, market_volatility
            )
            
            # Determinar si se debe generar señal usando postprocess.py
            should_signal = should_generate_signal(
                prob_up, 
                quality_metrics,
                min_quality=self.config.min_confidence,
                min_confidence=self.config.min_confidence,
                min_strength=self.config.min_strength
            )
            
            if not should_signal:
                logger.debug(f"Predicción {prediction.id} no cumple criterios de calidad para generar señal")
                return None
            
            # Usar postprocess.py para determinar lado y fuerza
            side = prob_to_side(prob_up, buy_th=0.55, sell_th=0.45)
            if side == 0:
                return None  # No generar señal si está en zona neutral
            
            strength = strength_from_prob(prob_up)
            
            # Extraer metadatos detallados usando postprocess.py
            signal_metadata = extract_signal_metadata(
                prediction.payload,
                features_used=features_used,
                model_version=model_version,
                processing_time_ms=processing_time_ms
            )
            
            # Crear metadatos finales
            meta = {
                "direction_ver_id": prediction.agent_version_id,
                "prob_up": prob_up,
                "confidence": quality_metrics["confidence"],
                "strength": strength,
                "horizon": prediction.horizon,
                "quality_metrics": quality_metrics,
                "signal_metadata": signal_metadata,
                "model_version": model_version,
                "features_used": features_used,
                "features_count": len(features_used) if features_used else 0,
                "processing_time_ms": processing_time_ms,
                "is_high_quality": quality_metrics["is_high_quality"],
                "is_clear_signal": quality_metrics["is_clear_signal"]
            }
            
            # Añadir metadatos adicionales si están configurados
            if rules and rules.get('include_latency', False):
                meta["latency_ms"] = processing_time_ms
            
            if rules and rules.get('include_features_count', False):
                meta["features_used"] = features_used
                meta["features_count"] = len(features_used) if features_used else 0
            
            return SignalData(
                symbol=prediction.symbol,
                timeframe=prediction.timeframe,
                timestamp=prediction.timestamp,
                side=side,
                strength=strength,
                meta=meta,
                agent_version_id=prediction.agent_version_id
            )
            
        except Exception as e:
            logger.error(f"Error convirtiendo predicción {prediction.id}: {e}")
            return None
    
    
    def apply_signal_filters(
        self,
        signals: List[SignalData],
        filters: List[str] = None
    ) -> List[SignalData]:
        """Aplicar filtros a las señales"""
        if filters is None:
            filters = self.config.filters
        
        filtered_signals = signals.copy()
        
        for filter_name in filters:
            if filter_name == "duplicate_filter":
                filtered_signals = self._filter_duplicates(filtered_signals)
            elif filter_name == "time_filter":
                filtered_signals = self._filter_time_gaps(filtered_signals)
            elif filter_name == "confidence_filter":
                filtered_signals = self._filter_confidence(filtered_signals)
            elif filter_name == "strength_filter":
                filtered_signals = self._filter_strength(filtered_signals)
        
        logger.info(f"Filtros aplicados: {len(signals)} -> {len(filtered_signals)} señales")
        return filtered_signals
    
    def _filter_duplicates(self, signals: List[SignalData]) -> List[SignalData]:
        """Filtrar señales duplicadas por (symbol, timeframe, timestamp)"""
        seen = set()
        filtered = []
        
        for signal in signals:
            key = (signal.symbol, signal.timeframe, signal.timestamp)
            if key not in seen:
                seen.add(key)
                filtered.append(signal)
        
        return filtered
    
    def _filter_time_gaps(self, signals: List[SignalData], min_gap_minutes: int = 1) -> List[SignalData]:
        """Filtrar señales muy cercanas en el tiempo"""
        if not signals:
            return signals
        
        # Ordenar por timestamp
        sorted_signals = sorted(signals, key=lambda x: x.timestamp)
        filtered = [sorted_signals[0]]  # Primera señal siempre se incluye
        
        for signal in sorted_signals[1:]:
            last_signal = filtered[-1]
            time_gap = (signal.timestamp - last_signal.timestamp).total_seconds() / 60
            
            if time_gap >= min_gap_minutes:
                filtered.append(signal)
        
        return filtered
    
    def _filter_confidence(self, signals: List[SignalData]) -> List[SignalData]:
        """Filtrar señales por confianza mínima"""
        return [s for s in signals if s.meta.get('confidence', 0) >= self.config.min_confidence]
    
    def _filter_strength(self, signals: List[SignalData]) -> List[SignalData]:
        """Filtrar señales por fuerza mínima"""
        return [s for s in signals if s.strength >= self.config.min_strength]
    
    def save_signals_to_db(
        self,
        signals: List[SignalData],
        batch_size: int = None
    ) -> int:
        """Guardar señales en la base de datos"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if not signals:
            return 0
        
        saved_count = 0
        
        try:
            with self.engine.begin() as conn:
                # Procesar en lotes
                for i in range(0, len(signals), batch_size):
                    batch = signals[i:i + batch_size]
                    
                    # Preparar datos para inserción
                    rows = []
                    for signal in batch:
                        rows.append({
                            "symbol": signal.symbol,
                            "timeframe": signal.timeframe,
                            "timestamp": signal.timestamp,
                            "side": signal.side,
                            "strength": signal.strength,
                            "meta": json.dumps(signal.meta),
                            "agent_version_id": signal.agent_version_id
                        })
                    
                    # Insertar lote
                    conn.execute(text("""
                        INSERT INTO trading.agentsignals 
                        (symbol, timeframe, timestamp, side, strength, meta, agent_version_id)
                        VALUES (:symbol, :timeframe, :timestamp, :side, :strength, :meta::jsonb, :agent_version_id)
                        ON CONFLICT (symbol, timeframe, timestamp)
                        DO UPDATE SET 
                            side = EXCLUDED.side,
                            strength = EXCLUDED.strength,
                            meta = EXCLUDED.meta,
                            agent_version_id = EXCLUDED.agent_version_id
                    """), rows)
                    
                    saved_count += len(batch)
                    logger.debug(f"Guardado lote de {len(batch)} señales")
                
                logger.info(f"Guardadas {saved_count} señales en la base de datos")
                return saved_count
                
        except Exception as e:
            logger.error(f"Error guardando señales: {e}")
            return 0
    
    def process_predictions_to_signals(
        self,
        symbol: str = None,
        timeframe: str = None,
        lookback_minutes: int = None,
        batch_size: int = None
    ) -> Dict[str, int]:
        """Procesar predicciones y generar señales"""
        start_time = time.time()
        
        try:
            # Obtener predicciones
            predictions = self.get_recent_predictions(
                symbol=symbol,
                timeframe=timeframe,
                lookback_minutes=lookback_minutes,
                batch_size=batch_size
            )
            
            if not predictions:
                logger.info("No hay predicciones nuevas para procesar")
                return {"predictions": 0, "signals": 0, "errors": 0}
            
            # Convertir predicciones a señales
            signals = []
            errors = 0
            
            for prediction in predictions:
                try:
                    signal = self.convert_prediction_to_signal(prediction)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error procesando predicción {prediction.id}: {e}")
                    errors += 1
            
            # Aplicar filtros
            filtered_signals = self.apply_signal_filters(signals)
            
            # Guardar señales
            saved_count = self.save_signals_to_db(filtered_signals)
            
            # Marcar predicciones como procesadas
            for prediction in predictions:
                self.processed_predictions.add(prediction.id)
            
            # Calcular métricas
            processing_time = time.time() - start_time
            conversion_rate = len(signals) / len(predictions) if predictions else 0
            filter_rate = len(filtered_signals) / len(signals) if signals else 0
            
            logger.info(
                f"Procesamiento completado: "
                f"predicciones={len(predictions)}, "
                f"señales={len(signals)}, "
                f"filtradas={len(filtered_signals)}, "
                f"guardadas={saved_count}, "
                f"errores={errors}, "
                f"tiempo={processing_time:.2f}s, "
                f"conversión={conversion_rate:.2%}, "
                f"filtro={filter_rate:.2%}"
            )
            
            return {
                "predictions": len(predictions),
                "signals": len(signals),
                "filtered_signals": len(filtered_signals),
                "saved_signals": saved_count,
                "errors": errors,
                "processing_time": processing_time,
                "conversion_rate": conversion_rate,
                "filter_rate": filter_rate
            }
            
        except Exception as e:
            logger.error(f"Error en procesamiento principal: {e}")
            return {"predictions": 0, "signals": 0, "errors": 1}
    
    def process_batch_predictions(
        self,
        start_time: datetime,
        end_time: datetime,
        symbols: List[str] = None
    ) -> Dict[str, int]:
        """Procesar predicciones en lote para un rango de tiempo"""
        logger.info(f"Procesamiento en lote: {start_time} - {end_time}")
        
        # Calcular minutos de lookback
        lookback_minutes = int((end_time - start_time).total_seconds() / 60)
        
        total_stats = {"predictions": 0, "signals": 0, "errors": 0}
        
        if symbols:
            for symbol in symbols:
                stats = self.process_predictions_to_signals(
                    symbol=symbol,
                    lookback_minutes=lookback_minutes
                )
                for key, value in stats.items():
                    total_stats[key] = total_stats.get(key, 0) + value
        else:
            total_stats = self.process_predictions_to_signals(
                lookback_minutes=lookback_minutes
            )
        
        return total_stats
    
    def process_realtime_predictions(
        self,
        lookback_minutes: int = None
    ) -> Dict[str, int]:
        """Procesar predicciones en tiempo real"""
        if lookback_minutes is None:
            lookback_minutes = self.config.lookback_minutes
        
        return self.process_predictions_to_signals(
            lookback_minutes=lookback_minutes
        )

# Funciones de utilidad
def create_signal_processor(config_path: str = "config/signals/signal_processing.yaml") -> SignalProcessor:
    """Crear instancia del procesador de señales"""
    return SignalProcessor(config_path)

def process_signals_realtime(config_path: str = "config/signals/signal_processing.yaml"):
    """Función de conveniencia para procesamiento en tiempo real"""
    processor = create_signal_processor(config_path)
    return processor.process_realtime_predictions()

def process_signals_batch(
    start_time: datetime,
    end_time: datetime,
    symbols: List[str] = None,
    config_path: str = "config/signals/signal_processing.yaml"
):
    """Función de conveniencia para procesamiento en lote"""
    processor = create_signal_processor(config_path)
    return processor.process_batch_predictions(start_time, end_time, symbols)

# Función principal
def main():
    """Función principal para ejecución desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Procesador de Señales de Trading")
    parser.add_argument("--mode", choices=["realtime", "batch"], default="realtime",
                       help="Modo de procesamiento")
    parser.add_argument("--symbol", help="Símbolo específico a procesar")
    parser.add_argument("--timeframe", help="Timeframe específico a procesar")
    parser.add_argument("--lookback", type=int, help="Minutos de lookback")
    parser.add_argument("--config", default="config/signals/signal_processing.yaml",
                       help="Ruta del archivo de configuración")
    
    args = parser.parse_args()
    
    # Crear procesador
    processor = create_signal_processor(args.config)
    
    if args.mode == "realtime":
        # Procesamiento en tiempo real
        stats = processor.process_realtime_predictions(
            lookback_minutes=args.lookback
        )
        print(f"Procesamiento en tiempo real completado: {stats}")
        
    elif args.mode == "batch":
        # Procesamiento en lote
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Última hora por defecto
        
        stats = processor.process_batch_predictions(
            start_time=start_time,
            end_time=end_time,
            symbols=[args.symbol] if args.symbol else None
        )
        print(f"Procesamiento en lote completado: {stats}")

if __name__ == "__main__":
    main()
