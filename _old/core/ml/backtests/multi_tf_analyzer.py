# core/ml/backtests/multi_tf_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sqlalchemy import text
from core.data.database import get_engine

class MultiTimeframeAnalyzer:
    """Analizador que combina señales de múltiples timeframes para mejorar decisiones"""
    
    def __init__(self, symbol: str, base_tf: str):
        self.symbol = symbol
        self.base_tf = base_tf
        self.engine = get_engine()
        
    def get_trend_direction(self, tf: str, timestamp: pd.Timestamp) -> int:
        """Obtiene dirección de tendencia para un timeframe específico"""
        with self.engine.begin() as conn:
            query = text("""
                SELECT ema20, ema50, ema200, supertrend, st_dir
                FROM trading.features
                WHERE symbol = :symbol AND timeframe = :tf
                  AND timestamp <= :ts
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = conn.execute(query, {
                "symbol": self.symbol,
                "tf": tf,
                "ts": timestamp
            }).mappings().first()
            
        if not result:
            return 0  # Neutral
            
        # Análisis de tendencia multi-criterio
        trend_score = 0
        
        # EMA alignment
        if result["ema20"] and result["ema50"] and result["ema200"]:
            if result["ema20"] > result["ema50"] > result["ema200"]:
                trend_score += 2  # Fuerte tendencia alcista
            elif result["ema20"] < result["ema50"] < result["ema200"]:
                trend_score -= 2  # Fuerte tendencia bajista
            elif result["ema20"] > result["ema50"]:
                trend_score += 1  # Tendencia alcista débil
            elif result["ema20"] < result["ema50"]:
                trend_score -= 1  # Tendencia bajista débil
                
        # SuperTrend
        if result["st_dir"]:
            trend_score += result["st_dir"]
            
        return np.sign(trend_score)
    
    def get_momentum_strength(self, tf: str, timestamp: pd.Timestamp) -> float:
        """Calcula fuerza del momentum para un timeframe"""
        with self.engine.begin() as conn:
            query = text("""
                SELECT rsi14, macd, macd_signal, macd_hist, obv
                FROM trading.features
                WHERE symbol = :symbol AND timeframe = :tf
                  AND timestamp <= :ts
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = conn.execute(query, {
                "symbol": self.symbol,
                "tf": tf,
                "ts": timestamp
            }).mappings().first()
            
        if not result:
            return 0.0
            
        momentum_score = 0.0
        
        # RSI momentum
        if result["rsi14"]:
            rsi = result["rsi14"]
            if rsi > 70:
                momentum_score += 0.3
            elif rsi > 60:
                momentum_score += 0.1
            elif rsi < 30:
                momentum_score -= 0.3
            elif rsi < 40:
                momentum_score -= 0.1
                
        # MACD momentum
        if result["macd"] and result["macd_signal"]:
            macd_diff = result["macd"] - result["macd_signal"]
            momentum_score += np.tanh(macd_diff * 10) * 0.2
            
        # MACD histogram
        if result["macd_hist"]:
            momentum_score += np.tanh(result["macd_hist"] * 10) * 0.1
            
        return np.clip(momentum_score, -1.0, 1.0)
    
    def get_volatility_regime(self, tf: str, timestamp: pd.Timestamp) -> str:
        """Determina régimen de volatilidad"""
        with self.engine.begin() as conn:
            query = text("""
                SELECT atr14, bb_upper, bb_lower, bb_mid
                FROM trading.features
                WHERE symbol = :symbol AND timeframe = :tf
                  AND timestamp <= :ts
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            result = conn.execute(query, {
                "symbol": self.symbol,
                "tf": tf,
                "ts": timestamp
            }).mappings().first()
            
        if not result or not result["atr14"]:
            return "normal"
            
        atr = result["atr14"]
        bb_width = result["bb_upper"] - result["bb_lower"] if result["bb_upper"] and result["bb_lower"] else None
        
        # Clasificación de volatilidad
        if atr > 0.05:  # >5% ATR
            return "high"
        elif atr < 0.01:  # <1% ATR
            return "low"
        else:
            return "normal"
    
    def analyze_multi_timeframe(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Análisis completo multi-timeframe"""
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        analysis = {
            "timestamp": timestamp,
            "trends": {},
            "momentum": {},
            "volatility": {},
            "consensus": 0,
            "strength": 0.0,
            "regime": "normal"
        }
        
        # Analizar cada timeframe
        for tf in timeframes:
            analysis["trends"][tf] = self.get_trend_direction(tf, timestamp)
            analysis["momentum"][tf] = self.get_momentum_strength(tf, timestamp)
            analysis["volatility"][tf] = self.get_volatility_regime(tf, timestamp)
            
        # Calcular consenso de tendencia
        trend_votes = [v for v in analysis["trends"].values() if v != 0]
        if trend_votes:
            analysis["consensus"] = np.sign(np.mean(trend_votes))
            
        # Calcular fuerza promedio del momentum
        momentum_values = [v for v in analysis["momentum"].values() if v != 0]
        if momentum_values:
            analysis["strength"] = np.mean(momentum_values)
            
        # Determinar régimen de volatilidad dominante
        vol_regimes = [v for v in analysis["volatility"].values() if v]
        if vol_regimes:
            analysis["regime"] = max(set(vol_regimes), key=vol_regimes.count)
            
        return analysis
    
    def get_tf_weight(self, tf: str) -> float:
        """Peso de cada timeframe en la decisión final"""
        weights = {
            "1m": 0.1,   # Ruido, peso bajo
            "5m": 0.2,   # Entrada, peso medio
            "15m": 0.3,  # Confirmación, peso alto
            "1h": 0.25,  # Tendencia, peso alto
            "4h": 0.1,   # Contexto, peso bajo
            "1d": 0.05   # Contexto, peso muy bajo
        }
        return weights.get(tf, 0.0)
    
    def calculate_final_signal(self, analysis: Dict[str, Any]) -> Tuple[int, float]:
        """Calcula señal final basada en análisis multi-timeframe"""
        consensus = analysis["consensus"]
        strength = analysis["strength"]
        
        # Ajustar fuerza basada en consenso
        if consensus != 0:
            # Más timeframes en la misma dirección = mayor fuerza
            trend_count = sum(1 for v in analysis["trends"].values() if v == consensus)
            total_tfs = len(analysis["trends"])
            consensus_ratio = trend_count / total_tfs
            
            # Ajustar fuerza por consenso
            strength *= (0.5 + consensus_ratio * 0.5)
            
        # Ajustar por régimen de volatilidad
        if analysis["regime"] == "high":
            strength *= 0.8  # Reducir fuerza en alta volatilidad
        elif analysis["regime"] == "low":
            strength *= 1.2  # Aumentar fuerza en baja volatilidad
            
        # Señal final
        signal = consensus if abs(strength) > 0.3 else 0  # Umbral mínimo
        final_strength = min(abs(strength), 1.0)
        
        return signal, final_strength
