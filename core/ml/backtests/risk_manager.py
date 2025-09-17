# core/ml/backtests/risk_manager.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sqlalchemy import text
from core.data.database import get_engine

class RiskManager:
    """Gestor de riesgo dinámico que ajusta leverage basado en volatilidad y correlación"""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.engine = get_engine()
        
    def get_volatility_adjusted_leverage(self, base_leverage: float, atr: float, 
                                       close_price: float) -> float:
        """Ajusta leverage basado en volatilidad (ATR)"""
        if atr is None or atr <= 0:
            return base_leverage
            
        # Volatilidad relativa (ATR como % del precio)
        vol_pct = atr / close_price
        
        # Ajuste de leverage: más volatilidad = menos leverage
        if vol_pct > 0.05:  # >5% volatilidad
            vol_multiplier = 0.5
        elif vol_pct > 0.03:  # 3-5% volatilidad
            vol_multiplier = 0.7
        elif vol_pct > 0.02:  # 2-3% volatilidad
            vol_multiplier = 0.85
        else:  # <2% volatilidad
            vol_multiplier = 1.0
            
        return base_leverage * vol_multiplier
    
    def get_correlation_adjusted_leverage(self, base_leverage: float, 
                                        other_symbols: list) -> float:
        """Ajusta leverage basado en correlación con otros símbolos"""
        if not other_symbols:
            return base_leverage
            
        # Por ahora, implementación simple
        # En una versión más avanzada, calcularías correlación real
        correlation_penalty = 0.1  # 10% de reducción por símbolo correlacionado
        return base_leverage * (1 - len(other_symbols) * correlation_penalty)
    
    def get_market_regime_adjusted_leverage(self, base_leverage: float, 
                                          rsi: float, bb_position: float) -> float:
        """Ajusta leverage basado en régimen de mercado"""
        regime_multiplier = 1.0
        
        # RSI extremos = menos leverage
        if rsi > 80 or rsi < 20:
            regime_multiplier *= 0.6
        elif rsi > 70 or rsi < 30:
            regime_multiplier *= 0.8
            
        # Bollinger Bands extremos = menos leverage
        if bb_position > 0.9 or bb_position < 0.1:
            regime_multiplier *= 0.7
        elif bb_position > 0.8 or bb_position < 0.2:
            regime_multiplier *= 0.85
            
        return base_leverage * regime_multiplier
    
    def calculate_dynamic_leverage(self, base_leverage: float, 
                                 atr: float, close_price: float,
                                 rsi: float, bb_position: float,
                                 other_symbols: list = None) -> float:
        """Calcula leverage dinámico considerando todos los factores"""
        if other_symbols is None:
            other_symbols = []
            
        # Ajuste por volatilidad
        lev_vol = self.get_volatility_adjusted_leverage(base_leverage, atr, close_price)
        
        # Ajuste por correlación
        lev_corr = self.get_correlation_adjusted_leverage(lev_vol, other_symbols)
        
        # Ajuste por régimen de mercado
        lev_regime = self.get_market_regime_adjusted_leverage(lev_corr, rsi, bb_position)
        
        # Límites finales
        return max(1.0, min(lev_regime, 50.0))
    
    def get_bb_position(self, close_price: float, bb_upper: float, 
                       bb_lower: float) -> float:
        """Calcula posición dentro de Bollinger Bands (0-1)"""
        if bb_upper is None or bb_lower is None or bb_upper <= bb_lower:
            return 0.5  # Neutral si no hay datos
            
        return (close_price - bb_lower) / (bb_upper - bb_lower)
    
    def get_recent_volatility(self, days: int = 7) -> float:
        """Obtiene volatilidad reciente del símbolo"""
        with self.engine.begin() as conn:
            query = text("""
                SELECT AVG(atr14) as avg_atr
                FROM trading.features
                WHERE symbol = :symbol AND timeframe = :tf
                  AND timestamp >= NOW() - INTERVAL :days DAY
            """)
            result = conn.execute(query, {
                "symbol": self.symbol,
                "tf": self.timeframe,
                "days": days
            }).scalar()
            
        return result or 0.0
    
    def should_reduce_exposure(self) -> bool:
        """Determina si se debe reducir exposición basado en condiciones de mercado"""
        with self.engine.begin() as conn:
            # Verificar drawdown reciente
            query = text("""
                SELECT AVG(pnl) as avg_pnl
                FROM trading.backtesttrades
                WHERE symbol = :symbol AND timeframe = :tf
                  AND entry_ts >= NOW() - INTERVAL '24 hours'
            """)
            recent_pnl = conn.execute(query, {
                "symbol": self.symbol,
                "tf": self.timeframe
            }).scalar()
            
            # Si PnL reciente es muy negativo, reducir exposición
            if recent_pnl and recent_pnl < -100:  # -100 USDT de pérdida
                return True
                
        return False
