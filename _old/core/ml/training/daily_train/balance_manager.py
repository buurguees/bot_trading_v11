# core/ml/training/daily_train/balance_manager.py

import yaml
import os
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone

def load_symbol_config(symbol: str) -> Dict:
    """Carga configuración de balance para un símbolo específico"""
    # Prioridad: training.yaml > symbols.yaml > valores por defecto
    
    # 1. Intentar cargar desde training.yaml primero
    training_config = load_training_config()
    if training_config and "symbols" in training_config.get("balance", {}):
        symbol_balance = training_config["balance"]["symbols"].get(symbol)
        if symbol_balance:
            # Cargar leverage desde symbols.yaml
            symbols_config = load_symbols_config()
            symbol_cfg = symbols_config.get("symbols", {}).get(symbol, {})
            return {
                "initial": symbol_balance.get("initial", 1000.0),
                "target": symbol_balance.get("target", 10000.0),
                "risk_per_trade": symbol_balance.get("risk_per_trade", 0.02),
                "min_leverage": symbol_cfg.get("min_leverage", 3),
                "max_leverage": symbol_cfg.get("max_leverage", 50)
            }
    
    # 2. Fallback a symbols.yaml
    symbols_config = load_symbols_config()
    symbol_cfg = symbols_config.get("symbols", {}).get(symbol, {})
    balance_cfg = symbol_cfg.get("balance", {})
    
    # 3. Valores por defecto
    return {
        "initial": balance_cfg.get("initial", 1000.0),
        "target": balance_cfg.get("target", 10000.0),
        "risk_per_trade": balance_cfg.get("risk_per_trade", 0.02),
        "min_leverage": symbol_cfg.get("min_leverage", 3),
        "max_leverage": symbol_cfg.get("max_leverage", 50)
    }

def load_training_config() -> Dict:
    """Carga configuración desde training.yaml"""
    try:
        with open("config/ml/training.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_symbols_config() -> Dict:
    """Carga configuración desde symbols.yaml"""
    try:
        with open("config/trading/symbols.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def calculate_position_size(
    symbol: str, 
    current_price: float, 
    signal_strength: float,
    current_balance: float,
    leverage: float = None
) -> Tuple[float, float, float]:
    """
    Calcula el tamaño de posición basado en:
    - Balance actual
    - Fuerza de la señal
    - Riesgo por trade configurado
    - Apalancamiento dinámico
    
    Returns:
        (quantity, leverage_used, risk_amount)
    """
    config = load_symbol_config(symbol)
    
    # Balance objetivo y actual
    initial_balance = config["initial"]
    target_balance = config["target"]
    risk_per_trade = config["risk_per_trade"]
    
    # Calcular leverage dinámico basado en:
    # 1. Fuerza de la señal (0.0-1.0)
    # 2. Progreso hacia el objetivo
    # 3. Límites del símbolo
    
    min_lev = config["min_leverage"]
    max_lev = config["max_leverage"]
    
    # Progreso hacia objetivo (0.0 = inicial, 1.0 = objetivo)
    progress = min(1.0, (current_balance - initial_balance) / (target_balance - initial_balance))
    
    # Leverage base basado en fuerza de señal
    base_leverage = min_lev + (max_lev - min_lev) * signal_strength
    
    # Ajustar leverage según progreso:
    # - Si estamos cerca del objetivo, reducir leverage (más conservador)
    # - Si estamos lejos, usar leverage más alto
    if progress > 0.8:  # Cerca del objetivo
        leverage_multiplier = 0.7  # 30% menos leverage
    elif progress > 0.5:  # Mitad del camino
        leverage_multiplier = 0.9  # 10% menos leverage
    else:  # Lejos del objetivo
        leverage_multiplier = 1.0  # Leverage completo
    
    # Leverage final
    if leverage is None:
        leverage = base_leverage * leverage_multiplier
    leverage = max(min_lev, min(max_lev, leverage))
    
    # Cantidad de riesgo por trade
    risk_amount = current_balance * risk_per_trade
    
    # Calcular cantidad de la posición
    # quantity = (risk_amount * leverage) / current_price
    quantity = (risk_amount * leverage) / current_price
    
    return quantity, leverage, risk_amount

def calculate_dynamic_leverage(
    symbol: str,
    signal_strength: float,
    current_balance: float,
    recent_performance: float = None  # Sharpe ratio o similar
) -> float:
    """
    Calcula leverage dinámico basado en:
    - Fuerza de la señal
    - Balance actual vs objetivo
    - Rendimiento reciente
    """
    config = load_symbol_config(symbol)
    
    initial_balance = config["initial"]
    target_balance = config["target"]
    min_lev = config["min_leverage"]
    max_lev = config["max_leverage"]
    
    # Progreso hacia objetivo
    progress = min(1.0, (current_balance - initial_balance) / (target_balance - initial_balance))
    
    # Leverage base por fuerza de señal
    base_leverage = min_lev + (max_lev - min_lev) * signal_strength
    
    # Ajustes por progreso
    if progress > 0.9:  # Muy cerca del objetivo
        progress_multiplier = 0.5  # Muy conservador
    elif progress > 0.7:  # Cerca del objetivo
        progress_multiplier = 0.7  # Conservador
    elif progress > 0.3:  # Progreso medio
        progress_multiplier = 0.9  # Ligeramente conservador
    else:  # Lejos del objetivo
        progress_multiplier = 1.0  # Agresivo
    
    # Ajuste por rendimiento reciente
    performance_multiplier = 1.0
    if recent_performance is not None:
        if recent_performance > 2.0:  # Excelente rendimiento
            performance_multiplier = 1.1  # Aumentar leverage
        elif recent_performance > 1.5:  # Buen rendimiento
            performance_multiplier = 1.0  # Mantener
        elif recent_performance > 1.0:  # Rendimiento aceptable
            performance_multiplier = 0.9  # Reducir ligeramente
        else:  # Mal rendimiento
            performance_multiplier = 0.7  # Reducir significativamente
    
    # Leverage final
    final_leverage = base_leverage * progress_multiplier * performance_multiplier
    return max(min_lev, min(max_lev, final_leverage))

def get_balance_targets(symbol: str) -> Tuple[float, float]:
    """Obtiene balance inicial y objetivo para un símbolo"""
    config = load_symbol_config(symbol)
    return config["initial"], config["target"]

def should_reduce_risk(symbol: str, current_balance: float) -> bool:
    """Determina si se debe reducir el riesgo basado en el balance actual"""
    initial, target = get_balance_targets(symbol)
    progress = (current_balance - initial) / (target - initial)
    
    # Reducir riesgo si estamos cerca del objetivo o si hemos superado el objetivo
    return progress > 0.8

def get_risk_level(symbol: str, current_balance: float) -> str:
    """Obtiene el nivel de riesgo actual basado en el balance"""
    initial, target = get_balance_targets(symbol)
    progress = (current_balance - initial) / (target - initial)
    
    if progress >= 1.0:
        return "CONSERVATIVE"  # Objetivo alcanzado
    elif progress > 0.8:
        return "LOW"  # Cerca del objetivo
    elif progress > 0.5:
        return "MEDIUM"  # Progreso medio
    else:
        return "HIGH"  # Lejos del objetivo

def calculate_long_short_ratio(
    recent_trades: list,  # Lista de trades recientes con side (1/-1)
    target_ratio: float = 0.5  # 50% longs, 50% shorts
) -> Tuple[float, float]:
    """
    Calcula la proporción actual de longs/shorts y sugiere ajustes
    """
    if not recent_trades:
        return 0.5, 0.5  # Balanceado por defecto
    
    total_trades = len(recent_trades)
    long_trades = sum(1 for trade in recent_trades if trade.get('side', 0) == 1)
    short_trades = total_trades - long_trades
    
    current_long_ratio = long_trades / total_trades if total_trades > 0 else 0.5
    current_short_ratio = 1.0 - current_long_ratio
    
    # Ajustar para acercarse al objetivo
    if current_long_ratio > target_ratio + 0.1:  # Demasiados longs
        suggested_long_ratio = max(0.3, current_long_ratio - 0.1)
    elif current_long_ratio < target_ratio - 0.1:  # Demasiados shorts
        suggested_long_ratio = min(0.7, current_long_ratio + 0.1)
    else:
        suggested_long_ratio = current_long_ratio
    
    suggested_short_ratio = 1.0 - suggested_long_ratio
    
    return suggested_long_ratio, suggested_short_ratio
