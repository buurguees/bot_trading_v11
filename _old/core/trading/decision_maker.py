# core/trading/decision_maker.py
"""
Sistema de Toma de Decisiones Inteligente para Trading

Integra todas las señales y componentes existentes del sistema:
- Lee AgentSignals de la base de datos (generadas por inference)
- Aplica filtros multi-timeframe para confirmación
- Usa balance_manager.py para gestión de riesgo por símbolo
- Genera planes usando planner.py existente
- Coordina con order_executor.py para ejecución

FLUJO PRINCIPAL:
1. Lee señales ML de trading.agentsignals
2. Aplica confirmación multi-timeframe (15m, 1h, 4h, 1d)
3. Valida con balance_manager y risk_manager
4. Genera planes usando planner.py
5. Ejecuta usando order_executor.py
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import yaml

# Importar componentes existentes
from .planner import plan_and_store
from .order_executor import OrderExecutor, create_training_executor, create_live_executor
from ..ml.training.daily_train.balance_manager import load_symbol_config, calculate_dynamic_leverage
from .position_sizer import load_risk_params, plan_from_price

load_dotenv("config/.env")

@dataclass
class MarketCondition:
    """Estado del mercado para análisis multi-timeframe"""
    symbol: str
    timestamp: datetime
    base_tf: str
    trend_15m: int  # -1, 0, 1
    trend_1h: int
    trend_4h: int  
    trend_1d: int
    strength_15m: float  # 0.0 - 1.0
    strength_1h: float
    strength_4h: float
    strength_1d: float
    volatility: float
    rsi_1h: float
    is_trending: bool
    confidence_score: float

@dataclass
class TradingDecision:
    """Decisión final de trading"""
    symbol: str
    timeframe: str
    timestamp: datetime
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    side: int    # 1, -1, 0
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasons: List[str]
    market_condition: MarketCondition
    risk_params: Dict
    plan_params: Optional[Dict] = None
    should_execute: bool = True

class DecisionMaker:
    """
    Sistema central de toma de decisiones que coordina todos los componentes
    
    RESPONSABILIDADES:
    - Leer y procesar señales ML de trading.agentsignals
    - Análisis multi-timeframe para confirmación 
    - Gestión de riesgo usando balance_manager existente
    - Coordinación con order_executor para ejecución
    - Logging completo de decisiones para auditoría
    """
    
    def __init__(self, mode: str = "training", symbols: List[str] = None):
        self.mode = mode.lower()
        self.logger = logging.getLogger(__name__)
        
        # Símbolos a gestionar
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
        
        # DB connection
        self.engine = create_engine(os.getenv("DB_URL"))
        
        # Executor para ejecución de órdenes
        if self.mode == "training":
            self.executor = create_training_executor()
        else:
            self.executor = create_live_executor()
            
        # Configuración de decisiones
        self.config = self._load_decision_config()
        
        # Cache para evitar recálculos
        self._market_conditions_cache = {}
        
        # Estadísticas de decisiones
        self.decision_stats = {
            "total_signals": 0,
            "signals_processed": 0,
            "decisions_made": 0,
            "decisions_executed": 0,
            "confirmations_passed": 0,
            "confirmations_failed": 0
        }
        
        self.logger.info(f"DecisionMaker initialized in {mode} mode for {len(self.symbols)} symbols")
        
    def _load_decision_config(self) -> Dict:
        """Carga configuración de decisiones desde YAML"""
        try:
            with open("config/trading/decisions.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            config = {}
            
        # Configuración por defecto
        default_config = {
            "confirmations": {
                "require_multi_tf": True,
                "min_timeframes_agree": 2,  # Al menos 2 TF deben confirmar
                "timeframes_weight": {
                    "15m": 0.2,
                    "1h": 0.3, 
                    "4h": 0.3,
                    "1d": 0.2
                }
            },
            "filters": {
                "min_signal_strength": 0.6,
                "min_confidence": 0.7,
                "max_volatility": 0.15,
                "rsi_overbought": 75,
                "rsi_oversold": 25
            },
            "risk": {
                "max_concurrent_positions": 3,
                "max_risk_per_trade": 0.02,
                "correlation_threshold": 0.7
            }
        }
        
        # Merge con defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config[key]}
                
        return config
        
    def process_signals_cycle(self, lookback_minutes: int = 10) -> List[TradingDecision]:
        """
        Procesa un ciclo completo de señales y genera decisiones
        
        Args:
            lookback_minutes: Minutos hacia atrás para buscar señales nuevas
        """
        decisions = []
        
        try:
            # 1. Cargar señales ML recientes de todos los símbolos
            signals = self._load_recent_signals(lookback_minutes)
            self.decision_stats["total_signals"] += len(signals)
            
            if not signals:
                self.logger.debug("No new signals to process")
                return decisions
                
            self.logger.info(f"Processing {len(signals)} new signals")
            
            # 2. Procesar cada señal
            for signal in signals:
                try:
                    decision = self._process_single_signal(signal)
                    if decision:
                        decisions.append(decision)
                        self.decision_stats["decisions_made"] += 1
                        
                    self.decision_stats["signals_processed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal {signal.get('id', 'unknown')}: {e}")
                    
            # 3. Ejecutar decisiones válidas
            executed_count = self._execute_decisions(decisions)
            self.decision_stats["decisions_executed"] += executed_count
            
            # 4. Log resumen
            if decisions:
                executed = sum(1 for d in decisions if d.should_execute)
                self.logger.info(f"Generated {len(decisions)} decisions, {executed} for execution, {executed_count} executed")
                
        except Exception as e:
            self.logger.error(f"Error in signals processing cycle: {e}")
            
        return decisions
        
    def _load_recent_signals(self, lookback_minutes: int) -> List[Dict]:
        """Carga señales ML recientes desde trading.agentsignals"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        
        with self.engine.connect() as conn:
            query = text("""
                SELECT 
                    id, symbol, timeframe, timestamp, side, strength,
                    sl, tp, meta, created_at
                FROM trading.agentsignals 
                WHERE created_at > :cutoff_time
                  AND symbol = ANY(:symbols)
                  AND side != 0  -- Solo señales con dirección
                ORDER BY created_at DESC
            """)
            
            result = conn.execute(query, {
                "cutoff_time": cutoff_time,
                "symbols": self.symbols
            })
            
            return [dict(row._mapping) for row in result]
            
    def _process_single_signal(self, signal: Dict) -> Optional[TradingDecision]:
        """
        Procesa una señal individual y genera decisión
        
        Pasos:
        1. Análisis multi-timeframe
        2. Validación de filtros
        3. Cálculo de confianza
        4. Generación de decisión
        """
        symbol = signal['symbol']
        timeframe = signal['timeframe']
        timestamp = signal['timestamp']
        
        try:
            # 1. Análizar condiciones multi-timeframe
            market_condition = self._analyze_market_conditions(symbol, timestamp)
            
            # 2. Aplicar confirmación multi-timeframe
            confirmation_result = self._apply_multi_tf_confirmation(signal, market_condition)
            
            if not confirmation_result['passed']:
                self.decision_stats["confirmations_failed"] += 1
                self.logger.debug(f"Signal {signal['id']} failed multi-TF confirmation: {confirmation_result['reason']}")
                return None
                
            self.decision_stats["confirmations_passed"] += 1
            
            # 3. Validar filtros de calidad
            filter_result = self._apply_signal_filters(signal, market_condition)
            
            if not filter_result['passed']:
                self.logger.debug(f"Signal {signal['id']} failed filters: {filter_result['reason']}")
                return None
                
            # 4. Calcular confianza final
            confidence = self._calculate_confidence(signal, market_condition, confirmation_result)
            
            # 5. Validar balance y riesgo
            risk_validation = self._validate_risk_constraints(symbol, signal, market_condition)
            
            if not risk_validation['allowed']:
                self.logger.debug(f"Signal {signal['id']} failed risk validation: {risk_validation['reason']}")
                return None
                
            # 6. Generar decisión final
            decision = TradingDecision(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                action="BUY" if signal['side'] == 1 else "SELL",
                side=signal['side'],
                strength=signal['strength'],
                confidence=confidence,
                reasons=confirmation_result['reasons'] + filter_result['reasons'],
                market_condition=market_condition,
                risk_params=risk_validation['params'],
                should_execute=True
            )
            
            self.logger.info(
                f"DECISION: {symbol} {decision.action} | "
                f"Strength: {decision.strength:.2f} | "
                f"Confidence: {confidence:.2f} | "
                f"Reasons: {len(decision.reasons)}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
            return None
            
    def _analyze_market_conditions(self, symbol: str, timestamp: datetime) -> MarketCondition:
        """
        Analiza condiciones multi-timeframe usando features existentes
        Integra con trading.features tabla
        """
        cache_key = f"{symbol}_{timestamp.isoformat()}"
        if cache_key in self._market_conditions_cache:
            return self._market_conditions_cache[cache_key]
            
        try:
            with self.engine.connect() as conn:
                # Obtener features de múltiples timeframes
                timeframes = ["15m", "1h", "4h", "1d"]
                tf_data = {}
                
                for tf in timeframes:
                    query = text("""
                        SELECT 
                            ema20, ema50, ema200, rsi14, supertrend, st_dir, 
                            atr14, bb_upper, bb_lower, bb_mid, macd, macd_signal
                        FROM trading.features 
                        WHERE symbol = :symbol AND timeframe = :tf 
                          AND timestamp <= :ts
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """)
                    
                    result = conn.execute(query, {
                        "symbol": symbol,
                        "tf": tf, 
                        "ts": timestamp
                    }).mappings().first()
                    
                    if result:
                        tf_data[tf] = dict(result)
                        
            # Análisis de tendencia por timeframe
            trends = {}
            strengths = {}
            
            for tf in timeframes:
                if tf in tf_data:
                    data = tf_data[tf]
                    trends[tf] = self._calculate_trend_direction(data)
                    strengths[tf] = self._calculate_trend_strength(data)
                else:
                    trends[tf] = 0
                    strengths[tf] = 0.0
                    
            # Volatilidad basada en ATR 
            volatility = 0.0
            if "1h" in tf_data and tf_data["1h"]["atr14"]:
                # ATR normalizado como proxy de volatilidad
                atr = tf_data["1h"]["atr14"]
                if "1h" in tf_data and tf_data["1h"]["bb_mid"]:
                    volatility = min(1.0, atr / tf_data["1h"]["bb_mid"])
                    
            # RSI de 1h para contexto
            rsi_1h = tf_data.get("1h", {}).get("rsi14", 50) or 50
            
            # Determinar si está en tendencia
            trend_agreement = sum(1 for t in trends.values() if abs(t) > 0)
            is_trending = trend_agreement >= 2 and volatility < 0.1
            
            # Confidence score basado en alineación de timeframes
            confidence = self._calculate_multi_tf_confidence(trends, strengths)
            
            condition = MarketCondition(
                symbol=symbol,
                timestamp=timestamp,
                base_tf="1m",  # Asumimos base de 1m
                trend_15m=trends.get("15m", 0),
                trend_1h=trends.get("1h", 0),
                trend_4h=trends.get("4h", 0),
                trend_1d=trends.get("1d", 0),
                strength_15m=strengths.get("15m", 0.0),
                strength_1h=strengths.get("1h", 0.0),
                strength_4h=strengths.get("4h", 0.0),
                strength_1d=strengths.get("1d", 0.0),
                volatility=volatility,
                rsi_1h=rsi_1h,
                is_trending=is_trending,
                confidence_score=confidence
            )
            
            # Cache result
            self._market_conditions_cache[cache_key] = condition
            return condition
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {symbol}: {e}")
            # Return neutral condition
            return MarketCondition(
                symbol=symbol, timestamp=timestamp, base_tf="1m",
                trend_15m=0, trend_1h=0, trend_4h=0, trend_1d=0,
                strength_15m=0.0, strength_1h=0.0, strength_4h=0.0, strength_1d=0.0,
                volatility=0.05, rsi_1h=50, is_trending=False, confidence_score=0.5
            )
            
    def _calculate_trend_direction(self, features: Dict) -> int:
        """Calcula dirección de tendencia basada en features técnicos"""
        score = 0
        
        # EMA alignment
        if all(features.get(k) for k in ["ema20", "ema50", "ema200"]):
            ema20, ema50, ema200 = features["ema20"], features["ema50"], features["ema200"]
            if ema20 > ema50 > ema200:
                score += 2  # Fuerte alcista
            elif ema20 < ema50 < ema200:
                score -= 2  # Fuerte bajista
            elif ema20 > ema50:
                score += 1  # Alcista débil
            elif ema20 < ema50:
                score -= 1  # Bajista débil
                
        # SuperTrend
        if features.get("st_dir") is not None:
            score += int(features["st_dir"])
            
        # MACD
        if all(features.get(k) for k in ["macd", "macd_signal"]):
            if features["macd"] > features["macd_signal"]:
                score += 1
            else:
                score -= 1
                
        # Normalizar a -1, 0, 1
        if score >= 2:
            return 1
        elif score <= -2:
            return -1
        else:
            return 0
            
    def _calculate_trend_strength(self, features: Dict) -> float:
        """Calcula fuerza de tendencia (0.0 - 1.0)"""
        strength_factors = []
        
        # EMA separation (mayor separación = mayor fuerza)
        if all(features.get(k) for k in ["ema20", "ema50", "ema200"]):
            ema20, ema50, ema200 = features["ema20"], features["ema50"], features["ema200"]
            separation = abs(ema20 - ema200) / ema200 if ema200 > 0 else 0
            strength_factors.append(min(1.0, separation * 20))  # Normalize
            
        # ATR relative to price (mayor ATR = mayor movimiento)
        if features.get("atr14") and features.get("bb_mid"):
            atr_ratio = features["atr14"] / features["bb_mid"]
            strength_factors.append(min(1.0, atr_ratio * 100))
            
        # RSI distance from 50 (mayor distancia = mayor momentum)
        if features.get("rsi14"):
            rsi_strength = abs(features["rsi14"] - 50) / 50
            strength_factors.append(rsi_strength)
            
        return np.mean(strength_factors) if strength_factors else 0.5
        
    def _calculate_multi_tf_confidence(self, trends: Dict, strengths: Dict) -> float:
        """Calcula confianza basada en alineación multi-timeframe"""
        weights = self.config["confirmations"]["timeframes_weight"]
        
        # Alineación de tendencias (mismo signo)
        trend_values = [trends.get(tf, 0) for tf in weights.keys()]
        if not trend_values:
            return 0.5
            
        # Calcular consenso ponderado
        weighted_trend = sum(trends.get(tf, 0) * weight for tf, weight in weights.items())
        consensus = abs(weighted_trend)
        
        # Calcular fuerza promedio ponderada
        weighted_strength = sum(strengths.get(tf, 0) * weight for tf, weight in weights.items())
        
        # Combinar consenso y fuerza
        confidence = (consensus * 0.6 + weighted_strength * 0.4)
        return min(1.0, max(0.0, confidence))
        
    def _apply_multi_tf_confirmation(self, signal: Dict, condition: MarketCondition) -> Dict:
        """Aplica lógica de confirmación multi-timeframe"""
        if not self.config["confirmations"]["require_multi_tf"]:
            return {"passed": True, "reasons": ["Multi-TF confirmation disabled"]}
            
        signal_side = signal['side']
        min_agree = self.config["confirmations"]["min_timeframes_agree"]
        
        # Contar timeframes que están de acuerdo
        agreeing_tfs = []
        if condition.trend_15m * signal_side > 0:
            agreeing_tfs.append("15m")
        if condition.trend_1h * signal_side > 0:
            agreeing_tfs.append("1h")  
        if condition.trend_4h * signal_side > 0:
            agreeing_tfs.append("4h")
        if condition.trend_1d * signal_side > 0:
            agreeing_tfs.append("1d")
            
        passed = len(agreeing_tfs) >= min_agree
        
        reason = (f"Multi-TF confirmation: {len(agreeing_tfs)}/{min_agree} required TFs agree ({agreeing_tfs})" 
                 if passed else f"Multi-TF confirmation failed: only {len(agreeing_tfs)}/{min_agree} TFs agree")
        
        return {
            "passed": passed,
            "reasons": [reason],
            "agreeing_timeframes": agreeing_tfs,
            "agreement_count": len(agreeing_tfs)
        }
        
    def _apply_signal_filters(self, signal: Dict, condition: MarketCondition) -> Dict:
        """Aplica filtros de calidad de señal"""
        filters = self.config["filters"]
        reasons = []
        
        # Filtro de fuerza mínima
        if signal['strength'] < filters["min_signal_strength"]:
            return {
                "passed": False, 
                "reasons": [f"Signal strength {signal['strength']:.2f} < {filters['min_signal_strength']}"]
            }
            
        # Filtro de volatilidad máxima
        if condition.volatility > filters["max_volatility"]:
            return {
                "passed": False,
                "reasons": [f"Volatility {condition.volatility:.3f} > {filters['max_volatility']}"]
            }
            
        # Filtro RSI (evitar extremos)
        rsi = condition.rsi_1h
        if signal['side'] == 1 and rsi > filters["rsi_overbought"]:
            return {
                "passed": False,
                "reasons": [f"RSI overbought: {rsi:.1f} > {filters['rsi_overbought']}"]
            }
        elif signal['side'] == -1 and rsi < filters["rsi_oversold"]:
            return {
                "passed": False,
                "reasons": [f"RSI oversold: {rsi:.1f} < {filters['rsi_oversold']}"]
            }
            
        reasons.append(f"Signal filters passed: strength={signal['strength']:.2f}, vol={condition.volatility:.3f}, rsi={rsi:.1f}")
        
        return {"passed": True, "reasons": reasons}
        
    def _calculate_confidence(self, signal: Dict, condition: MarketCondition, confirmation: Dict) -> float:
        """Calcula confianza final combinando todos los factores"""
        # Base: confianza del mercado
        confidence = condition.confidence_score * 0.4
        
        # Fuerza de la señal ML
        confidence += signal['strength'] * 0.3
        
        # Confirmación multi-timeframe  
        tf_agreement = confirmation['agreement_count'] / 4.0  # Max 4 TFs
        confidence += tf_agreement * 0.2
        
        # Bonus por tendencia clara
        if condition.is_trending:
            confidence += 0.1
            
        return min(1.0, max(0.0, confidence))
        
    def _validate_risk_constraints(self, symbol: str, signal: Dict, condition: MarketCondition) -> Dict:
        """Valida restricciones de riesgo usando balance_manager"""
        try:
            # Cargar configuración del símbolo usando balance_manager existente
            symbol_config = load_symbol_config(symbol)
            
            # Obtener balance actual del executor
            portfolio_status = self.executor.get_portfolio_status()
            current_balance = portfolio_status['symbol_balances'][symbol]['current']
            
            # Verificar límites de riesgo
            risk_per_trade = symbol_config["risk_per_trade"]
            max_risk = self.config["risk"]["max_risk_per_trade"]
            
            if risk_per_trade > max_risk:
                return {
                    "allowed": False,
                    "reason": f"Risk per trade {risk_per_trade} > max {max_risk}"
                }
                
            # Verificar posiciones concurrentes (simulado para training)
            # En producción, esto se haría contra posiciones reales
            concurrent_positions = len(self.executor.positions)
            max_concurrent = self.config["risk"]["max_concurrent_positions"]
            
            if concurrent_positions >= max_concurrent:
                return {
                    "allowed": False, 
                    "reason": f"Max concurrent positions reached: {concurrent_positions}/{max_concurrent}"
                }
                
            # Calcular leverage dinámico usando balance_manager existente
            dynamic_leverage = calculate_dynamic_leverage(
                symbol=symbol,
                signal_strength=signal['strength'],
                current_balance=current_balance,
                recent_performance=1.0
            )
            
            return {
                "allowed": True,
                "params": {
                    "current_balance": current_balance,
                    "risk_per_trade": risk_per_trade,
                    "dynamic_leverage": dynamic_leverage,
                    "symbol_config": symbol_config
                },
                "reason": "Risk validation passed"
            }
            
        except Exception as e:
            self.logger.error(f"Error validating risk for {symbol}: {e}")
            return {"allowed": False, "reason": f"Risk validation error: {e}"}
            
    def _execute_decisions(self, decisions: List[TradingDecision]) -> int:
        """
        Ejecuta las decisiones usando el sistema existente
        
        Flow: decision → plan_and_store → executor.execute_pending_plans
        """
        executed_count = 0
        
        for decision in decisions:
            if not decision.should_execute:
                continue
                
            try:
                # 1. Generar plan usando planner.py existente  
                plan_id = plan_and_store(
                    symbol=decision.symbol,
                    tf=decision.timeframe,
                    ts=decision.timestamp,
                    side=decision.side,
                    strength=decision.strength,
                    direction_ver_id=1,  # TODO: obtener del signal.meta
                    engine=self.engine
                )
                
                if plan_id > 0:
                    decision.plan_params = {"plan_id": plan_id}
                    
                    # Log the plan creation
                    self.logger.info(
                        f"PLAN CREATED: {decision.symbol} {decision.action} | "
                        f"Plan ID: {plan_id} | Confidence: {decision.confidence:.2f}"
                    )
                    
                else:
                    self.logger.warning(f"Failed to create plan for {decision.symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error executing decision for {decision.symbol}: {e}")
                
        # 2. Ejecutar todos los planes pendientes usando order_executor
        try:
            execution_results = self.executor.execute_pending_plans(limit=20)
            executed_count = sum(1 for r in execution_results if r.success)
            
            if executed_count > 0:
                self.logger.info(f"Executed {executed_count} plans successfully")
                
        except Exception as e:
            self.logger.error(f"Error in batch execution: {e}")
            
        return executed_count
        
    def get_decision_stats(self) -> Dict:
        """Obtiene estadísticas de decisiones"""
        stats = self.decision_stats.copy()
        
        # Calcular ratios
        if stats["signals_processed"] > 0:
            stats["decision_rate"] = stats["decisions_made"] / stats["signals_processed"]
            stats["confirmation_rate"] = stats["confirmations_passed"] / stats["signals_processed"]
        else:
            stats["decision_rate"] = 0.0
            stats["confirmation_rate"] = 0.0
            
        if stats["decisions_made"] > 0:
            stats["execution_rate"] = stats["decisions_executed"] / stats["decisions_made"]
        else:
            stats["execution_rate"] = 0.0
            
        return stats
        
    def run_decision_loop(self, interval_seconds: int = 60):
        """
        Loop principal de toma de decisiones
        
        Args:
            interval_seconds: Intervalo entre ciclos de decisión
        """
        self.logger.info(f"Starting decision loop in {self.mode} mode (interval: {interval_seconds}s)")
        
        try:
            while True:
                start_time = datetime.now()
                
                # Procesar ciclo de decisiones
                decisions = self.process_signals_cycle(lookback_minutes=interval_seconds//60 + 5)
                
                # Log estadísticas cada 10 ciclos
                if sum(self.decision_stats.values()) % 10 == 0:
                    stats = self.get_decision_stats()
                    self.logger.info(f"Decision Stats: {stats}")
                    
                # Limpiar cache periódicamente
                if len(self._market_conditions_cache) > 100:
                    self._market_conditions_cache.clear()
                    
                # Esperar hasta el siguiente ciclo
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    import time
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("Decision loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Decision loop failed: {e}")
            raise


# Funciones de utilidad para integración

def create_training_decision_maker(symbols: List[str] = None) -> DecisionMaker:
    """Crea decision maker en modo training"""
    return DecisionMaker(mode="training", symbols=symbols)

def create_live_decision_maker(symbols: List[str] = None) -> DecisionMaker:
    """Crea decision maker en modo live"""
    return DecisionMaker(mode="live", symbols=symbols)

def process_signals_batch(mode: str = "training", symbols: List[str] = None, lookback_minutes: int = 30) -> Dict:
    """
    Procesa un lote de señales de forma síncrona
    
    Args:
        mode: "training" o "live" 
        symbols: Lista de símbolos o None para todos
        lookback_minutes: Minutos hacia atrás para buscar señales
        
    Returns:
        Dict con resultados del procesamiento
    """
    decision_maker = DecisionMaker(mode=mode, symbols=symbols)
    decisions = decision_maker.process_signals_cycle(lookback_minutes)
    
    return {
        "total_decisions": len(decisions),
        "decisions_to_execute": sum(1 for d in decisions if d.should_execute),
        "avg_confidence": np.mean([d.confidence for d in decisions]) if decisions else 0.0,
        "decisions": [
            {
                "symbol": d.symbol,
                "action": d.action,
                "confidence": d.confidence,
                "strength": d.strength,
                "reasons": d.reasons,
                "market_trending": d.market_condition.is_trending
            }
            for d in decisions
        ],
        "stats": decision_maker.get_decision_stats()
    }


# Configuración YAML de ejemplo para config/trading/decisions.yaml

EXAMPLE_CONFIG = """
# config/trading/decisions.yaml - Configuración del sistema de decisiones

version: "1.0.0"

confirmations:
  require_multi_tf: true
  min_timeframes_agree: 2  # Al menos 2 de 4 timeframes deben confirmar
  timeframes_weight:
    "15m": 0.2
    "1h": 0.3
    "4h": 0.3 
    "1d": 0.2

filters:
  min_signal_strength: 0.6    # Mínima fuerza de señal ML
  min_confidence: 0.7         # Mínima confianza final
  max_volatility: 0.15        # Máxima volatilidad permitida
  rsi_overbought: 75          # RSI máximo para longs
  rsi_oversold: 25            # RSI mínimo para shorts

risk:
  max_concurrent_positions: 3      # Máximo posiciones simultáneas
  max_risk_per_trade: 0.02        # 2% máximo riesgo por trade
  correlation_threshold: 0.7       # Umbral correlación entre símbolos

execution:
  batch_size: 10              # Máximo decisiones por ciclo
  cooldown_minutes: 5         # Minutos entre decisiones mismo símbolo
  require_plan_validation: true   # Validar planes antes de ejecutar

logging:
  log_decisions: true         # Log todas las decisiones
  log_market_analysis: false  # Log análisis de mercado (verbose)
  log_confirmations: true     # Log resultados de confirmación
"""

if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧠 Bot Trading v11 - Decision Maker")
    print("=" * 50)
    
    # Crear decision maker en modo training
    decision_maker = create_training_decision_maker()
    
    # Procesar un lote de señales
    print("Procesando señales...")
    decisions = decision_maker.process_signals_cycle(lookback_minutes=60)
    
    print(f"\n📊 Resultados:")
    print(f"Decisiones generadas: {len(decisions)}")
    print(f"Para ejecutar: {sum(1 for d in decisions if d.should_execute)}")
    
    if decisions:
        print(f"Confianza promedio: {np.mean([d.confidence for d in decisions]):.2f}")
        print(f"Fuerza promedio: {np.mean([d.strength for d in decisions]):.2f}")
        
        print(f"\n📋 Decisiones por símbolo:")
        for decision in decisions:
            print(f"  {decision.symbol}: {decision.action} (conf: {decision.confidence:.2f})")
            
    # Mostrar estadísticas
    stats = decision_maker.get_decision_stats()
    print(f"\n📈 Estadísticas:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
            
    print(f"\nEjemplo completado. Para uso continuo, ejecutar decision_maker.run_decision_loop()")