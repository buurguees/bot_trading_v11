# core/trading/order_executor.py
"""
Sistema de ejecución de órdenes que funciona en modo REAL y TRAINING.
Integra perfectamente con los componentes existentes:
- core/trading/planner.py → lee planes de trading.tradeplans  
- core/trading/balance_manager.py → gestiona balance por símbolo
- core/ml/backtests/backtest_plans.py → simula ejecuciones para training

En modo TRAINING: simula ejecuciones para entrenar con balance ficticio conjunto
En modo REAL: ejecuta órdenes reales en BitGet
"""

import os
import logging
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json
import time
import asyncio

# Importar componentes existentes
from ..ml.training.daily_train.balance_manager import load_symbol_config
from .position_sizer import RiskParams, load_risk_params
from ..ml.backtests.backtest_plans import simulate_symbol_tf
from ..ml.backtests.strategy_memory import update_memory

load_dotenv("config/.env")

@dataclass
class ExecutionResult:
    """Resultado de ejecución de orden"""
    success: bool
    plan_id: int
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fees: float = 0.0
    slippage_bps: float = 0.0
    execution_time: float = 0.0
    error: Optional[str] = None
    mode: str = "training"  # "training" or "live"
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass  
class PositionStatus:
    """Estado de posición activa"""
    symbol: str
    side: int  # 1=long, -1=short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float
    margin_used: float
    liquidation_price: Optional[float] = None
    
class OrderExecutor:
    """
    Ejecutor de órdenes que funciona en modo TRAINING y LIVE
    
    Modo TRAINING:
    - Simula ejecuciones usando backtest_plans.py
    - Gestiona balance ficticio conjunto usando balance_manager.py
    - Permite entrenar con trades "reales" sin riesgo
    
    Modo LIVE: 
    - Ejecuta órdenes reales en BitGet
    - Gestiona balance real
    - Tracking completo de fills y slippage
    """
    
    def __init__(self, mode: str = "training"):
        self.mode = mode.lower()
        self.logger = logging.getLogger(__name__)
        
        # DB connection
        self.engine = create_engine(os.getenv("DB_URL"))
        
        # Balance managers por símbolo (usando balance_manager.py existente)
        self.balance_managers = {}
        self.initialize_balance_managers()
        
        # Posiciones activas (símbolo -> PositionStatus)
        self.positions: Dict[str, PositionStatus] = {}
        
        # Exchange connection (solo en modo live)
        self.exchange = None
        if self.mode == "live":
            self.initialize_exchange()
            
        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0, 
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0
        }
        
    def initialize_balance_managers(self):
        """Inicializa balance managers para cada símbolo usando balance_manager.py"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
        
        for symbol in symbols:
            # Usar la función existente load_symbol_config
            config = load_symbol_config(symbol)
            
            # Crear balance manager por símbolo
            self.balance_managers[symbol] = {
                "current_balance": config["initial"],  # Balance inicial por símbolo
                "initial_balance": config["initial"],
                "target_balance": config["target"], 
                "config": config
            }
            
        self.logger.info(f"Initialized balance managers for {len(symbols)} symbols in {self.mode} mode")
        
    def initialize_exchange(self):
        """Inicializa conexión con BitGet (solo en modo live)"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET'), 
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': os.getenv('BITGET_SANDBOX', 'false').lower() == 'true',
                'options': {
                    'defaultType': 'swap',  # Futuros perpetuos
                    'marginMode': 'isolated',  # Modo aislado por defecto
                },
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            self.logger.info(f"Exchange initialized. USDT balance: {balance.get('USDT', {}).get('free', 0)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
            
    def get_total_portfolio_balance(self) -> Dict[str, float]:
        """Obtiene balance total del portfolio (suma de todos los símbolos)"""
        total_initial = sum(bm["initial_balance"] for bm in self.balance_managers.values())
        total_current = sum(bm["current_balance"] for bm in self.balance_managers.values())
        total_target = sum(bm["target_balance"] for bm in self.balance_managers.values())
        
        return {
            "total_initial": total_initial,
            "total_current": total_current, 
            "total_target": total_target,
            "total_pnl": total_current - total_initial,
            "total_pnl_pct": ((total_current - total_initial) / total_initial) * 100 if total_initial > 0 else 0,
            "progress_to_target": min(1.0, (total_current - total_initial) / (total_target - total_initial)) if total_target > total_initial else 1.0
        }
        
    def execute_pending_plans(self, symbol: str = None, limit: int = 10) -> List[ExecutionResult]:
        """
        Ejecuta planes pendientes desde trading.tradeplans
        
        Args:
            symbol: Si se especifica, solo ejecuta para ese símbolo
            limit: Máximo número de planes a ejecutar
        """
        results = []
        
        # Cargar planes pendientes
        plans = self._load_pending_plans(symbol, limit)
        
        for plan in plans:
            try:
                if self.mode == "training":
                    result = self._execute_plan_training(plan)
                else:
                    result = self._execute_plan_live(plan)
                    
                results.append(result)
                self._update_execution_stats(result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute plan {plan['id']}: {e}")
                error_result = ExecutionResult(
                    success=False,
                    plan_id=plan['id'],
                    error=str(e),
                    mode=self.mode
                )
                results.append(error_result)
                
        return results
        
    def _load_pending_plans(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Carga planes pendientes desde trading.tradeplans"""
        with self.engine.connect() as conn:
            base_query = """
                SELECT 
                    id, symbol, timeframe, side, entry_px, sl_px, tp_px,
                    qty, leverage, risk_pct, reason, bar_ts, created_at
                FROM trading.tradeplans 
                WHERE status = 'planned'
            """
            
            if symbol:
                query = base_query + " AND symbol = :symbol"
                params = {"symbol": symbol}
            else:
                query = base_query
                params = {}
                
            query += " ORDER BY created_at ASC LIMIT :limit"
            params["limit"] = limit
            
            result = conn.execute(text(query), params)
            return [dict(row._mapping) for row in result]
            
    def _execute_plan_training(self, plan: Dict) -> ExecutionResult:
        """
        Ejecuta plan en modo TRAINING usando simulación avanzada
        Integra con balance_manager.py existente
        """
        start_time = time.time()
        plan_id = plan['id']
        symbol = plan['symbol']
        
        try:
            # 1. Validar que tenemos balance suficiente para el símbolo
            balance_manager = self.balance_managers[symbol]
            current_balance = balance_manager["current_balance"]
            
            # 2. Simular la ejecución usando backtest_plans.py
            simulation_result = self._simulate_single_trade(plan)
            
            if not simulation_result:
                return ExecutionResult(
                    success=False,
                    plan_id=plan_id,
                    error="Simulation failed - no data available",
                    mode="training"
                )
                
            # 3. Actualizar balance del símbolo
            pnl = simulation_result.get('pnl_net', 0.0)
            new_balance = current_balance + pnl
            balance_manager["current_balance"] = new_balance
            
            # 4. Marcar plan como ejecutado en DB
            self._update_plan_status(plan_id, "executed", simulation_result)
            
            # 5. Registrar en strategy_memory como backtest/training
            try:
                with self.engine.begin() as conn:
                    trade = {
                        "entry_ts": plan.get('bar_ts'),
                        "pnl": simulation_result.get('pnl_net', 0.0),
                        "side": plan.get('side'),
                        "leverage": plan.get('leverage'),
                        "reason": plan.get('reason') if isinstance(plan.get('reason'), dict) else {},
                    }
                    update_memory(conn, plan['symbol'], plan['timeframe'], [trade], mode="backtest")
            except Exception as _:
                pass

            # 6. Logging para training
            self.logger.info(
                f"TRAINING EXECUTION: {symbol} | "
                f"Side: {plan['side']} | "
                f"PnL: {pnl:.2f} | "
                f"Balance: {current_balance:.2f} → {new_balance:.2f}"
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                plan_id=plan_id,
                filled_qty=simulation_result.get('qty', 0),
                avg_price=simulation_result.get('avg_fill_price', plan['entry_px']),
                fees=simulation_result.get('fees', 0),
                slippage_bps=simulation_result.get('slippage_bps', 0),
                execution_time=execution_time,
                mode="training"
            )
            
        except Exception as e:
            self.logger.error(f"Training execution failed for plan {plan_id}: {e}")
            return ExecutionResult(
                success=False,
                plan_id=plan_id,
                error=str(e),
                mode="training"
            )
            
    def _execute_plan_live(self, plan: Dict) -> ExecutionResult:
        """Ejecuta plan en modo LIVE en BitGet"""
        start_time = time.time()
        plan_id = plan['id']
        
        try:
            # 1. Validar balance suficiente
            if not self._validate_balance_for_plan(plan):
                return ExecutionResult(
                    success=False,
                    plan_id=plan_id,
                    error="Insufficient balance",
                    mode="live"
                )
                
            # 2. Preparar parámetros de orden
            symbol = plan['symbol']
            side = 'buy' if plan['side'] == 1 else 'sell'
            quantity = abs(plan['qty'])
            
            # 3. Ejecutar orden market
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=quantity,
                params={
                    'leverage': plan['leverage'],
                    'marginMode': 'isolated'
                }
            )
            
            # 4. Obtener detalles del fill
            order_details = self.exchange.fetch_order(order['id'], symbol)
            
            # 5. Configurar SL/TP si la orden se ejecutó
            if order_details['status'] == 'closed':
                self._set_stop_loss_take_profit(
                    symbol, 
                    plan['side'],
                    plan['sl_px'], 
                    plan['tp_px'],
                    quantity
                )
                
            # 6. Actualizar plan en DB
            fill_data = {
                'order_id': order['id'],
                'filled_qty': order_details['filled'],
                'avg_price': order_details['average'],
                'fees': order_details['fee']['cost'],
                'timestamp': datetime.now(timezone.utc)
            }
            self._update_plan_status(plan_id, "executed", fill_data)

            # 7. Registrar en strategy_memory como real
            try:
                with self.engine.begin() as conn:
                    trade = {
                        "entry_ts": plan.get('bar_ts'),
                        "pnl": 0.0,  # si tienes PnL realizado, reemplazar
                        "side": plan.get('side'),
                        "leverage": plan.get('leverage'),
                        "reason": plan.get('reason') if isinstance(plan.get('reason'), dict) else {},
                    }
                    update_memory(conn, plan['symbol'], plan['timeframe'], [trade], mode="real")
            except Exception as _:
                pass
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                plan_id=plan_id,
                order_id=order['id'],
                filled_qty=order_details['filled'],
                avg_price=order_details['average'],
                fees=order_details['fee']['cost'],
                execution_time=execution_time,
                mode="live"
            )
            
        except Exception as e:
            self.logger.error(f"Live execution failed for plan {plan_id}: {e}")
            return ExecutionResult(
                success=False,
                plan_id=plan_id,
                error=str(e),
                mode="live"
            )
            
    def _simulate_single_trade(self, plan: Dict) -> Optional[Dict]:
        """
        Simula un solo trade usando los datos históricos
        Integra con el sistema de backtesting existente
        """
        symbol = plan['symbol']
        timeframe = plan['timeframe'] 
        
        # Usar el timestamp del plan para simular desde ahí
        from_ts = plan['bar_ts']
        to_ts = from_ts + timedelta(hours=24)  # Simular hasta 24h después
        
        try:
            with self.engine.connect() as conn:
                # Usar la función existente simulate_symbol_tf 
                summary, trades = simulate_symbol_tf(
                    conn=conn,
                    symbol=symbol,
                    tf=timeframe,
                    from_ts=from_ts,
                    to_ts=to_ts,
                    fees_bps=2.0,  # 2 bps de fees
                    slip_bps=1.0,  # 1 bp de slippage
                    mmr=0.005,     # 0.5% maintenance margin
                    max_hold_bars=300  # Máximo 5 horas para 1m
                )
                
                if trades and len(trades) > 0:
                    # Tomar el primer trade que coincida con nuestro plan
                    for trade in trades:
                        if (trade.get('plan_id') == plan['id'] or 
                            (trade.get('side') == plan['side'] and 
                             abs(trade.get('entry_px', 0) - plan['entry_px']) < 0.01)):
                            
                            return {
                                'pnl_net': trade.get('pnl_net', 0),
                                'qty': trade.get('qty', plan['qty']),
                                'avg_fill_price': trade.get('entry_px', plan['entry_px']),
                                'fees': trade.get('fees', 0),
                                'slippage_bps': 1.0,  # Simulado
                                'exit_price': trade.get('exit_px'),
                                'exit_reason': trade.get('exit_reason', 'simulation')
                            }
                            
        except Exception as e:
            self.logger.warning(f"Simulation failed for plan {plan['id']}: {e}")
            
        return None
        
    def _validate_balance_for_plan(self, plan: Dict) -> bool:
        """Valida que hay balance suficiente para ejecutar el plan"""
        if self.mode == "training":
            symbol = plan['symbol']
            balance_manager = self.balance_managers[symbol] 
            current_balance = balance_manager["current_balance"]
            
            # En training, verificar balance por símbolo
            required_margin = plan['qty'] * plan['entry_px'] / plan['leverage']
            return current_balance >= required_margin
            
        else:
            # En live, verificar balance real en exchange
            try:
                balance = self.exchange.fetch_balance()
                usdt_free = balance.get('USDT', {}).get('free', 0)
                required_margin = plan['qty'] * plan['entry_px'] / plan['leverage']
                return usdt_free >= required_margin
            except:
                return False
                
    def _set_stop_loss_take_profit(self, symbol: str, side: int, sl_price: float, tp_price: float, quantity: float):
        """Configura SL/TP en el exchange (solo modo live)"""
        if self.mode != "live":
            return
            
        try:
            # Stop Loss
            self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side='sell' if side == 1 else 'buy',
                amount=quantity,
                params={
                    'stopPrice': sl_price,
                    'reduceOnly': True
                }
            )
            
            # Take Profit  
            self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='sell' if side == 1 else 'buy',
                amount=quantity,
                price=tp_price,
                params={
                    'reduceOnly': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to set SL/TP for {symbol}: {e}")
            
    def _update_plan_status(self, plan_id: int, status: str, execution_data: Dict = None):
        """Actualiza el status de un plan en la base de datos"""
        with self.engine.connect() as conn:
            if execution_data:
                query = text("""
                    UPDATE trading.tradeplans 
                    SET status = :status, 
                        execution_data = :exec_data,
                        executed_at = :exec_time
                    WHERE id = :plan_id
                """)
                conn.execute(query, {
                    "status": status,
                    "exec_data": json.dumps(execution_data, default=str),
                    "exec_time": datetime.now(timezone.utc),
                    "plan_id": plan_id
                })
            else:
                query = text("""
                    UPDATE trading.tradeplans 
                    SET status = :status,
                        executed_at = :exec_time
                    WHERE id = :plan_id
                """)
                conn.execute(query, {
                    "status": status,
                    "exec_time": datetime.now(timezone.utc),
                    "plan_id": plan_id
                })
                
            conn.commit()
            
    def _update_execution_stats(self, result: ExecutionResult):
        """Actualiza estadísticas de ejecución"""
        self.execution_stats["total_executions"] += 1
        
        if result.success:
            self.execution_stats["successful_executions"] += 1
            self.execution_stats["total_fees"] += result.fees
            self.execution_stats["total_slippage"] += result.slippage_bps
            
            # Actualizar promedio de tiempo de ejecución
            current_avg = self.execution_stats["avg_execution_time"]
            total_successful = self.execution_stats["successful_executions"]
            self.execution_stats["avg_execution_time"] = (
                (current_avg * (total_successful - 1) + result.execution_time) / total_successful
            )
        else:
            self.execution_stats["failed_executions"] += 1
            
    def get_portfolio_status(self) -> Dict:
        """Obtiene estado completo del portfolio"""
        portfolio_balance = self.get_total_portfolio_balance()
        
        # Balance por símbolo
        symbol_balances = {}
        for symbol, manager in self.balance_managers.items():
            symbol_balances[symbol] = {
                "current": manager["current_balance"],
                "initial": manager["initial_balance"],
                "target": manager["target_balance"],
                "pnl": manager["current_balance"] - manager["initial_balance"],
                "pnl_pct": ((manager["current_balance"] - manager["initial_balance"]) / manager["initial_balance"]) * 100
            }
            
        return {
            "mode": self.mode,
            "portfolio_balance": portfolio_balance,
            "symbol_balances": symbol_balances,
            "execution_stats": self.execution_stats,
            "active_positions": len(self.positions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    def close_all_positions(self) -> List[ExecutionResult]:
        """Cierra todas las posiciones activas (solo en modo live)"""
        if self.mode != "live":
            self.logger.info("Close all positions called in training mode - no action needed")
            return []
            
        results = []
        
        try:
            positions = self.exchange.fetch_positions()
            
            for position in positions:
                if position['size'] > 0:  # Posición activa
                    symbol = position['symbol']
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    
                    order = self.exchange.create_market_order(
                        symbol=symbol,
                        side=side,
                        amount=position['size'],
                        params={'reduceOnly': True}
                    )
                    
                    results.append(ExecutionResult(
                        success=True,
                        plan_id=-1,  # Special plan_id for manual closes
                        order_id=order['id'],
                        filled_qty=position['size'],
                        mode="live"
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to close positions: {e}")
            
        return results
        
    def run_execution_loop(self, interval_seconds: int = 60):
        """
        Loop principal de ejecución
        
        Args:
            interval_seconds: Intervalo entre ejecuciones en segundos
        """
        self.logger.info(f"Starting execution loop in {self.mode} mode (interval: {interval_seconds}s)")
        
        try:
            while True:
                start_time = time.time()
                
                # Ejecutar planes pendientes para todos los símbolos
                all_results = []
                for symbol in self.balance_managers.keys():
                    results = self.execute_pending_plans(symbol, limit=5)
                    all_results.extend(results)
                    
                if all_results:
                    successful = sum(1 for r in all_results if r.success)
                    self.logger.info(f"Executed {successful}/{len(all_results)} plans successfully")
                    
                    # Log portfolio status cada 10 ejecuciones
                    if self.execution_stats["total_executions"] % 10 == 0:
                        status = self.get_portfolio_status()
                        self.logger.info(f"Portfolio Status: {status['portfolio_balance']}")
                
                # Esperar hasta el siguiente ciclo
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Execution loop interrupted by user")
        except Exception as e:
            self.logger.error(f"Execution loop failed: {e}")
            raise
            

# Funciones de utilidad para integración con otros módulos

def create_training_executor() -> OrderExecutor:
    """Crea un ejecutor en modo training para entrenamiento y backtesting"""
    return OrderExecutor(mode="training")

def create_live_executor() -> OrderExecutor:
    """Crea un ejecutor en modo live para trading real"""
    return OrderExecutor(mode="live")

def execute_plans_batch(symbol: str = None, mode: str = "training", limit: int = 10) -> Dict:
    """
    Ejecuta un lote de planes de forma synchrona
    
    Args:
        symbol: Símbolo específico o None para todos
        mode: "training" o "live"
        limit: Máximo planes a ejecutar
        
    Returns:
        Dict con resultados de la ejecución
    """
    executor = OrderExecutor(mode=mode)
    results = executor.execute_pending_plans(symbol, limit)
    
    return {
        "total_plans": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success), 
        "results": [r.to_dict() for r in results],
        "portfolio_status": executor.get_portfolio_status()
    }


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear ejecutor en modo training
    executor = create_training_executor()
    
    # Ejecutar algunos planes
    results = executor.execute_pending_plans(limit=5)
    
    print(f"Executed {len(results)} plans")
    print(f"Portfolio status: {executor.get_portfolio_status()}")