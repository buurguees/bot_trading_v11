import os
import logging
from sqlalchemy import create_engine, Column, BigInteger, String, DateTime, Numeric, JSON, CheckConstraint, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:160501@192.168.10.109:5432/trading_db')

# Configurar logging
logger = logging.getLogger(__name__)

# Engine y Session
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    isolation_level="READ COMMITTED",
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo HistoricalData
class HistoricalData(Base):
    __tablename__ = "historicaldata"
    __table_args__ = (
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    timeframe = Column(String(5), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(15, 8), nullable=False)
    high = Column(Numeric(15, 8), nullable=False)
    low = Column(Numeric(15, 8), nullable=False)
    close = Column(Numeric(15, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ += (
        CheckConstraint("timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d')", name="chk_timeframe"),
        {"schema": "trading"}
    )

# Modelo Trades
class Trades(Base):
    __tablename__ = "trades"
    __table_args__ = (
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)
    quantity = Column(Numeric(15, 8), nullable=False)
    price = Column(Numeric(15, 8), nullable=False)
    pnl = Column(Numeric(15, 8), nullable=False)
    entry_timestamp = Column(DateTime(timezone=True), nullable=False)
    exit_timestamp = Column(DateTime(timezone=True))
    duration = Column(String(20))  # e.g., '1 day 02:30:00'
    leverage = Column(Numeric(5, 2), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ += (
        CheckConstraint("side IN ('long', 'short')", name="chk_side"),
        CheckConstraint("leverage >= 1 AND leverage <= 125", name="chk_leverage"),
        {"schema": "trading"}
    )

# Modelo MLStrategies
class MLStrategies(Base):
    __tablename__ = "mlstrategies"
    __table_args__ = (
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    action = Column(String(10), nullable=False)
    timeframes = Column(JSON, nullable=False)
    indicators = Column(JSON, nullable=False)
    tools = Column(JSON, nullable=False)
    leverage = Column(Numeric(5, 2), nullable=False)
    pnl = Column(Numeric(15, 8), nullable=False)
    performance = Column(Numeric(5, 2), nullable=False)
    confidence_score = Column(Numeric(5, 2), nullable=False)
    feature_importance = Column(JSON, nullable=False)
    outcome = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ += (
        CheckConstraint("action IN ('long', 'short', 'hold')", name="chk_action"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="chk_confidence"),
        {"schema": "trading"}
    )

# Modelo AuditLog (opcional para auditoría)
class AuditLog(Base):
    __tablename__ = "auditlog"
    __table_args__ = (
        {"schema": "trading"}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    table_name = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)
    record_id = Column(BigInteger, nullable=False)
    changed_by = Column(String(100))
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    details = Column(JSON)

# Crear tablas si no existen
Base.metadata.create_all(bind=engine)

def ensure_indexes(conn):
    """Crea índices idempotentes (IF NOT EXISTS) y sólo si existen las tablas."""

    def _table_exists(schema: str, table: str) -> bool:
        return conn.execute(text(
            """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :s AND table_name = :t
            )
            """
        ), {"s": schema, "t": table}).scalar()

    # HistoricalData (estandarizado en minúsculas)
    if _table_exists("trading", "historicaldata"):
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_historical_timestamp
            ON trading.historicaldata (timestamp)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_hist_symbol_timeframe
            ON trading.historicaldata (symbol, timeframe)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_hist_symbol_tf_ts
            ON trading.historicaldata (symbol, timeframe, timestamp DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_historical_covering
            ON trading.historicaldata (symbol, timeframe, timestamp)
            INCLUDE (open, high, low, volume, close)
        """))

    # Features (si existe)
    if _table_exists("trading", "features"):
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_features_sym_tf_ts
            ON trading.features (symbol, timeframe, timestamp DESC)
        """))

    # AgentPreds / AgentSignals (si existen)
    if _table_exists("trading", "agentpreds"):
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agentpreds_ver_sym_tf_ts
            ON trading.agentpreds (agent_version_id, symbol, timeframe, timestamp DESC)
        """))
    if _table_exists("trading", "agentsignals"):
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_signals_sym_tf_ts
            ON trading.agentsignals (symbol, timeframe, timestamp DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_sig_meta_ts
            ON trading.agentsignals ((meta->>'direction_ver_id'), timestamp DESC)
        """))

    # TradePlans (si existe)
    if _table_exists("trading", "tradeplans"):
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_bar
            ON trading.tradeplans (symbol, timeframe, bar_ts DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_created
            ON trading.tradeplans (symbol, timeframe, created_at DESC)
        """))

def get_engine():
    return engine

# Asegurar índices al inicio (idempotente)
try:
    with engine.begin() as _c:
        ensure_indexes(_c)
except Exception as _e:
    logger.warning(f"No se pudieron asegurar índices al inicio: {_e}")

def get_db() -> Session:
    """Obtiene una sesión de DB para uso en dependencias."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_query(query: str, params: Dict = None) -> Optional[List[Dict]]:
    """Ejecuta una query personalizada y retorna resultados como diccionario."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]
    except SQLAlchemyError as e:
        logger.error(f"Error executing query: {e}")
        return None

def insert_historical_data(symbol: str, timeframe: str, ohlcv: List) -> bool:
    """Inserta datos OHLCV específicos (wrapper para save_to_db)."""
    return save_to_db(ohlcv, symbol, timeframe)

def get_historical_data(symbol: str, timeframe: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
    """Obtiene datos históricos para un símbolo y timeframe."""
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM trading.historicaldata
        WHERE symbol = :symbol AND timeframe = :timeframe
    """
    params = {"symbol": symbol, "timeframe": timeframe}
    if start_date:
        query += " AND timestamp >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND timestamp <= :end_date"
        params["end_date"] = end_date
    query += " ORDER BY timestamp ASC"

    return execute_query(query, params)

def get_trades(symbol: str = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
    """Obtiene trades para un símbolo o período."""
    query = """
        SELECT * FROM trading.trades
    """
    params = {}
    if symbol:
        query += " WHERE symbol = :symbol"
        params["symbol"] = symbol
    if start_date:
        query += " AND entry_timestamp >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND entry_timestamp <= :end_date"
        params["end_date"] = end_date
    query += " ORDER BY entry_timestamp DESC"

    return execute_query(query, params)

def get_symbol_id(symbol: str) -> Optional[int]:
    """Obtiene el ID del símbolo desde la tabla symbols."""
    query = "SELECT id FROM trading.symbols WHERE symbol = :symbol"
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), {"symbol": symbol})
            row = result.fetchone()
            return row[0] if row else None
    except SQLAlchemyError as e:
        logger.error(f"Error getting symbol_id for {symbol}: {e}")
        return None

def insert_trade(trade_data: Dict) -> bool:
    """Inserta un trade en la base de datos."""
    
    # Obtener symbol_id si no está proporcionado
    if 'symbol_id' not in trade_data and 'symbol' in trade_data:
        trade_data['symbol_id'] = get_symbol_id(trade_data['symbol'])
    
    # Valores por defecto para campos opcionales
    defaults = {
        'plan_id': None,
        'order_ids': '{}',
        'fees_paid': 0.0,
        'slip_bps': 0.0,
        'entry_balance': None,
        'exit_balance': None,
        'symbol_id': None
    }
    
    for key, default_value in defaults.items():
        if key not in trade_data:
            trade_data[key] = default_value
    
    # Convertir order_ids a JSON string si es un dict
    if isinstance(trade_data.get('order_ids'), dict):
        import json
        trade_data['order_ids'] = json.dumps(trade_data['order_ids'])
    
    insert_query = text("""
        INSERT INTO trading.trades (
            symbol, side, quantity, price, pnl, entry_timestamp, exit_timestamp, 
            duration, leverage, plan_id, order_ids, fees_paid, slip_bps, 
            entry_balance, exit_balance, symbol_id
        )
        VALUES (
            :symbol, :side, :quantity, :price, :pnl, :entry_timestamp, :exit_timestamp, 
            :duration, :leverage, :plan_id, :order_ids, :fees_paid, :slip_bps, 
            :entry_balance, :exit_balance, :symbol_id
        )
    """)
    try:
        with engine.begin() as connection:
            connection.execute(insert_query, trade_data)
        logger.info(f"Trade inserted for {trade_data.get('symbol')}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Error inserting trade: {e}")
        return False

def insert_strategy(strategy_data: Dict) -> bool:
    """Inserta una estrategia ML en la base de datos."""
    insert_query = text("""
        INSERT INTO trading.mlstrategies (symbol, timestamp, action, timeframes, indicators, tools, leverage, pnl, performance, confidence_score, feature_importance, outcome)
        VALUES (:symbol, :timestamp, :action, :timeframes, :indicators, :tools, :leverage, :pnl, :performance, :confidence_score, :feature_importance, :outcome)
    """)
    try:
        with engine.begin() as connection:
            connection.execute(insert_query, strategy_data)
        logger.info(f"Strategy inserted for {strategy_data.get('symbol')}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Error inserting strategy: {e}")
        return False

def log_audit(table_name: str, action: str, record_id: int, changed_by: str = None, details: Dict = None):
    """Registra una acción de auditoría."""
    insert_query = text("""
        INSERT INTO trading.auditlog (table_name, action, record_id, changed_by, details)
        VALUES (:table_name, :action, :record_id, :changed_by, :details)
    """)
    try:
        with engine.begin() as connection:
            connection.execute(insert_query, {
                "table_name": table_name,
                "action": action,
                "record_id": record_id,
                "changed_by": changed_by,
                "details": details
            })
        logger.debug(f"Audit log for {table_name}: {action} on ID {record_id}")
    except SQLAlchemyError as e:
        logger.error(f"Error logging audit: {e}")