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
DB_URL = os.getenv('DB_URL', 'postgresql://postgres:160501@localhost:5432/trading_db')

# Configurar logging
logger = logging.getLogger(__name__)

# Engine y Session
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo HistoricalData
class HistoricalData(Base):
    __tablename__ = "HistoricalData"
    __table_args__ = (
        Index('idx_historical_symbol_timeframe', "symbol", "timeframe"),
        Index('idx_historical_timestamp', "timestamp"),
        Index('idx_historical_covering', "symbol", "timeframe", "timestamp", "close"),
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
    __tablename__ = "Trades"
    __table_args__ = (
        Index('idx_trades_symbol', "symbol"),
        Index('idx_trades_entry_timestamp', "entry_timestamp"),
        Index('idx_trades_covering', "symbol", "entry_timestamp", "side", "pnl"),
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
    __tablename__ = "MLStrategies"
    __table_args__ = (
        Index('idx_strategies_symbol', "symbol"),
        Index('idx_strategies_timestamp', "timestamp"),
        Index('idx_strategies_covering', "symbol", "timestamp", "action", "pnl"),
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
    __tablename__ = "AuditLog"
    __table_args__ = (
        Index('idx_auditlog_timestamp', "timestamp"),
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
    """Crea índices idempotentes (IF NOT EXISTS) para acelerar lecturas comunes."""
    # HistoricalData
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_historical_timestamp
        ON trading."HistoricalData" USING btree (timestamp);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_hist_symbol_timeframe
        ON trading."HistoricalData" USING btree (symbol, timeframe);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_hist_symbol_tf_ts
        ON trading."HistoricalData" USING btree (symbol, timeframe, timestamp DESC);
    """))
    # covering (si tu Postgres >= 11)
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_historical_covering
        ON trading."HistoricalData" USING btree (symbol, timeframe, timestamp)
        INCLUDE (open, high, low, volume, close);
    """))

    # Features
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_features_sym_tf_ts
        ON trading."Features" USING btree (symbol, timeframe, timestamp DESC);
    """))

    # AgentPreds / AgentSignals
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_pred_ver_ts
        ON trading."AgentPreds" (agent_version_id, timestamp DESC);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_signals_sym_tf_ts
        ON trading."AgentSignals" (symbol, timeframe, timestamp DESC);
    """))
    # índice funcional sobre JSONB
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_sig_meta_ts
        ON trading."AgentSignals" ((meta->>'direction_ver_id'), timestamp DESC);
    """))

    # TradePlans: intentar con bar_ts; si falla, fallback a created_at
    try:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_bar
            ON trading."TradePlans" (symbol, timeframe, bar_ts DESC);
        """))
    except Exception:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_created
            ON trading."TradePlans" (symbol, timeframe, created_at DESC);
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
        FROM trading.HistoricalData
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
        SELECT * FROM trading.Trades
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

def insert_trade(trade_data: Dict) -> bool:
    """Inserta un trade en la base de datos."""
    insert_query = text("""
        INSERT INTO trading.Trades (symbol, side, quantity, price, pnl, entry_timestamp, exit_timestamp, duration, leverage)
        VALUES (:symbol, :side, :quantity, :price, :pnl, :entry_timestamp, :exit_timestamp, :duration, :leverage)
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
        INSERT INTO trading.MLStrategies (symbol, timestamp, action, timeframes, indicators, tools, leverage, pnl, performance, confidence_score, feature_importance, outcome)
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
        INSERT INTO trading.AuditLog (table_name, action, record_id, changed_by, details)
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