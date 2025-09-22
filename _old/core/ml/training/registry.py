# core/ml/training/registry.py
import os
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

def get_or_create_agent(name: str, kind: str) -> int:
    qsel = text("SELECT id FROM trading.Agents WHERE name=:n AND kind=:k")
    qins = text("INSERT INTO trading.Agents(name,kind) VALUES(:n,:k) RETURNING id")
    with ENGINE.begin() as c:
        row = c.execute(qsel, {"n": name, "k": kind}).fetchone()
        if row:
            return row[0]
        return c.execute(qins, {"n": name, "k": kind}).scalar()

def register_version(agent_id: int, version: str, params: Dict[str,Any], artifact_uri: str,
                     train_start=None, train_end=None, metrics: Optional[Dict[str,Any]]=None,
                     promoted: bool=False) -> int:
    # Definimos el SQL con parámetros normales (sin ::jsonb en el texto)
    q = text("""
        INSERT INTO trading.AgentVersions(
          agent_id, version, params, artifact_uri, train_start, train_end, metrics, promoted
        )
        VALUES (:aid, :ver, :par, :uri, :ts, :te, :met, :pro)
        RETURNING id
    """).bindparams(
        bindparam("par", type_=JSONB),   # 👈 fuerza JSONB
        bindparam("met", type_=JSONB)
    )
    with ENGINE.begin() as c:
        return c.execute(q, {
            "aid": agent_id,
            "ver": version,
            "par": params or {},        # 👈 pasa dicts directamente
            "uri": artifact_uri,
            "ts": train_start,
            "te": train_end,
            "met": metrics or {},
            "pro": promoted
        }).scalar()
