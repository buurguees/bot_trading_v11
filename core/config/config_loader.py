"""
core/config/config_loader.py
----------------------------
Carga y valida YAMLs de entrenamiento y símbolos.
Construye objetos de configuración para el encoder y parámetros de batching.

Requisitos:
- pip install pyyaml
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml

from core.ml.encoders.multitf_encoder import MultiTFEncoderConfig

TRAIN_DEFAULT_PATH = os.path.join("config", "train", "training.yaml")
SYMBOLS_DEFAULT_PATH = os.path.join("config", "market", "symbols.yaml")
RISK_DEFAULT_PATH = os.path.join("config", "trading", "risk.yaml")


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_training_config(path: str = TRAIN_DEFAULT_PATH) -> Dict:
    return load_yaml(path)


def load_symbols_config(path: str = SYMBOLS_DEFAULT_PATH) -> Dict:
    return load_yaml(path)


def load_risk_config(path: str = RISK_DEFAULT_PATH) -> Dict:
    """
    Carga configuración de riesgo desde risk.yaml.
    Devuelve configuración con valores por defecto si no están definidos.
    """
    try:
        cfg = load_yaml(path)
        limits = cfg.get("limits", {})
    except FileNotFoundError:
        # Si no existe el archivo, usar valores por defecto
        limits = {}
    
    return {
        "max_leverage": limits.get("max_leverage", 3.0),
        "max_open_positions": limits.get("max_open_positions", 6),
        "max_exposure_per_symbol_usdt": limits.get("max_exposure_per_symbol_usdt", 50000),
        "min_qty": limits.get("min_qty", 0.0001),
        "max_qty": limits.get("max_qty", 1000000),
        "cooldown_minutes": limits.get("cooldown_minutes", 0),
        "dd_day_pct": limits.get("dd_day_pct", 5.0),
    }


def build_encoder_config_from_training(cfg_train: Dict) -> MultiTFEncoderConfig:
    """
    Extrae del training.yaml los campos que corresponden al encoder y a la lista de features.
    """
    enc = cfg_train.get("encoder", {})
    feats = cfg_train.get("features", {})
    numerical = feats.get("numerical", [])
    discrete = feats.get("discrete", [])
    flags = feats.get("smc_flags", [])

    in_features = list(numerical) + list(discrete) + list(flags)

    return MultiTFEncoderConfig(
        encoder_type=enc.get("type", "tcn"),
        in_features=in_features,
        d_model=enc.get("d_model", 128),
        hidden=enc.get("hidden", 128),
        layers=enc.get("layers", 3),
        kernel_size=enc.get("kernel_size", 3),
        base_dilation=enc.get("base_dilation", 1),
        dropout=enc.get("dropout", 0.1),
        n_heads=enc.get("n_heads", 4),
        query_tf_default=enc.get("query_tf_default", "1m"),
        device=cfg_train.get("system", {}).get("device", "cpu"),
    )


def extract_symbols_and_tfs(cfg_symbols: Dict, cfg_train: Dict) -> Tuple[List[str], List[str]]:
    """
    Devuelve (symbols, tfs) consolidados.
    Prioridad:
      - training.symbols_override si no está vacío
      - si no, símbolos de symbols.yaml
    Los timeframes salen de training.timeframes; si no, de symbols.yaml por símbolo.
    """
    # Símbolos
    override = cfg_train.get("data", {}).get("symbols_override", []) or []
    if override:
        symbols = override
    else:
        symbols = [s["id"] for s in cfg_symbols.get("symbols", [])]

    # Timeframes
    tfs_train = cfg_train.get("data", {}).get("timeframes", []) or []
    if tfs_train:
        tfs = tfs_train
    else:
        # unión de los tfs definidos por símbolo
        seen = set()
        tfs = []
        for s in cfg_symbols.get("symbols", []):
            for tf in s.get("timeframes", []):
                if tf not in seen:
                    seen.add(tf)
                    tfs.append(tf)

    return symbols, tfs


def get_window_from_training(cfg_train: Dict) -> int:
    return int(cfg_train.get("data", {}).get("window", 256))


def get_planner_config(cfg_train: Dict) -> Dict:
    """
    Extrae configuración del planner desde training.yaml.
    Devuelve configuración con valores por defecto si no están definidos.
    """
    planner = cfg_train.get("planner", {})
    execution = planner.get("execution", {})
    
    return {
        "ttl_minutes": execution.get("ttl_minutes", 60),
        "risk_pct": execution.get("risk_pct", 0.5),
        "leverage": execution.get("leverage", 2.0),
        "account_balance_usdt": execution.get("account_balance_usdt", 1000.0),
        "atr_sl_mult": execution.get("atr_sl_mult", 1.5),
        "atr_tp_mult_1": execution.get("atr_tp_mult_1", 1.0),
        "atr_tp_mult_2": execution.get("atr_tp_mult_2", 2.0),
    }


def get_execution_config(cfg_train: Dict) -> Dict:
    """
    Extrae configuración específica para execution desde training.yaml.
    Incluye query_tf para execution y parámetros de PPO.
    """
    heads = cfg_train.get("heads", {})
    execution_head = heads.get("execution", {})
    
    # Query TF para execution (por defecto usa encoder.query_tf_default)
    encoder = cfg_train.get("encoder", {})
    query_tf = execution_head.get("query_tf", encoder.get("query_tf_default", "1m"))
    
    # Configuración PPO
    ppo = cfg_train.get("ppo_execution", {})
    env = ppo.get("env", {})
    action_space = env.get("action_space", {})
    
    return {
        "query_tf": query_tf,
        "max_offset_bp": action_space.get("max_offset_bp", 50),
        "default_leverage": action_space.get("default_leverage", 2.0),
        "account_balance_usdt": env.get("account_balance_usdt", 1000.0),
    }


# ============================================================================
# HELPERS ADICIONALES PARA CONFIGURACIONES ESPECÍFICAS
# ============================================================================

_CFG_DIR = os.path.join("config")


def _load_yaml(path: str, default: dict | None = None) -> dict:
    """
    Helper interno para cargar YAML con manejo de errores.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or (default or {})
    except FileNotFoundError:
        return default or {}


def load_promotion_config(path: str | None = None) -> dict:
    """
    Carga configuración de promoción desde config/promotion/promotion.yaml.
    """
    path = path or os.path.join(_CFG_DIR, "promotion", "promotion.yaml")
    return _load_yaml(path, default={})


def load_backtest_config(path: str | None = None) -> dict:
    """
    Carga configuración de backtest desde config/backtest/backtest.yaml.
    """
    path = path or os.path.join(_CFG_DIR, "backtest", "backtest.yaml")
    return _load_yaml(path, default={})


def load_risk_config_yaml(path: str | None = None) -> dict:
    """
    Carga configuración de riesgo desde config/trading/risk.yaml.
    Si no existe el archivo, devuelve configuración vacía.
    """
    path = path or os.path.join(_CFG_DIR, "trading", "risk.yaml")
    return _load_yaml(path, default={})


def load_heuristic_agents_config(path: str | None = None) -> dict:
    """
    Carga configuración de agentes heurísticos desde config/agents/heuristic_agents.yaml.
    Si no existe el archivo, devuelve configuración vacía.
    """
    path = path or os.path.join(_CFG_DIR, "agents", "heuristic_agents.yaml")
    return _load_yaml(path, default={})


def load_feature_updater_config(path: str | None = None) -> dict:
    """
    Carga configuración del feature updater desde config/data/feature_updater.yaml.
    Si no existe el archivo, devuelve configuración vacía.
    """
    path = path or os.path.join(_CFG_DIR, "data", "feature_updater.yaml")
    return _load_yaml(path, default={})