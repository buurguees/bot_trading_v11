"""
core/ml/encoders/multitf_encoder.py
-----------------------------------
MultiTFEncoder: encoder temporal + fusión multi-timeframe con atención.

Capas:
  2) Temporal Encoder (por TF):
     - Opción 'tcn': Conv1D dilatada con bloques residuales.
     - Opción 'lstm': BiLSTM + pooling.
  3) Cross-TF Attention:
     - MultiHeadAttention para fusionar embeddings de TFs.
     - Query por TF objetivo (p.ej., '1m') o token learnable.

Lectura de datos:
  - Lee de market.features por (symbol, timeframe).
  - Convierte JSONB smc_flags a columnas binarias (fvg_up, bos, etc.)
  - Normaliza numéricos con z-score por ventana (binarios y st_direction quedan tal cual).
  - Alinea por "última vela cerrada" de cada TF.
  - Si faltan barras para la ventana, devuelve None.

Salidas:
  - embeddings_por_tf: dict[tf] -> Tensor (batch, d_model)
  - fused_embedding: Tensor (batch, d_model)
  - attn_weights: Tensor (batch, n_heads, 1, n_tf)  (si query_tf único)

Uso (ejemplo):
  from core.ml.encoders.multitf_encoder import MultiTFEncoder, MultiTFEncoderConfig
  enc = MultiTFEncoder(MultiTFEncoderConfig())
  batch = enc.make_batch_from_db(symbol="BTCUSDT",
                                 tfs=["1m","5m","15m","1h","4h","1d"],
                                 window=256)
  out = enc.forward(batch["tensors"], query_tf="1m")
  fused = out["fused"]  # (1, d_model)

Autor: BOT TRADING v9.1
"""

from __future__ import annotations

import os
import json
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------------------------------------
# Configuración / Constantes
# ------------------------------------------------------------
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/trading_db")

logger = logging.getLogger("MultiTFEncoder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

TF_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# Features numéricas continuas (z-score)
NUM_FEATURES = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "ema_20", "ema_50", "ema_200", "atr_14", "obv",
    "supertrend"
]
# Señales discretas / binarias (sin z-score)
DISCRETE_FEATURES = [
    "st_direction"  # 1 / -1
]
# Flags SMC en JSONB -> columnas binarias
SMC_FLAG_KEYS = [
    "fvg_up", "fvg_dn", "swing_high", "swing_low",
    "bos", "choch", "ob_bull", "ob_bear"
]

# ------------------------------------------------------------
# Utilidades de tiempo
# ------------------------------------------------------------
def last_closed_ts(timeframe: str) -> datetime:
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    tf_ms = TF_MS.get(timeframe, 60_000)
    last_closed_ms = (now_ms // tf_ms) * tf_ms - tf_ms
    return datetime.fromtimestamp(last_closed_ms / 1000, tz=timezone.utc)

# ------------------------------------------------------------
# Bloques de red
# ------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    """Bloque Conv1D dilatado con residual + LayerNorm + GELU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        residual = x
        y = self.conv(x)  # (B, C_out, L)
        # LayerNorm espera (B, L, C) → trasponemos temporalmente
        y = y.permute(0, 2, 1)         # (B, L, C_out)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.permute(0, 2, 1)         # (B, C_out, L)
        if self.proj is not None:
            residual = self.proj(residual)
        return y + residual


class TemporalConvEncoder(nn.Module):
    """Encoder temporal TCN: pila de ResidualBlock1D con dilataciones crecientes."""
    def __init__(self, in_ch: int, hidden: int = 128, layers: int = 3, kernel_size: int = 3, base_dilation: int = 1, dropout: float = 0.1, d_model: int = 128):
        super().__init__()
        blocks = []
        c_in = in_ch
        for i in range(layers):
            dilation = base_dilation * (2 ** i)
            blocks.append(ResidualBlock1D(c_in, hidden, kernel_size, dilation, dropout))
            c_in = hidden
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)   # pool sobre L
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> convertir a (B, C, L)
        x = x.permute(0, 2, 1)
        y = self.blocks(x)                  # (B, hidden, L)
        y = self.pool(y).squeeze(-1)        # (B, hidden)
        y = self.proj(y)                    # (B, d_model)
        return y


class BiLSTMEncoder(nn.Module):
    """Encoder temporal BiLSTM con mean-pooling + proyección."""
    def __init__(self, in_ch: int, hidden: int = 128, layers: int = 1, dropout: float = 0.1, d_model: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y, _ = self.lstm(x)                 # (B, L, 2H)
        y = y.mean(dim=1)                   # (B, 2H)
        y = self.proj(y)                    # (B, d_model)
        return y


class CrossTFAttention(nn.Module):
    """Fusión de embeddings de TFs con Multi-Head Attention.
       - keys/values: embeddings de todos los TFs (secuencias de longitud n_tf)
       - query: embedding del TF 'query_tf' o un token learnable si no se especifica
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, use_learnable_query: bool = False):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.use_learnable_query = use_learnable_query
        if use_learnable_query:
            self.q_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tf_embed_stack: torch.Tensor, query_index: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tf_embed_stack: (B, n_tf, d_model)
        query_index: índice del TF que actúa como query; si None y use_learnable_query=True -> token
        returns: (fused (B, d_model), attn_weights (B, n_heads, 1, n_tf))
        """
        B, n_tf, d = tf_embed_stack.shape

        if self.use_learnable_query or query_index is None:
            q = self.q_token.expand(B, -1, -1)  # (B, 1, d)
        else:
            q = tf_embed_stack[:, query_index:query_index+1, :]  # (B, 1, d)

        k = tf_embed_stack                              # (B, n_tf, d)
        v = tf_embed_stack
        fused, attn = self.mha(q, k, v, need_weights=True)  # fused: (B, 1, d), attn: (B, n_heads, 1, n_tf)
        fused = self.norm(fused.squeeze(1))                 # (B, d)
        return fused, attn


# ------------------------------------------------------------
# Config del encoder
# ------------------------------------------------------------
@dataclass
class MultiTFEncoderConfig:
    encoder_type: str = "tcn"           # "tcn" | "lstm"
    in_features: List[str] = field(default_factory=lambda: NUM_FEATURES + DISCRETE_FEATURES + SMC_FLAG_KEYS)
    d_model: int = 128
    hidden: int = 128
    layers: int = 3
    kernel_size: int = 3
    base_dilation: int = 1
    dropout: float = 0.1
    n_heads: int = 4
    query_tf_default: str = "1m"        # TF que actúa como query por defecto
    device: str = "cpu"                 # "cpu" | "cuda"

# ------------------------------------------------------------
# Encoder principal + acceso a BD
# ------------------------------------------------------------
class MultiTFEncoder(nn.Module):
    """
    Orquesta encoders por TF y la atención multi-TF.
    También expone utilidades para construir batches desde PostgreSQL.
    """
    def __init__(self, cfg: MultiTFEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.encoder_type == "tcn":
            self.enc_factory = lambda in_ch: TemporalConvEncoder(
                in_ch=in_ch, hidden=cfg.hidden, layers=cfg.layers,
                kernel_size=cfg.kernel_size, base_dilation=cfg.base_dilation,
                dropout=cfg.dropout, d_model=cfg.d_model
            )
        elif cfg.encoder_type == "lstm":
            self.enc_factory = lambda in_ch: BiLSTMEncoder(
                in_ch=in_ch, hidden=cfg.hidden, layers=max(1, cfg.layers),
                dropout=cfg.dropout, d_model=cfg.d_model
            )
        else:
            raise ValueError("encoder_type debe ser 'tcn' o 'lstm'")

        # creamos encoders por TF bajo demanda (lazy) en forward
        self.tf_encoders = nn.ModuleDict({})
        self.cross = CrossTFAttention(d_model=cfg.d_model, n_heads=cfg.n_heads, use_learnable_query=False)

        # DB engine (lazy)
        self._engine = None

    # ----------------------------- DB helpers -----------------------------
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
        return self._engine

    def _fetch_features_window(self, symbol: str, timeframe: str, window: int, end_ts: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Carga las últimas 'window' filas <= última vela cerrada (o end_ts si se da).
        Retorna DataFrame con columnas requeridas o None si no hay suficientes filas.
        """
        lc = end_ts or last_closed_ts(timeframe)
        sql = text(f"""
            SELECT ts,
                   {", ".join(NUM_FEATURES + DISCRETE_FEATURES)},
                   smc_flags
            FROM market.features
            WHERE symbol = :symbol AND timeframe = :tf AND ts <= :lc
            ORDER BY ts DESC
            LIMIT :win
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"symbol": symbol, "tf": timeframe, "lc": lc, "win": window * 3}).mappings().all()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # expandir smc_flags
        if "smc_flags" in df.columns:
            flags_df = df["smc_flags"].apply(lambda x: (x if isinstance(x, dict) else (json.loads(x) if x else {})))
            flags_exp = pd.json_normalize(flags_df).reindex(columns=SMC_FLAG_KEYS)
            flags_exp = flags_exp.fillna(False).astype(bool)
            flags_exp.columns = SMC_FLAG_KEYS
            df = pd.concat([df.drop(columns=["smc_flags"]), flags_exp], axis=1)

        # orden ascendente y recortar a 'window' finales
        df = df.sort_values("ts").tail(window)

        # chequear suficientes filas
        if len(df) < window:
            return None
        return df.reset_index(drop=True)

    def _zscore(self, x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd is None or sd == 0 or math.isnan(sd):
            return x * 0.0
        return (x - mu) / sd

    def _prepare_tensor(self, df: pd.DataFrame, feature_list: List[str]) -> torch.Tensor:
        """
        Devuelve Tensor shape (1, L, C). Z-score a columnas numéricas;
        discretas y flags se dejan 0/1 y st_direction en {-1,1} (NaN→0).
        """
        mat = []
        for col in feature_list:
            if col in NUM_FEATURES:
                colv = self._zscore(df[col].astype(float)).fillna(0.0).values
            elif col in DISCRETE_FEATURES:
                colv = df[col].fillna(0).astype(float).values
            elif col in SMC_FLAG_KEYS:
                colv = df[col].fillna(False).astype(float).values
            else:
                # si no existe, columna cero
                colv = np.zeros(len(df), dtype=float)
            mat.append(colv)
        arr = np.stack(mat, axis=1)            # (C, L) -> queremos (L, C)
        arr = arr.T                             # (L, C)
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, L, C)
        return t

    # ----------------------------- Batching desde DB -----------------------------
    def make_batch_from_db(self, symbol: str, tfs: List[str], window: int = 256, end_ts_by_tf: Optional[Dict[str, datetime]] = None) -> Optional[Dict]:
        """
        Construye batch dict con tensores por TF después de leer de la BD.
        end_ts_by_tf: puedes fijar manualmente el ts final por TF; si None => usa última vela cerrada.
        """
        tensors = {}
        frames_ok = []
        for tf in tfs:
            end_ts = (end_ts_by_tf or {}).get(tf)
            df = self._fetch_features_window(symbol, tf, window, end_ts=end_ts)
            if df is None:
                logger.warning(f"[{symbol}][{tf}] ventana insuficiente para window={window}")
                return None
            tensors[tf] = self._prepare_tensor(df, self.cfg.in_features)
            frames_ok.append(df[["ts"]])

        # opcional: comprobar que las ventanas terminan en ts cercano (consistencia)
        # aquí asumimos que cada TF ya está en su grid temporal.
        return {"symbol": symbol, "tfs": tfs, "window": window, "tensors": tensors}

    # ----------------------------- Forward -----------------------------
    def forward(self, tensors_by_tf: Dict[str, torch.Tensor], query_tf: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        tensors_by_tf: dict[tf] -> Tensor (B, L, C)
        query_tf: TF que actúa como query para la atención (default cfg.query_tf_default)
        return: {"per_tf": dict[tf]->(B,d_model), "fused": (B,d_model), "attn": (B, n_heads, 1, n_tf)}
        """
        device = torch.device(self.cfg.device)
        # Asegurar encoders por TF
        per_tf_embed = []
        tf_order = []
        for tf, tens in tensors_by_tf.items():
            tens = tens.to(device)
            if tf not in self.tf_encoders:
                self.tf_encoders[tf] = self.enc_factory(in_ch=tens.shape[-1])
            self.tf_encoders[tf].to(device)
            emb = self.tf_encoders[tf](tens)      # (B, d_model)
            per_tf_embed.append(emb)
            tf_order.append(tf)

        # Pila (B, n_tf, d_model)
        stack = torch.stack(per_tf_embed, dim=1)
        # Índice de query
        q_tf = query_tf or self.cfg.query_tf_default
        q_idx = tf_order.index(q_tf) if q_tf in tf_order else None

        fused, attn = self.cross(stack, q_idx)    # (B,d_model), (B,heads,1,n_tf)

        per_tf_dict = {tf: emb for tf, emb in zip(tf_order, per_tf_embed)}
        return {"per_tf": per_tf_dict, "fused": fused, "attn": attn}

    @staticmethod
    def from_training_yaml(path: str = "config/train/training.yaml") -> "MultiTFEncoder":
        """
        Factory method para crear MultiTFEncoder desde training.yaml.
        
        Args:
            path: Ruta al archivo training.yaml
            
        Returns:
            MultiTFEncoder configurado según training.yaml
        """
        from core.config.config_loader import load_training_config, build_encoder_config_from_training
        
        cfg_train = load_training_config(path)
        enc_cfg = build_encoder_config_from_training(cfg_train)
        return MultiTFEncoder(enc_cfg)

# ------------------------------------------------------------
# Modo script simple para depuración rápida
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de uso con configuración desde YAML (recomendado)
    enc = MultiTFEncoder.from_training_yaml()
    
    # Cargar símbolos y TFs desde configuración
    from core.config.config_loader import load_symbols_config, load_training_config, extract_symbols_and_tfs, get_window_from_training
    
    cfg_symbols = load_symbols_config()
    cfg_train = load_training_config()
    symbols, tfs = extract_symbols_and_tfs(cfg_symbols, cfg_train)
    window = get_window_from_training(cfg_train)
    
    # Usar configuración para crear batch
    batch = enc.make_batch_from_db(
        symbol=symbols[0], 
        tfs=tfs[:3],  # Primeros 3 TFs para ejemplo
        window=min(window, 64)  # Ventana pequeña para ejemplo
    )
    
    if batch:
        out = enc.forward(batch["tensors"], query_tf=enc.cfg.query_tf_default)
        fused = out["fused"]
        print("Fused embedding shape:", tuple(fused.shape))
        print("TFs:", list(out["per_tf"].keys()))
        print("Configuración cargada desde YAML exitosamente!")
    else:
        print("No se pudo crear batch - verificar datos en market.features")
