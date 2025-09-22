"""
Capa 4 - Fusion Layer (BaseFusionModel)
=======================================
Lee:    - Embeddings que genera MultiTFEncoder (de market.features vía encoder.make_batch_from_db)
Escribe: - No escribe en BD; expone vectores comunes para las cabezas (heads).

Funciones:
- FusionProjection: Linear(256->128) + LayerNorm + GELU + Dropout.
- BaseFusionModel.compute_common_from_tensors(tensors_by_tf, query_tf):
    Concatena [embedding_query_tf, embedding_fusionado] -> proyecta a 128.
- BaseFusionModel.compute_common_from_db(symbol, tfs, window, query_tf):
    Construye batch desde BD (market.features) y devuelve el vector común.

Notas:
- El tamaño del embedding por TF es d_model (p.ej. 128).
- El fusionado por atención también es d_model. Se concatenan -> 2*d_model -> 128.
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn

from core.ml.encoders.multitf_encoder import MultiTFEncoder, MultiTFEncoderConfig


class FusionProjection(nn.Module):
    """Dense 2*d_model -> d_model + LayerNorm + GELU + Dropout."""
    def __init__(self, in_dim: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        return y


class BaseFusionModel(nn.Module):
    """
    Envuelve un MultiTFEncoder y la proyección a vector común.
    """
    def __init__(self, enc_cfg: MultiTFEncoderConfig, dropout: float = 0.1):
        super().__init__()
        self.encoder = MultiTFEncoder(enc_cfg)
        self.d_model = enc_cfg.d_model
        self.fusion = FusionProjection(in_dim=self.d_model * 2, out_dim=self.d_model, dropout=dropout)
        self.query_tf_default = enc_cfg.query_tf_default
        self.device = torch.device(enc_cfg.device)

    @torch.no_grad()
    def compute_common_from_tensors(self, tensors_by_tf: Dict[str, torch.Tensor], query_tf: Optional[str] = None) -> torch.Tensor:
        """
        Concatena [embedding del TF de ejecución + embedding fusionado por atención] y proyecta a 128.
        """
        out = self.encoder.forward(tensors_by_tf, query_tf=query_tf or self.query_tf_default)
        fused = out["fused"]                      # (B, d_model)
        per_tf = out["per_tf"]
        q_tf = query_tf or self.query_tf_default
        q_emb = per_tf[q_tf]                      # (B, d_model)
        both = torch.cat([q_emb, fused], dim=-1)  # (B, 2*d_model)
        common = self.fusion(both.to(self.device))
        return common

    @torch.no_grad()
    def compute_common_from_db(self, symbol: str, tfs: list, window: int = 256, query_tf: Optional[str] = None):
        """
        Construye batch desde BD y devuelve el vector común (1, d_model) + metadatos.
        """
        batch = self.encoder.make_batch_from_db(symbol=symbol, tfs=tfs, window=window)
        if not batch:
            return None, None
        common = self.compute_common_from_tensors(batch["tensors"], query_tf=query_tf)
        meta = {"symbol": symbol, "tfs": tfs, "window": window, "query_tf": query_tf or self.query_tf_default}
        return common, meta
