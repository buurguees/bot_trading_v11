"""
Head SMC (Sesgo basado en Smart Money Concepts)
===============================================
Lee:    - Vector común (128) desde BaseFusionModel (usa market.features con flags SMC)
Escribe: - No escribe en BD (eso lo hace el agente).

Qué hace:
- Usa la información fusionada multi-TF (que ya incluye flags SMC como features)
  para evaluar sesgo SMC en {bull, bear, neutral}.

Funciones:
- SMCHead: MLP 128->64->3 (bull/bear/neutral).
- SMCModel.predict_from_db(...): devuelve dict con label y probs.

Notas:
- La cabeza no calcula los flags; eso lo hace feature_engineer y vienen como input del encoder.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from core.ml.models.base_model import BaseFusionModel
from core.ml.encoders.multitf_encoder import MultiTFEncoderConfig


CLASSES = ["bull", "bear", "neutral"]

class SMCHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

class SMCModel(nn.Module):
    def __init__(self, enc_cfg: MultiTFEncoderConfig, head_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.base = BaseFusionModel(enc_cfg, dropout=dropout)
        self.head = SMCHead(in_dim=enc_cfg.d_model, hidden=head_hidden, n_classes=len(CLASSES), dropout=dropout)

    @torch.no_grad()
    def predict_from_db(self, symbol: str, tfs: list, window: int = 256, query_tf: Optional[str] = None):
        common, meta = self.base.compute_common_from_db(symbol, tfs, window, query_tf)
        if common is None:
            return None
        logits = self.head(common)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        idx = int(torch.argmax(logits, dim=-1).item())
        return {
            "symbol": symbol,
            "label": CLASSES[idx],
            "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "meta": meta,
        }
