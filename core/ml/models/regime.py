"""
Head Regime (Trend vs Range + Volatility Buckets)
=================================================
Lee:    - Vector común (128) desde BaseFusionModel (que a su vez usa market.features)
Escribe: - No escribe en BD (eso lo hace el agente).

Qué hace:
- Clasifica el "régimen" en {trend, range}.
- Estima buckets de volatilidad (Q1..Q4) como salida adicional (incluida en el JSON de probs).

Funciones:
- RegimeHeadMain: MLP 128->64->2 (trend/range).
- RegimeHeadVol:  MLP 128->32->4  (vol_q1..vol_q4).
- RegimeModel.predict_from_db(...): devuelve {"label": "trend|range", "probs": {...}}.

Nota: El agente escribirá task='regime' con pred_label trend|range y
dejará los vol_q* en el JSON de probs para auditoría.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from core.ml.models.base_model import BaseFusionModel
from core.ml.encoders.multitf_encoder import MultiTFEncoderConfig


REGIME = ["trend", "range"]
VOL = ["vol_q1", "vol_q2", "vol_q3", "vol_q4"]

class RegimeHeadMain(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 64, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

class RegimeHeadVol(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 32, n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

class RegimeModel(nn.Module):
    def __init__(self, enc_cfg: MultiTFEncoderConfig, main_hidden: int = 64, vol_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.base = BaseFusionModel(enc_cfg, dropout=dropout)
        self.main = RegimeHeadMain(in_dim=enc_cfg.d_model, hidden=main_hidden, n_classes=len(REGIME), dropout=dropout)
        self.vol  = RegimeHeadVol (in_dim=enc_cfg.d_model, hidden=vol_hidden,  n_classes=len(VOL),    dropout=dropout)

    @torch.no_grad()
    def predict_from_db(self, symbol: str, tfs: list, window: int = 256, query_tf: Optional[str] = None):
        common, meta = self.base.compute_common_from_db(symbol, tfs, window, query_tf)
        if common is None:
            return None

        # Régimen
        logits_r = self.main(common)
        probs_r = torch.softmax(logits_r, dim=-1)[0].cpu().numpy().tolist()
        idx_r = int(torch.argmax(logits_r, dim=-1).item())

        # Vol buckets
        logits_v = self.vol(common)
        probs_v = torch.softmax(logits_v, dim=-1)[0].cpu().numpy().tolist()
        idx_v = int(torch.argmax(logits_v, dim=-1).item())

        probs = {REGIME[i]: float(probs_r[i]) for i in range(len(REGIME))}
        probs.update({VOL[i]: float(probs_v[i]) for i in range(len(VOL))})
        probs["vol_bucket"] = VOL[idx_v]

        return {
            "symbol": symbol,
            "label": REGIME[idx_r],  # trend | range
            "probs": probs,
            "meta": meta,
        }
