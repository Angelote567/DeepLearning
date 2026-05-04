"""① MLP — Clasificador de zona de cancha del jugador (custom, 100% PyTorch).

Diseño:
* Input: vector 14-D de features hand-crafted extraídas del bbox + crop del jugador
  - 6 geométricas: cx, cy, w, h, aspect_ratio, area_relativa
  - 8 estadísticas: R_mean, G_mean, B_mean, brillo, std_R, std_G, std_B, contraste
* Salida: 3 clases — lejos / medio / cerca (zona en la imagen, respecto a la cámara)
* Etiquetas auto-generadas a partir de cy del bbox (sin etiquetado manual).

Arquitectura: 3 capas fully-connected con BatchNorm + ReLU + Dropout — el clásico
MLP que se defiende fácil en oral (activación, regularización, backprop).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MLP_FEATURES, ZONE_CLASSES


class ActionMLP(nn.Module):
    def __init__(self, in_features: int = MLP_FEATURES, hidden: int = 64,
                 num_classes: int = len(ZONE_CLASSES), dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        cls = probs.argmax(dim=-1)
        return cls, probs


def cy_to_zone(cy: float) -> int:
    """Devuelve el id de zona (0=lejos, 1=medio, 2=cerca) a partir del cy normalizado."""
    if cy < 0.45:
        return 0
    if cy < 0.72:
        return 1
    return 2
