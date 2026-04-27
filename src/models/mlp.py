"""① MLP — Clasificador de acción de jugador propio.

Input: vector de features de 256 dimensiones (extraído de:
    a) crop del jugador pasado por el encoder de la CNN ② (avgpool de e4 -> 256-D), o
    b) features hand-crafted: 8 vectores velocidad de últimas 8 detecciones + altura de pelota
       relativa + posición jugador en cancha 2D (32-D) + zero-pad).

Output: 5 clases — saque / ataque / bloqueo / recepción / colocación.

Arquitectura: 3 capas fully-connected con dropout, BatchNorm y ReLU. Diseño clásico
de MLP — sirve para defender activación, dropout y backprop en oral.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ACTION_CLASSES


class ActionMLP(nn.Module):
    def __init__(self, in_features: int = 256, hidden: int = 128,
                 num_classes: int = len(ACTION_CLASSES), dropout: float = 0.3) -> None:
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
