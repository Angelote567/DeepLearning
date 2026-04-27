"""④ LSTM — Predicción de trayectoria de pelota + clasificación táctica.

Inspirado en el framework PathFinder (volleyIEEE/VolleyStats) pero implementado
desde cero en PyTorch. NO se importa ninguna red pre-entrenada.

Modelo dual-head:
* Encoder: LSTM de 2 capas (input: secuencia de 30 coords (x,y) en cancha 2D).
* Head A — predicción regresiva: 10 coords futuras (x,y).
* Head B — clasificación táctica: 4 clases (ataque potente / finta / bloqueo / saque).

Math review (para defensa oral):
    h_t, c_t = LSTM(x_t, (h_{t-1}, c_{t-1}))
    forget = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
    input  = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
    output = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
    cand   = tanh(W_c · [h_{t-1}, x_t] + b_c)
    c_t = forget * c_{t-1} + input * cand
    h_t = output * tanh(c_t)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LSTM_INPUT_LEN, LSTM_OUTPUT_LEN

TACTIC_CLASSES = ["ataque_potente", "finta", "bloqueo", "saque"]


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 128, num_layers: int = 2,
                 horizon: int = LSTM_OUTPUT_LEN, num_tactics: int = len(TACTIC_CLASSES),
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.coord_head = nn.Linear(hidden, input_dim)
        self.tactic_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_tactics),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, 2). Returns (predicted_xy: (B, horizon, 2), tactic_logits: (B, num_tactics))."""
        _, (h, c) = self.encoder(x)

        # Auto-regresivo: empezamos con la última coord conocida y predecimos paso a paso
        last = x[:, -1:, :]
        outputs = []
        for _ in range(self.horizon):
            out, (h, c) = self.decoder(last, (h, c))
            pred = self.coord_head(out)
            outputs.append(pred)
            last = pred                                  # teacher forcing off — usamos pred como next input
        coords = torch.cat(outputs, dim=1)

        # Clasificación táctica usa el último hidden state de la capa final
        tactic_logits = self.tactic_head(h[-1])
        return coords, tactic_logits


def trajectory_loss(pred_xy: torch.Tensor, true_xy: torch.Tensor,
                    pred_tactic: torch.Tensor, true_tactic: torch.Tensor,
                    alpha: float = 1.0, beta: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Pérdida combinada: MSE en regresión de coords + CE en táctica."""
    coord_loss = F.mse_loss(pred_xy, true_xy)
    tactic_loss = F.cross_entropy(pred_tactic, true_tactic)
    total = alpha * coord_loss + beta * tactic_loss
    return total, {"coord": coord_loss.item(), "tactic": tactic_loss.item()}
