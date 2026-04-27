"""② CNN — Heatmap de pelota propio (estilo TrackNet, NO importa la librería).

Idea: dadas N=3 frames consecutivos (concatenados en canal -> entrada de 9 canales),
la red predice un mapa de calor (1 canal) donde el pico indica la posición de la pelota.
Esto resuelve el problema clásico de detección de pelota pequeña/desenfocada que el
detector general (③) falla a menudo.

Arquitectura: encoder-decoder con skip connections (estilo U-Net miniatura).
Definida capa a capa, sin usar `torchvision.models`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import HEATMAP_INPUT_FRAMES, HEATMAP_SIZE


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BallHeatmapNet(nn.Module):
    """Encoder-Decoder propio. Input (B, 3*N, H, W) → Output (B, 1, H, W) sigmoid."""

    def __init__(self, in_frames: int = HEATMAP_INPUT_FRAMES) -> None:
        super().__init__()
        in_ch = 3 * in_frames
        self.enc1 = _double_conv(in_ch, 32)
        self.enc2 = _double_conv(32, 64)
        self.enc3 = _double_conv(64, 128)
        self.enc4 = _double_conv(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _double_conv(64, 32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def gaussian_target(cx: float, cy: float, h: int, w: int, sigma: float = 4.0) -> torch.Tensor:
    """Genera un heatmap target gaussiano centrado en (cx, cy)."""
    y = torch.arange(h, dtype=torch.float32).view(-1, 1)
    x = torch.arange(w, dtype=torch.float32).view(1, -1)
    return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal-style weighted MSE — penaliza más los píxeles positivos (raros)."""
    pos_mask = target > 0.1
    pos = F.mse_loss(pred[pos_mask], target[pos_mask], reduction="sum") if pos_mask.any() else 0.0
    neg = F.mse_loss(pred[~pos_mask], target[~pos_mask], reduction="sum") if (~pos_mask).any() else 0.0
    n_pos = pos_mask.sum().clamp(min=1)
    return (5.0 * pos + neg) / n_pos


def heatmap_to_point(heatmap: torch.Tensor, threshold: float = 0.3) -> tuple[int, int] | None:
    """Devuelve (cx, cy) del pico más alto si supera threshold; None si la pelota no está visible."""
    flat = heatmap.flatten()
    val, idx = flat.max(0)
    if val.item() < threshold:
        return None
    h, w = heatmap.shape[-2:]
    return (int(idx.item() % w), int(idx.item() // w))
