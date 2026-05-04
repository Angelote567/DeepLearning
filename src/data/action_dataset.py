"""Dataset para el MLP ① — clasificador de zona de cancha del jugador.

Para cada bbox de clase `player` (id=1) en el dataset Roboflow:
* Extrae el crop de la imagen
* Computa 14 features (geométricas + estadísticas de color)
* La etiqueta es auto-generada con `cy_to_zone(cy)` a partir del cy del bbox

Resultado: dataset PyTorch sin etiquetado manual, listo para entrenar el MLP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import MLP_FEATURES
from src.models.mlp import cy_to_zone

PLAYER_CLASS_ID = 1   # según src.config.CLASSES = ["ball", "player", "referee"]


def extract_features(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """img: array RGB (H, W, 3) en [0, 255]. bbox: (cx, cy, w, h) normalizado [0, 1].
    Devuelve vector (14,) float32.
    """
    H, W = img.shape[:2]
    cx, cy, w, h = bbox
    x1 = max(0, int((cx - w / 2) * W))
    y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W, int((cx + w / 2) * W))
    y2 = min(H, int((cy + h / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(MLP_FEATURES, dtype=np.float32)

    crop = img[y1:y2, x1:x2].astype(np.float32) / 255.0
    aspect = w / h if h > 1e-6 else 0.0
    area = w * h

    r_mean, g_mean, b_mean = crop[..., 0].mean(), crop[..., 1].mean(), crop[..., 2].mean()
    brillo = (r_mean + g_mean + b_mean) / 3
    r_std, g_std, b_std = crop[..., 0].std(), crop[..., 1].std(), crop[..., 2].std()
    contraste = crop.std()

    return np.array([
        cx, cy, w, h, aspect, area,
        r_mean, g_mean, b_mean, brillo,
        r_std, g_std, b_std, contraste,
    ], dtype=np.float32)


class ZoneDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root) / split
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.samples: list[tuple[Path, tuple[float, float, float, float], int]] = []

        for img_path in sorted(self.images_dir.glob("*.[jp][pn]g")):
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or int(parts[0]) != PLAYER_CLASS_ID:
                        continue
                    cx, cy, w, h = map(float, parts[1:5])
                    label = cy_to_zone(cy)
                    self.samples.append((img_path, (cx, cy, w, h), label))
        print(f"[{split}] {len(self.samples)} bboxes de jugador")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, bbox, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        feats = extract_features(img, bbox)
        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.long)
