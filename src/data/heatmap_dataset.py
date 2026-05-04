"""Dataset para la CNN heatmap pelota ②.

Estrategia: leemos las anotaciones YOLO del dataset Roboflow (clase 0 = ball),
y generamos un heatmap GT gaussiano centrado en la posición anotada.

Para simplificar (y porque las imágenes Roboflow son frames sueltos sin secuencia
temporal), usamos solo 1 frame en lugar de 3 → input shape (3, H, W) en lugar de (9, H, W).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import HEATMAP_SIZE
from src.models.cnn import gaussian_target

BALL_CLASS_ID = 0   # según el orden en src.config.CLASSES = ["ball", "player", "referee"]


class BallHeatmapDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", size: int = HEATMAP_SIZE,
                 sigma: float = 4.0, only_with_ball: bool = True) -> None:
        self.root = Path(root) / split
        self.size = size
        self.sigma = sigma
        self.to_tensor = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        all_imgs = sorted((self.root / "images").glob("*.[jp][pn]g"))
        if only_with_ball:
            self.images = [p for p in all_imgs if self._has_ball(p)]
            print(f"[{split}] {len(self.images)}/{len(all_imgs)} imágenes con pelota anotada")
        else:
            self.images = all_imgs

    def _label_path(self, img_path: Path) -> Path:
        return self.root / "labels" / (img_path.stem + ".txt")

    def _has_ball(self, img_path: Path) -> bool:
        lp = self._label_path(img_path)
        if not lp.exists():
            return False
        with open(lp) as f:
            for line in f:
                if line.strip().startswith(f"{BALL_CLASS_ID} "):
                    return True
        return False

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.to_tensor(img)                                      # (3, H, W)
        # Replicamos el frame 3 veces para mantener la entrada (9, H, W) que espera la red
        x = torch.cat([x, x, x], dim=0)

        # GT heatmap gaussiano sobre la pelota anotada (puede haber 0 o 1)
        target = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        lp = self._label_path(img_path)
        if lp.exists():
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or int(parts[0]) != BALL_CLASS_ID:
                        continue
                    cx, cy = float(parts[1]), float(parts[2])
                    target[0] = gaussian_target(cx * self.size, cy * self.size,
                                                self.size, self.size, sigma=self.sigma)
                    break
        return x, target
