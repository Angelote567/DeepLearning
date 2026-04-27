"""Dataset PyTorch que lee el formato YOLO de Roboflow (.txt por imagen).

Estructura esperada (descargada por notebooks/01_dataset_download.ipynb):

    data/voley.yolov8/
        train/images/*.jpg
        train/labels/*.txt   # cada línea: cls cx cy w h (normalizado [0,1])
        valid/images/...
        valid/labels/...
        test/images/...
        test/labels/...
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import GRID_SIZE, IMG_SIZE, NUM_CLASSES


class VolleyYoloDataset(Dataset):
    """Devuelve (image_tensor, target_grid).

    target_grid: (S, S, 5 + C) con (obj_mask, x, y, w, h, one-hot clase) en escala celda.
    Si una celda contiene varios objetos, se queda con el de mayor área (heurística).
    """

    def __init__(self, root: str | Path, split: str = "train", img_size: int = IMG_SIZE,
                 grid: int = GRID_SIZE, augment: bool = False) -> None:
        self.root = Path(root) / split
        self.img_size = img_size
        self.grid = grid
        self.augment = augment and split == "train"

        self.images = sorted((self.root / "images").glob("*.[jp][pn]g"))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.root / 'images'}")

        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        label_path = self.root / "labels" / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        if self.augment:
            img = self._augment(img)
        img_tensor = self.to_tensor(img)

        target = self._build_target(label_path)
        return img_tensor, target

    def _augment(self, img: Image.Image) -> Image.Image:
        # Augment muy ligero — flips se aplican en GPU vía colate si queremos. Por ahora solo color.
        if torch.rand(1).item() < 0.5:
            img = transforms.functional.adjust_brightness(img, 1 + (torch.rand(1).item() - 0.5) * 0.4)
        if torch.rand(1).item() < 0.5:
            img = transforms.functional.adjust_contrast(img, 1 + (torch.rand(1).item() - 0.5) * 0.4)
        return img

    def _build_target(self, label_path: Path) -> torch.Tensor:
        """target shape: (S, S, 5 + C). Coords (x, y) son offset dentro de celda en [0,1],
        (w, h) son proporción de imagen en [0,1]."""
        S = self.grid
        target = torch.zeros((S, S, 5 + NUM_CLASSES), dtype=torch.float32)
        if not label_path.exists():
            return target

        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Si caen 2 objetos en la misma celda, nos quedamos con el de mayor área
        cell_taken: dict[tuple[int, int], float] = {}
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            if cls_id >= NUM_CLASSES:
                continue
            gx = int(cx * S)
            gy = int(cy * S)
            gx = min(gx, S - 1)
            gy = min(gy, S - 1)
            area = w * h
            if (gy, gx) in cell_taken and cell_taken[(gy, gx)] > area:
                continue
            cell_taken[(gy, gx)] = area

            local_x = cx * S - gx
            local_y = cy * S - gy
            target[gy, gx, 0] = 1.0
            target[gy, gx, 1] = local_x
            target[gy, gx, 2] = local_y
            target[gy, gx, 3] = w
            target[gy, gx, 4] = h
            target[gy, gx, 5:] = 0.0
            target[gy, gx, 5 + cls_id] = 1.0
        return target
