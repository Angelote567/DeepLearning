"""Entrenamiento del detector ③ — invocable como script o desde la GUI.

Uso desde Colab:
    !python -m src.train.train_detector --epochs 80 --batch 16 --lr 1e-3

Uso desde GUI: ver `gui.tabs.train.TrainTab` que llama a `train(...)` con los parámetros
seleccionados por el usuario (epochs, lr, batch).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DATASET_DIR, WEIGHTS_DIR
from src.data.visual_dataset import VolleyYoloDataset
from src.models.detector import VolleyDetector, DetectorLoss


def train(epochs: int = 80, batch_size: int = 16, lr: float = 1e-3,
          dataset_dir: Path | None = None, weights_dir: Path | None = None,
          device: str | None = None, log_cb=None) -> Path:
    """Loop de entrenamiento. `log_cb(epoch, metrics)` permite a la GUI suscribirse al progreso."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    weights_dir = Path(weights_dir or WEIGHTS_DIR)
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VolleyYoloDataset(dataset_dir, split="train", augment=True)
    val_ds = VolleyYoloDataset(dataset_dir, split="valid", augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VolleyDetector().to(device)
    loss_fn = DetectorLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_path = weights_dir / "detector.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in tqdm(train_dl, desc=f"epoch {epoch}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss, parts = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                loss, _ = loss_fn(model(imgs), targets)
                val_loss += loss.item()
        val_loss /= len(val_dl)

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "lr": scheduler.get_last_lr()[0]}
        print(metrics)
        if log_cb is not None:
            log_cb(epoch, metrics)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)

    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    t0 = time.time()
    out = train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    print(f"Best weights: {out} ({time.time()-t0:.1f}s)")
