"""Entrenamiento del módulo ② — CNN heatmap pelota.

Usa BallHeatmapDataset para leer las imágenes con pelota anotada y entrena la red
BallHeatmapNet contra un GT gaussiano. La pérdida es focal-weighted MSE (penaliza
más los píxeles positivos, que son raros).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DATASET_DIR, WEIGHTS_DIR
from src.data.heatmap_dataset import BallHeatmapDataset
from src.models.cnn import BallHeatmapNet, heatmap_loss


def train(epochs: int = 60, batch_size: int = 8, lr: float = 1e-3,
          dataset_dir: Path | None = None, weights_dir: Path | None = None,
          device: str | None = None, log_cb=None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    weights_dir = Path(weights_dir or WEIGHTS_DIR)
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BallHeatmapDataset(dataset_dir, split="train")
    val_ds = BallHeatmapDataset(dataset_dir, split="valid")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BallHeatmapNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_path = weights_dir / "ball_heatmap.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in tqdm(train_dl, desc=f"epoch {epoch}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = heatmap_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(len(train_dl), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                val_loss += heatmap_loss(model(imgs), targets).item()
        val_loss /= max(len(val_dl), 1)

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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    out = train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    print("Best weights:", out)
