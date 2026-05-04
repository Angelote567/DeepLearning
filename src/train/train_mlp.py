"""Entrenamiento del MLP ① — zona de cancha del jugador."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import DATASET_DIR, WEIGHTS_DIR
from src.data.action_dataset import ZoneDataset
from src.models.mlp import ActionMLP


def train(epochs: int = 40, batch_size: int = 64, lr: float = 1e-3,
          dataset_dir: Path | None = None, weights_dir: Path | None = None,
          device: str | None = None, log_cb=None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(dataset_dir or DATASET_DIR)
    weights_dir = Path(weights_dir or WEIGHTS_DIR)
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ZoneDataset(dataset_dir, split="train")
    val_ds = ZoneDataset(dataset_dir, split="valid")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ActionMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = weights_dir / "action_mlp.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        train_correct = 0
        for feats, labels in train_dl:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(-1) == labels).sum().item()
            n_train += len(labels)

        model.eval()
        val_loss, n_val = 0.0, 0
        val_correct = 0
        with torch.no_grad():
            for feats, labels in val_dl:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                val_loss += loss_fn(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                n_val += len(labels)

        train_loss /= n_train; val_loss /= n_val
        train_acc = train_correct / n_train
        val_acc = val_correct / n_val
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "train_acc": train_acc, "val_acc": val_acc}
        print(metrics)
        if log_cb is not None:
            log_cb(epoch, metrics)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    out = train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    print("Best weights:", out)
