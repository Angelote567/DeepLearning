"""Dataset para el LSTM ④ — secuencias de trayectoria de pelota.

Formato esperado: archivos JSON en `data/trajectories/` con:
    {
        "frames": [...],
        "ball": [[x0,y0], [x1,y1], ...],         # coords en cancha 2D normalizadas [0,1]
        "tactic": "ataque_potente"               # una etiqueta por jugada
    }

Si no hay trayectorias reales aún, se puede generar sintéticamente con `synthetic_trajectory()`
que simula parábolas con ruido — útil para arrancar antes de tener datos reales anotados.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import LSTM_INPUT_LEN, LSTM_OUTPUT_LEN
from src.models.lstm import TACTIC_CLASSES


class TrajectoryDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train",
                 input_len: int = LSTM_INPUT_LEN, output_len: int = LSTM_OUTPUT_LEN) -> None:
        self.root = Path(root) / split
        self.input_len = input_len
        self.output_len = output_len
        self.samples = sorted(self.root.glob("*.json"))
        if not self.samples:
            print(f"[warn] No trajectories in {self.root}; usa synthetic_dataset() para arrancar")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(self.samples[idx]) as f:
            sample = json.load(f)
        ball = np.array(sample["ball"], dtype=np.float32)
        tactic = TACTIC_CLASSES.index(sample["tactic"])

        if len(ball) < self.input_len + self.output_len:
            pad = self.input_len + self.output_len - len(ball)
            ball = np.concatenate([np.repeat(ball[:1], pad, axis=0), ball], axis=0)
        ball = ball[-(self.input_len + self.output_len):]
        x = torch.from_numpy(ball[: self.input_len])
        y = torch.from_numpy(ball[self.input_len :])
        return x, y, torch.tensor(tactic, dtype=torch.long)


def synthetic_trajectory(num: int = 1000, input_len: int = LSTM_INPUT_LEN,
                          output_len: int = LSTM_OUTPUT_LEN) -> list[dict]:
    """Genera trayectorias parabólicas sintéticas para validar el pipeline LSTM antes de
    tener datos anotados reales. Cada táctica tiene un perfil físico distinto."""
    profiles = {
        "ataque_potente": {"vx": 0.8, "vy": -0.6, "g": 1.4},
        "finta":          {"vx": 0.4, "vy": -0.3, "g": 0.8},
        "bloqueo":        {"vx": 0.0, "vy": 0.0, "g": 0.5},
        "saque":          {"vx": 0.6, "vy": -0.5, "g": 1.0},
    }
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(num):
        tactic = rng.choice(TACTIC_CLASSES)
        p = profiles[tactic]
        t = np.linspace(0, 1, input_len + output_len)
        x = p["vx"] * t + 0.5 + rng.normal(0, 0.02, size=t.shape)
        y = 0.5 + p["vy"] * t + 0.5 * p["g"] * t ** 2 + rng.normal(0, 0.02, size=t.shape)
        ball = np.stack([np.clip(x, 0, 1), np.clip(y, 0, 1)], axis=1)
        samples.append({"ball": ball.tolist(), "tactic": tactic})
    return samples
