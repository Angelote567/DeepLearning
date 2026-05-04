"""Pipeline E2E — orquesta los 5 módulos sobre vídeo o imagen.

Flujo:
    frame → ③ detector → bboxes (jugadores, pelota, red)
          → ② CNN heatmap (refuerza la pelota cuando es muy pequeña)
          → cancha 2D (homografía) → coords en metros
          → ④ LSTM (acumula trayectoria pelota → predice futuro + táctica)
          → ① MLP (clasifica acción de cada jugador con features de crops)
          → ⑤ VLM (cada N frames, descripción en lenguaje natural)

Cada paso es opcional — si los pesos no existen aún, se deshabilita ese paso pero el resto
sigue funcionando. Esto permite ir trabajando módulo a módulo sin romper la GUI.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.config import (
    CLASSES,
    HEATMAP_SIZE,
    IMG_SIZE,
    LSTM_INPUT_LEN,
    WEIGHTS_DIR,
)
from src.inference.court import compute_homography, image_to_court
from src.models.cnn import BallHeatmapNet, heatmap_to_point
from src.models.detector import VolleyDetector
from src.models.lstm import TACTIC_CLASSES, TrajectoryLSTM
from src.models.mlp import ActionMLP


class VolleyPipeline:
    def __init__(self, weights_dir: Path | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path(weights_dir or WEIGHTS_DIR)

        self.detector: VolleyDetector | None = self._try_load(VolleyDetector(), "detector.pt")
        self.heatmap_net: BallHeatmapNet | None = self._try_load(BallHeatmapNet(), "ball_heatmap.pt")
        self.lstm: TrajectoryLSTM | None = self._try_load(TrajectoryLSTM(), "trajectory_lstm.pt")
        self.mlp: ActionMLP | None = self._try_load(ActionMLP(), "action_mlp.pt")

        self.homography: np.ndarray | None = None
        self.ball_history: deque = deque(maxlen=LSTM_INPUT_LEN)
        self.frame_buffer: deque = deque(maxlen=3)        # para CNN heatmap (3 frames)

    def _try_load(self, model: torch.nn.Module, filename: str) -> torch.nn.Module | None:
        path = self.weights_dir / filename
        if not path.exists():
            print(f"[pipeline] {filename} no encontrado — módulo deshabilitado")
            return None
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def set_court_corners(self, corners: np.ndarray) -> None:
        """corners: (4,2) — esquinas de la cancha en píxeles. Suelen seleccionarse en la GUI."""
        self.homography = compute_homography(corners)

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """Procesa un frame y devuelve detecciones + trayectoria + táctica."""
        result: dict = {"frame": frame_bgr, "detections": [], "ball_court": None,
                        "predicted_path": None, "tactic": None}
        h, w = frame_bgr.shape[:2]

        # ③ Detector
        if self.detector is not None:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = TF.to_tensor(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(self.device)
            preds = self.detector.predict(tensor)[0]
            sx, sy = w / IMG_SIZE, h / IMG_SIZE
            for box, score, cls in zip(preds["boxes"], preds["scores"], preds["classes"]):
                x1, y1, x2, y2 = box.tolist()
                result["detections"].append({
                    "box": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    "score": float(score),
                    "class": CLASSES[int(cls)],
                })

        # ② Heatmap pelota — solo si tenemos 3 frames acumulados
        ball_xy_image: tuple[float, float] | None = None
        self.frame_buffer.append(frame_bgr)
        if self.heatmap_net is not None and len(self.frame_buffer) == 3:
            stack = []
            for fr in self.frame_buffer:
                tn = TF.to_tensor(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((HEATMAP_SIZE, HEATMAP_SIZE)))
                stack.append(tn)
            x = torch.cat(stack, dim=0).unsqueeze(0).to(self.device)
            heat = self.heatmap_net(x)[0, 0]
            pt = heatmap_to_point(heat)
            if pt is not None:
                ball_xy_image = (pt[0] * w / HEATMAP_SIZE, pt[1] * h / HEATMAP_SIZE)
        if ball_xy_image is None:
            for d in result["detections"]:
                if d["class"] == "ball":
                    x1, y1, x2, y2 = d["box"]
                    ball_xy_image = ((x1 + x2) / 2, (y1 + y2) / 2)
                    break

        # ① MLP — clasifica zona del jugador (lejos/medio/cerca) usando features hand-crafted
        if self.mlp is not None and result["detections"]:
            from src.config import ZONE_CLASSES
            from src.data.action_dataset import extract_features
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            feats_batch = []
            player_idxs = []
            for i, d in enumerate(result["detections"]):
                if d["class"] != "player":
                    continue
                x1, y1, x2, y2 = d["box"]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                feats = extract_features(img_rgb, (cx, cy, bw, bh))
                feats_batch.append(feats)
                player_idxs.append(i)
            if feats_batch:
                fbatch = torch.tensor(np.array(feats_batch), dtype=torch.float32).to(self.device)
                cls, _ = self.mlp.predict(fbatch)
                for i, c in zip(player_idxs, cls.tolist()):
                    result["detections"][i]["zone"] = ZONE_CLASSES[c]

        # Mapeo a cancha + ④ LSTM trayectoria
        if ball_xy_image is not None and self.homography is not None:
            ball_court = image_to_court(np.array([ball_xy_image]), self.homography)[0]
            self.ball_history.append(ball_court / np.array([720, 360]))    # normalizado
            result["ball_court"] = ball_court.tolist()

            if self.lstm is not None and len(self.ball_history) == LSTM_INPUT_LEN:
                seq = torch.tensor(np.array(list(self.ball_history)),
                                   dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_xy, tactic_logits = self.lstm(seq)
                result["predicted_path"] = pred_xy[0].cpu().numpy().tolist()
                result["tactic"] = TACTIC_CLASSES[int(tactic_logits.argmax(-1).item())]

        return result

    def process_video(self, path: str | Path) -> Iterator[dict]:
        cap = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.process_frame(frame)
        finally:
            cap.release()
