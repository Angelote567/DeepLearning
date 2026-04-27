"""Mapeo cancha — homografía 4 puntos manualmente seleccionados → cancha 2D FIVB 18×9 m.

Inspirado en TheSmike/VolleyCourt-mapping (que usa OpenCV findHomography). Aquí lo
implementamos directamente con cv2 y exponemos `image_to_court(...)` que la GUI usa
para proyectar las detecciones a la cancha 2D mostrada en el panel lateral.
"""
from __future__ import annotations

import numpy as np
import cv2

from src.config import COURT_H_M, COURT_W_M


CANVAS_W = 720   # px del minicampo en la GUI
CANVAS_H = int(CANVAS_W * COURT_H_M / COURT_W_M)


def compute_homography(image_corners: np.ndarray) -> np.ndarray:
    """image_corners: (4, 2) en orden top-left, top-right, bottom-right, bottom-left.
    Devuelve matriz 3x3 que mapea píxeles de la imagen a coordenadas del minicampo (px)."""
    court_corners = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(image_corners.astype(np.float32), court_corners)
    return H


def image_to_court(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Proyecta puntos (N, 2) de imagen → cancha 2D usando la homografía H."""
    if len(points) == 0:
        return points
    pts = np.concatenate([points, np.ones((len(points), 1))], axis=1)  # (N, 3)
    proj = pts @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def draw_court_template(canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H) -> np.ndarray:
    """Dibuja una cancha de voleibol vacía (líneas FIVB) sobre la que pintar detecciones."""
    img = np.full((canvas_h, canvas_w, 3), 220, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (canvas_w - 1, canvas_h - 1), (0, 0, 0), 2)
    cx = canvas_w // 2
    cv2.line(img, (cx, 0), (cx, canvas_h), (0, 0, 0), 2)             # línea de red
    # líneas de ataque a 3 m de la red
    attack_offset = int(canvas_w * 3 / COURT_W_M)
    cv2.line(img, (cx - attack_offset, 0), (cx - attack_offset, canvas_h), (180, 180, 180), 1)
    cv2.line(img, (cx + attack_offset, 0), (cx + attack_offset, canvas_h), (180, 180, 180), 1)
    return img
