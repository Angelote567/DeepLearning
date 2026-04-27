"""③ Object Detector — modelo PROPIO escrito desde cero en PyTorch puro.

============================================================
IMPORTANTE — esto NO es la librería YOLO (Ultralytics).
============================================================
* No se importa `ultralytics`, `yolov5`, `yolov8` ni similar.
* No se cargan pesos pre-entrenados (`pretrained=True` no aparece).
* Cada `nn.Conv2d`, `nn.BatchNorm2d`, la cabeza, la pérdida y el NMS
  están definidos a mano en este archivo o en `src/train/utils.py`.
* La inspiración es el ALGORITMO YOLOv1 (Redmon et al. 2016) —
  detección single-shot basada en grid — pero el código y la
  arquitectura concreta son nuestros.

Decisiones de diseño (justificarlas en la defensa oral):

* Backbone: 6 bloques Conv-BN-LeakyReLU + MaxPool. ~5M parámetros para
  entrenar en Colab T4 free en ~30 min con 630 imágenes.
* Cabeza: predice tensor (B, S, S, B*5 + C) con S=13, B=2 boxes/celda, C=3 clases.
  Cada predicción por celda = (x, y, w, h, conf) + softmax sobre clases.
* Pérdida: multi-task estilo YOLOv1 (coord + obj + noobj + cls) con λ_coord=5, λ_noobj=0.5.
* Inferencia: filtra por confidence threshold y aplica NMS class-aware
  escrito a mano en `src/train/utils.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    GRID_SIZE,
    IMG_SIZE,
    NUM_BOXES_PER_CELL,
    NUM_CLASSES,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_NMS_IOU,
)
from src.train.utils import class_aware_nms, xywh_to_xyxy


def _conv_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class VolleyDetector(nn.Module):
    """YOLOv1 simplificado. Input 3x416x416 → output (B, S, S, 2*5 + C).

    Reducción espacial 416→13 = 32 (5 maxpool de stride 2).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, num_boxes: int = NUM_BOXES_PER_CELL,
                 grid: int = GRID_SIZE) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid = grid

        self.backbone = nn.Sequential(
            _conv_block(3, 16),     nn.MaxPool2d(2, 2),    # 416 -> 208
            _conv_block(16, 32),    nn.MaxPool2d(2, 2),    # 208 -> 104
            _conv_block(32, 64),    nn.MaxPool2d(2, 2),    # 104 -> 52
            _conv_block(64, 128),   nn.MaxPool2d(2, 2),    #  52 -> 26
            _conv_block(128, 256),  nn.MaxPool2d(2, 2),    #  26 -> 13
            _conv_block(256, 512),
            _conv_block(512, 256, k=1),
            _conv_block(256, 512),
        )
        out_per_cell = num_boxes * 5 + num_classes
        self.head = nn.Conv2d(512, out_per_cell, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        out = self.head(feats)                            # (B, B*5+C, S, S)
        out = out.permute(0, 2, 3, 1).contiguous()        # (B, S, S, B*5+C)
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor,
                conf_threshold: float = DETECTOR_CONF_THRESHOLD,
                iou_threshold: float = DETECTOR_NMS_IOU) -> list[dict]:
        """Devuelve lista (uno por imagen del batch) con dict{boxes, scores, classes}.

        Coordenadas en píxeles del input (rango [0, IMG_SIZE]).
        """
        self.eval()
        raw = self.forward(x)
        return decode_predictions(raw, self.num_boxes, self.num_classes,
                                  conf_threshold, iou_threshold)


def decode_predictions(raw: torch.Tensor, num_boxes: int, num_classes: int,
                       conf_threshold: float, iou_threshold: float) -> list[dict]:
    """Convierte el tensor crudo del modelo en bounding boxes filtradas por NMS.

    raw: (B, S, S, B*5 + C) con activaciones lineales.
    """
    B, S, _, _ = raw.shape
    device = raw.device

    # Aplicamos sigmoid a (x, y, conf) y clase, exp a (w, h)
    box_preds = raw[..., : num_boxes * 5].view(B, S, S, num_boxes, 5)
    xy = torch.sigmoid(box_preds[..., :2])              # offsets dentro de la celda
    wh = torch.sigmoid(box_preds[..., 2:4])             # ancho/alto en proporción de imagen
    conf = torch.sigmoid(box_preds[..., 4])             # objectness
    cls_logits = raw[..., num_boxes * 5:]               # (B, S, S, C)
    cls_probs = F.softmax(cls_logits, dim=-1)

    # Coordenadas absolutas en píxeles
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij"
    )
    cx = (xy[..., 0] + grid_x[None, :, :, None]) / S * IMG_SIZE
    cy = (xy[..., 1] + grid_y[None, :, :, None]) / S * IMG_SIZE
    bw = wh[..., 0] * IMG_SIZE
    bh = wh[..., 1] * IMG_SIZE

    boxes_xywh = torch.stack([cx, cy, bw, bh], dim=-1)  # (B, S, S, num_boxes, 4)
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # Score = obj * max(class) — clase del slot box es la mejor del cell (compartida YOLOv1)
    cls_score, cls_id = cls_probs.max(dim=-1)           # (B, S, S)
    scores = conf * cls_score[..., None]                # broadcast a num_boxes
    classes = cls_id[..., None].expand_as(conf)

    results: list[dict] = []
    for b in range(B):
        boxes_b = boxes_xyxy[b].reshape(-1, 4)
        scores_b = scores[b].reshape(-1)
        classes_b = classes[b].reshape(-1)

        mask = scores_b > conf_threshold
        boxes_b, scores_b, classes_b = boxes_b[mask], scores_b[mask], classes_b[mask]
        if boxes_b.numel() == 0:
            results.append({"boxes": boxes_b, "scores": scores_b, "classes": classes_b})
            continue

        keep = class_aware_nms(boxes_b, scores_b, classes_b, iou_threshold)
        results.append({
            "boxes": boxes_b[keep],
            "scores": scores_b[keep],
            "classes": classes_b[keep],
        })
    return results


class DetectorLoss(nn.Module):
    """Pérdida multi-task estilo YOLOv1 — coord + objectness + noobj + clase.

    Coords se aprenden en escala de celda (sx, sy) y proporción imagen (sw, sh).
    targets: (B, S, S, 5 + C) con (obj_mask, x, y, w, h, one-hot clase).
    """

    def __init__(self, num_boxes: int = NUM_BOXES_PER_CELL, num_classes: int = NUM_CLASSES,
                 lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, S, _, _ = pred.shape
        box_preds = pred[..., : self.num_boxes * 5].view(B, S, S, self.num_boxes, 5)
        cls_logits = pred[..., self.num_boxes * 5 :]

        obj_mask = target[..., 0] > 0                    # (B, S, S) bool
        tgt_xywh = target[..., 1:5]                      # (B, S, S, 4)
        tgt_cls = target[..., 5:]                        # one-hot (B, S, S, C)

        # ---- 1) Elegimos el "responsible" box: el que mejor IoU contra el GT en celdas con obj
        with torch.no_grad():
            pred_xy = torch.sigmoid(box_preds[..., :2])
            pred_wh = torch.sigmoid(box_preds[..., 2:4])
            ious = self._compute_iou_per_box(pred_xy, pred_wh, tgt_xywh)  # (B, S, S, num_boxes)
            best_box = ious.argmax(dim=-1)               # (B, S, S)

        # Index del responsible box → mask (B, S, S, num_boxes)
        responsible = F.one_hot(best_box, self.num_boxes).bool()        # (B, S, S, num_boxes)
        responsible = responsible & obj_mask[..., None]

        # ---- 2) Coord loss (solo responsible boxes en celdas con obj)
        pred_xy = torch.sigmoid(box_preds[..., :2])
        pred_wh = torch.sigmoid(box_preds[..., 2:4])
        pred_conf = torch.sigmoid(box_preds[..., 4])

        tgt_xy = tgt_xywh[..., None, :2].expand_as(pred_xy)
        tgt_wh = tgt_xywh[..., None, 2:4].expand_as(pred_wh)

        coord_loss = self.lambda_coord * (
            ((pred_xy - tgt_xy) ** 2).sum(-1)[responsible].sum()
            + ((pred_wh.sqrt() - tgt_wh.sqrt().clamp(min=1e-6)) ** 2).sum(-1)[responsible].sum()
        )

        # ---- 3) Objectness loss: target = IoU del responsible box; noobj boxes → 0
        obj_loss = ((pred_conf - ious.detach()) ** 2)[responsible].sum()
        noobj_loss = self.lambda_noobj * ((pred_conf - 0) ** 2)[~responsible].sum()

        # ---- 4) Class loss (cross-entropy en celdas con obj)
        if obj_mask.any():
            cls_pred = cls_logits[obj_mask]               # (N_obj, C)
            cls_true = tgt_cls[obj_mask].argmax(-1)       # (N_obj,)
            cls_loss = F.cross_entropy(cls_pred, cls_true, reduction="sum")
        else:
            cls_loss = torch.tensor(0.0, device=pred.device)

        total = (coord_loss + obj_loss + noobj_loss + cls_loss) / B
        parts = {
            "coord": (coord_loss / B).item(),
            "obj": (obj_loss / B).item(),
            "noobj": (noobj_loss / B).item(),
            "cls": (cls_loss / B).item(),
        }
        return total, parts

    @staticmethod
    def _compute_iou_per_box(pred_xy: torch.Tensor, pred_wh: torch.Tensor,
                             tgt_xywh: torch.Tensor) -> torch.Tensor:
        """IoU entre cada box predicho y el GT de su celda. Devuelve (B, S, S, num_boxes)."""
        tgt_xy = tgt_xywh[..., None, :2]
        tgt_wh = tgt_xywh[..., None, 2:4]

        p_x1 = pred_xy[..., 0] - pred_wh[..., 0] / 2
        p_y1 = pred_xy[..., 1] - pred_wh[..., 1] / 2
        p_x2 = pred_xy[..., 0] + pred_wh[..., 0] / 2
        p_y2 = pred_xy[..., 1] + pred_wh[..., 1] / 2

        t_x1 = tgt_xy[..., 0] - tgt_wh[..., 0] / 2
        t_y1 = tgt_xy[..., 1] - tgt_wh[..., 1] / 2
        t_x2 = tgt_xy[..., 0] + tgt_wh[..., 0] / 2
        t_y2 = tgt_xy[..., 1] + tgt_wh[..., 1] / 2

        inter = (torch.minimum(p_x2, t_x2) - torch.maximum(p_x1, t_x1)).clamp(0) * \
                (torch.minimum(p_y2, t_y2) - torch.maximum(p_y1, t_y1)).clamp(0)
        area_p = pred_wh[..., 0] * pred_wh[..., 1]
        area_t = tgt_wh[..., 0] * tgt_wh[..., 1]
        union = area_p + area_t - inter
        return inter / union.clamp(min=1e-9)
