"""IoU, NMS y conversión de coordenadas — todo escrito a mano para defenderlo en oral."""
import torch


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU entre dos conjuntos de cajas en formato (x1, y1, x2, y2). Devuelve matriz (N, M)."""
    a_area = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = a_area[:, None] + b_area[None, :] - inter
    return inter / union.clamp(min=1e-9)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    """Centro+wh -> esquinas. Acepta tensor (..., 4)."""
    out = xywh.clone()
    out[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    out[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    out[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    out[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return out


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Non-Max Suppression vectorizada — implementación propia, no torchvision.ops.nms.

    Devuelve los índices de las cajas que sobreviven, ordenados por score descendiente.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious < iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """NMS aplicado independiente por clase."""
    keep_all: list[torch.Tensor] = []
    for c in classes.unique():
        mask = classes == c
        idx = torch.nonzero(mask).flatten()
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.append(idx[kept])
    if not keep_all:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.cat(keep_all)
