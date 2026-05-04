"""Pestaña Live Analytics — visualización en tiempo real de los 5 módulos."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ClickableLabel(QLabel):
    """QLabel que emite clicked(QPoint) en mousePressEvent — para marcar esquinas."""

    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(ev.position().toPoint())
        super().mousePressEvent(ev)

from src.inference.court import draw_court_template
from src.inference.pipeline import VolleyPipeline


class LiveTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = VolleyPipeline()
        self.cap: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.video_label = ClickableLabel("Sube un vídeo para empezar")
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background: #111; color: white;")
        self.video_label.clicked.connect(self._on_video_click)

        # Estado para marcar las 4 esquinas: lista de QPoint mientras está activo el modo
        self._picking_corners: bool = False
        self._corner_points: list[QPoint] = []
        self._last_frame: np.ndarray | None = None       # frame original para overlay marcado

        self.court_label = QLabel()
        self.court_label.setMinimumSize(360, 180)
        self.court_label.setStyleSheet("background: #f0f0f0;")

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(200)

        btn_video = QPushButton("📁 Abrir vídeo")
        btn_video.clicked.connect(self.open_video)
        btn_image = QPushButton("🖼️ Abrir imagen")
        btn_image.clicked.connect(self.open_image)
        btn_camera = QPushButton("📷 Cámara")
        btn_camera.clicked.connect(self.start_camera)
        btn_corners = QPushButton("⊟ Marcar esquinas cancha")
        btn_corners.clicked.connect(self.pick_corners)

        controls = QHBoxLayout()
        controls.addWidget(btn_video)
        controls.addWidget(btn_image)
        controls.addWidget(btn_camera)
        controls.addWidget(btn_corners)
        controls.addStretch()

        right = QVBoxLayout()
        right.addWidget(QLabel("Cancha 2D"))
        right.addWidget(self.court_label)
        right.addWidget(QLabel("Análisis"))
        right.addWidget(self.info)

        center = QHBoxLayout()
        center.addWidget(self.video_label, stretch=2)
        center.addLayout(right, stretch=1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(center)

    # ---- input handlers
    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.timer.start(33)                                # ~30 fps

    def start_camera(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.timer.start(33)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Imagen", "", "Imágenes (*.jpg *.jpeg *.png)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            return
        self.timer.stop()
        self._last_frame = frame.copy()
        result = self.pipeline.process_frame(frame)
        self._render(result)

    def pick_corners(self) -> None:
        """Activa el modo "marcar esquinas": el usuario hace 4 clicks sobre el frame
        en orden top-left, top-right, bottom-right, bottom-left."""
        if self._last_frame is None:
            self.info.append("Carga primero una imagen o vídeo.")
            return
        self._picking_corners = True
        self._corner_points = []
        self.timer.stop()                                 # pausa el vídeo mientras marcamos
        self.info.append("[esquinas] Click en TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT.")
        self._refresh_corner_overlay()

    def _on_video_click(self, point: QPoint) -> None:
        """Recibe clicks sobre `video_label`. Si estamos en modo "marcar esquinas",
        acumula el punto convertido a coordenadas de imagen original."""
        if not self._picking_corners or self._last_frame is None:
            return
        # Convertir la posición del click (en coords del label) a coords de la imagen original
        label_w, label_h = self.video_label.width(), self.video_label.height()
        img_h, img_w = self._last_frame.shape[:2]
        scale = min(label_w / img_w, label_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        offset_x = (label_w - disp_w) // 2
        offset_y = (label_h - disp_h) // 2
        click_x = (point.x() - offset_x) / scale
        click_y = (point.y() - offset_y) / scale
        if not (0 <= click_x < img_w and 0 <= click_y < img_h):
            return
        self._corner_points.append(QPoint(int(click_x), int(click_y)))
        self.info.append(f"[esquina {len(self._corner_points)}/4] ({int(click_x)}, {int(click_y)})")
        self._refresh_corner_overlay()

        if len(self._corner_points) == 4:
            corners = np.array([[p.x(), p.y()] for p in self._corner_points], dtype=np.float32)
            self.pipeline.set_court_corners(corners)
            self._picking_corners = False
            self.info.append("[ok] Homografía calculada — la cancha 2D ya está activa.")
            # Re-renderizar el frame actual con el pipeline completo
            result = self.pipeline.process_frame(self._last_frame)
            self._render(result)

    def _refresh_corner_overlay(self) -> None:
        """Pinta los puntos ya seleccionados sobre el frame actual mientras se marcan."""
        if self._last_frame is None:
            return
        overlay = self._last_frame.copy()
        for i, pt in enumerate(self._corner_points):
            cv2.circle(overlay, (pt.x(), pt.y()), 8, (0, 255, 255), -1)
            cv2.putText(overlay, f"{i+1}", (pt.x() + 12, pt.y() + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(self._corner_points) > 1:
            pts = np.array([[p.x(), p.y()] for p in self._corner_points], dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=(len(self._corner_points) == 4),
                          color=(0, 255, 255), thickness=2)
        self.video_label.setPixmap(self._to_pixmap(overlay, self.video_label.size()))

    # ---- frame loop
    def next_frame(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            return
        self._last_frame = frame.copy()
        result = self.pipeline.process_frame(frame)
        self._render(result)

    # ---- rendering
    def _render(self, result: dict) -> None:
        annotated = self._annotate(result)
        self.video_label.setPixmap(self._to_pixmap(annotated, self.video_label.size()))

        court = self._render_court(result)
        self.court_label.setPixmap(self._to_pixmap(court, self.court_label.size()))

        info_lines = [
            f"Detecciones: {len(result['detections'])}",
        ]
        if result["tactic"]:
            info_lines.append(f"Táctica predicha: {result['tactic']}")
        if result["ball_court"]:
            x, y = result["ball_court"]
            info_lines.append(f"Pelota cancha: ({x:.0f}, {y:.0f})")
        self.info.setPlainText("\n".join(info_lines))

    def _annotate(self, result: dict) -> np.ndarray:
        frame = result["frame"].copy()
        colors = {"ball": (0, 255, 255), "player": (0, 255, 0), "referee": (255, 0, 255)}
        for det in result["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            color = colors.get(det["class"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['score']:.2f}"
            if "zone" in det:
                label += f" [{det['zone']}]"
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _render_court(self, result: dict) -> np.ndarray:
        court = draw_court_template()
        if result["ball_court"]:
            x, y = result["ball_court"]
            cv2.circle(court, (int(x), int(y)), 6, (0, 165, 255), -1)
        if result["predicted_path"]:
            for px, py in result["predicted_path"]:
                cv2.circle(court, (int(px * court.shape[1]), int(py * court.shape[0])),
                            3, (255, 0, 255), -1)
        return court

    @staticmethod
    def _to_pixmap(frame_bgr: np.ndarray, size) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
