"""Pestaña AI Assistant — text-prompt + frame para pedir descripciones al VLM ⑤."""
from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.models.vlm import SceneVLM


class VLMWorker(QThread):
    answer = pyqtSignal(str)

    def __init__(self, vlm: SceneVLM, image_path: str, prompt: str) -> None:
        super().__init__()
        self.vlm = vlm
        self.image_path = image_path
        self.prompt = prompt

    def run(self) -> None:
        try:
            response = self.vlm.describe(self.image_path, prompt=self.prompt)
            self.answer.emit(response)
        except Exception as e:
            self.answer.emit(f"[error] {e}")


class VLMTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = SceneVLM()
        self.current_image: str | None = None
        self.worker: VLMWorker | None = None

        self.image_label = QLabel("(sin imagen)")
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setStyleSheet("background:#222;color:white;")

        self.prompt = QLineEdit("Describe la escena de voleibol en español. ¿Qué jugada se está ejecutando?")
        self.history = QTextEdit()
        self.history.setReadOnly(True)

        btn_open = QPushButton("📁 Cargar frame")
        btn_open.clicked.connect(self.open_image)
        btn_send = QPushButton("➤ Preguntar al VLM")
        btn_send.clicked.connect(self.ask)

        bar = QHBoxLayout(); bar.addWidget(btn_open); bar.addWidget(self.prompt); bar.addWidget(btn_send)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addLayout(bar)
        layout.addWidget(QLabel("Historial:"))
        layout.addWidget(self.history)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Frame", "", "Imágenes (*.jpg *.png *.jpeg)")
        if not path:
            return
        self.current_image = path
        self.image_label.setPixmap(QPixmap(path).scaled(self.image_label.size(), aspectRatioMode=1))

    def ask(self) -> None:
        if not self.current_image:
            self.history.append("Carga primero una imagen.")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.history.append(f"\n👤 {self.prompt.text()}")
        self.worker = VLMWorker(self.vlm, self.current_image, self.prompt.text())
        self.worker.answer.connect(lambda txt: self.history.append(f"🤖 {txt}"))
        self.worker.start()
