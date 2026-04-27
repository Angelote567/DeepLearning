"""GUI unificada PyQt6 — orquesta los 5 módulos AI según el enunciado.

Pestañas:
    Live Analytics    → carga vídeo/imagen/cámara + visualiza detecciones, trayectoria, cancha 2D
    Training          → lanza entrenamiento de cualquier módulo con hyperparams configurables
    AI Assistant (VLM) → text-prompt + frame para Q&A
    Reports           → resultados, métricas, export

Lanzar con:  python -m src.gui.main
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget

from src.gui.tabs.live import LiveTab
from src.gui.tabs.train import TrainTab
from src.gui.tabs.vlm import VLMTab
from src.gui.tabs.reports import ReportsTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Volleyball AI — USJ NN/DL Group Project")
        self.resize(1400, 900)

        tabs = QTabWidget()
        tabs.addTab(LiveTab(), "🏐 Live Analytics")
        tabs.addTab(TrainTab(), "🎯 Training")
        tabs.addTab(VLMTab(), "💬 AI Assistant")
        tabs.addTab(ReportsTab(), "📊 Reports")
        self.setCentralWidget(tabs)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
