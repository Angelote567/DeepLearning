"""Pestaña Reports — métricas de los modelos entrenados (loss curves, mAP, accuracy)."""
from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ReportsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Reports — pendiente integrar matplotlib en QWidget para mostrar:\n"
            "  • Loss curves (train/val) por módulo\n"
            "  • Confusion matrix MLP / LSTM táctica\n"
            "  • mAP@0.5 del detector\n"
            "  • Tabla resumen final del experimento"
        ))
