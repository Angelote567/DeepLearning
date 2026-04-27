"""Pestaña Training — el enunciado pide pipeline de entrenamiento DENTRO de la app.

Permite seleccionar módulo (detector/cnn/mlp/lstm) + hyperparams (epochs, lr, batch)
y lanza el entrenamiento en un QThread para no congelar la GUI.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainingWorker(QThread):
    """Hilo que llama a la función `train()` del módulo seleccionado."""
    log = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(self, module: str, epochs: int, lr: float, batch: int) -> None:
        super().__init__()
        self.module = module
        self.epochs = epochs
        self.lr = lr
        self.batch = batch

    def run(self) -> None:
        try:
            self.log.emit(f"Iniciando entrenamiento: {self.module}")
            if self.module == "detector":
                from src.train.train_detector import train
                out = train(epochs=self.epochs, batch_size=self.batch, lr=self.lr,
                            log_cb=lambda e, m: self.log.emit(f"[{e}] {m}"))
            elif self.module == "cnn":
                self.log.emit("CNN heatmap pendiente — pega aquí la llamada cuando exista train_cnn.py")
                return
            elif self.module == "mlp":
                self.log.emit("MLP pendiente — train_mlp.py se completa con datos anotados")
                return
            elif self.module == "lstm":
                self.log.emit("LSTM pendiente — train_lstm.py listo cuando haya trayectorias")
                return
            else:
                self.log.emit(f"Módulo desconocido: {self.module}")
                return
            self.done.emit(f"OK — pesos en {out}")
        except Exception as e:
            self.log.emit(f"[error] {e}")


class TrainTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.module_select = QComboBox()
        self.module_select.addItems(["detector", "cnn", "mlp", "lstm"])

        self.epochs = QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(80)
        self.batch = QSpinBox(); self.batch.setRange(1, 256); self.batch.setValue(16)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(5); self.lr.setRange(1e-5, 1.0); self.lr.setValue(1e-3)

        form = QFormLayout()
        form.addRow("Módulo:", self.module_select)
        form.addRow("Epochs:", self.epochs)
        form.addRow("Batch size:", self.batch)
        form.addRow("Learning rate:", self.lr)

        self.btn_start = QPushButton("▶ Lanzar entrenamiento")
        self.btn_start.clicked.connect(self.start)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.btn_start)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_area)

        self.worker: TrainingWorker | None = None

    def start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.log_area.append("Ya hay un entrenamiento en curso.")
            return
        self.worker = TrainingWorker(
            module=self.module_select.currentText(),
            epochs=self.epochs.value(),
            lr=self.lr.value(),
            batch=self.batch.value(),
        )
        self.worker.log.connect(self.log_area.append)
        self.worker.done.connect(self.log_area.append)
        self.worker.start()
