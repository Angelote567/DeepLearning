"""Configuración global compartida por todos los módulos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
DATASET_DIR = DATA_DIR / "volleyball-2"

# Clases del dataset Roboflow `qc/volleyball-hwxp2` v2 (330 imágenes Tokio 2020).
# El orden coincide con el id de la clase en los .txt YOLO. Si cambiamos de dataset,
# hay que verificar este orden contra el `data.yaml` que descarga Roboflow.
CLASSES = ["ball", "player", "referee"]
NUM_CLASSES = len(CLASSES)

# Detector ③ — YOLOv1-tiny
IMG_SIZE = 416
GRID_SIZE = 13
NUM_BOXES_PER_CELL = 2
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_NMS_IOU = 0.45

# CNN ② — heatmap pelota TrackNet-style
HEATMAP_SIZE = 288
HEATMAP_INPUT_FRAMES = 3  # se concatenan 3 frames consecutivos en canal

# LSTM ④ — predicción trayectoria
LSTM_INPUT_LEN = 30
LSTM_OUTPUT_LEN = 10

# MLP ① — clasificación de acción
ACTION_CLASSES = ["saque", "ataque", "bloqueo", "recepcion", "colocacion"]

# Cancha FIVB en metros
COURT_W_M = 18.0
COURT_H_M = 9.0
