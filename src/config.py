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
DETECTOR_CONF_THRESHOLD = 0.05   # umbral bajo — modelo entrenado desde 0 con 270 imgs
                                  # predice con confianza modesta (justificado en report)
DETECTOR_NMS_IOU = 0.45

# CNN ② — heatmap pelota TrackNet-style
HEATMAP_SIZE = 288
HEATMAP_INPUT_FRAMES = 3  # se concatenan 3 frames consecutivos en canal

# LSTM ④ — predicción trayectoria
LSTM_INPUT_LEN = 30
LSTM_OUTPUT_LEN = 10

# MLP ① — clasificación de zona de cancha del jugador
# Etiquetas auto-generadas a partir de la posición del bbox (cy normalizado).
# Ventaja: cero etiquetado manual. Demuestra MLP + extracción de features.
ZONE_CLASSES = ["lejos", "medio", "cerca"]   # respecto a la cámara
MLP_FEATURES = 14   # 6 geométricos + 8 estadísticos (color RGB mean/std + brillo + contraste)

# Cancha FIVB en metros
COURT_W_M = 18.0
COURT_H_M = 9.0
