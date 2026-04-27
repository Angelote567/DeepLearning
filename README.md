# Volleyball AI — Multimodal Sports Activity Analysis System

Sistema multimodal de análisis de voleibol que integra **5 arquitecturas de redes neuronales**
en una única aplicación con GUI unificada. Proyecto final de la asignatura **Redes Neuronales
y Deep Learning** (Universidad San Jorge, 3º carrera, curso 2025/2026).

> Hola chicas — este README es la entrada del repo. Antes de tocar nada leed la sección "Cómo
> arrancar" y el `docs/architecture.md`.

## Los 5 pilares (todos en PyTorch desde cero salvo el VLM)

| # | Módulo | Arquitectura | Tarea |
|---|--------|--------------|-------|
| ① | MLP    | `src/models/mlp.py`      | Clasificación de acciones (saque / ataque / bloqueo / recepción / colocación) |
| ② | CNN    | `src/models/cnn.py`      | Heatmap de pelota estilo TrackNet (encoder–decoder propio) |
| ③ | Detector | `src/models/detector.py` | YOLOv1-tiny escrito a mano (anchors + NMS propios) — personas, pelota, red |
| ④ | LSTM   | `src/models/lstm.py`     | Predicción de trayectoria de pelota + clasificación táctica |
| ⑤ | VLM    | `src/models/vlm.py`      | Qwen2.5-VL (pre-entrenado) — descripción de escena y Q&A |

**Por qué desde cero:** el enunciado pide "design, train and evaluate" para CNN/Detector/RNN.
La librería Ultralytics (YOLOv8) requiere aprobación previa del profesor — asumimos denegado.
El VLM sí puede ser pre-entrenado (el enunciado dice "incorporate a VLM").

## Estructura del repositorio

```
Proyecto-Grupal/
├── data/                  # voley.yolov8 (descargar con notebooks/01_*.ipynb)
├── weights/               # pesos entrenados (descargados de Drive después de entrenar)
├── src/
│   ├── models/            # las 5 arquitecturas
│   ├── data/              # PyTorch Datasets / DataLoaders
│   ├── train/             # loops de entrenamiento + utils (NMS, IoU, anchors)
│   ├── inference/         # pipeline E2E (vídeo → detec → tracking → VLM)
│   └── gui/               # aplicación PyQt6 (orquesta los 5 módulos)
├── notebooks/             # Colab — descarga dataset + entrenamiento por módulo
├── reports/               # technical_report.pdf + ai_usage_log.md
└── docs/                  # diagramas, architecture.md
```

## Cómo arrancar

### 1. Entorno local (para correr la GUI)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Entrenamiento (Colab, GPU gratis T4)

Abrir en Colab `notebooks/01_dataset_download.ipynb` y ejecutar. Después, en orden:

```
02_train_detector.ipynb        # ③ — primero, según orden recomendado del enunciado
03_train_cnn_ball.ipynb        # ②
04_train_mlp.ipynb             # ①
05_train_lstm.ipynb            # ④
```

Cada notebook guarda los pesos en `/content/drive/MyDrive/volley-ai/weights/`. Descargar
los `.pt` resultantes a la carpeta local `weights/`.

### 3. Lanzar la aplicación

```bash
python -m src.gui.main
```

## División del equipo

- **P1 — Visión / Detección:** preprocesado, CNN, detector, métricas visuales
- **P2 — Secuencias / Tracking:** trayectorias, LSTM, predicción movimiento
- **P3 — Integración / Interfaz / VLM:** GUI, Transformer/VLM, integración, README/demo

## Calendario

| Fase | Fechas | Hito |
|------|--------|------|
| 1 — Definición | 17–27 abril | Sport, dataset, arquitectura, scaffold |
| 2 — Desarrollo | 4–15 mayo  | Modelos entrenados + GUI conectada |
| 3 — Integración | 18–22 mayo | Report PDF + defensa individual |

Entrega: **2026-05-22** (1 semana antes del examen).

## Política de uso de IA

Todo uso de Claude / ChatGPT / Copilot está documentado en
[`reports/ai_usage_log.md`](reports/ai_usage_log.md) según pide el enunciado.

## Dataset

`voley.yolov8` (Roboflow) — 630 imágenes, 3 clases: `pelota`, `persona`, `red`.
Splits 70/20/10. Descargado mediante Roboflow API (ver notebook 01).
