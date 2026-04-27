# Technical Report — outline

> Esto es el guion del PDF que entregamos. Lo rellenamos a partir del 18 de mayo cuando
> tengamos modelos entrenados y métricas reales. Cada sección numerada se corresponde con
> los requisitos del enunciado (slide 11).

## 1. Project Overview

- Deporte elegido: **voleibol** (descartados fútbol 11 y baloncesto por enunciado).
- Dataset: `voley.yolov8` (Roboflow, 630 imágenes, 3 clases — pelota, persona, red).
- Objetivo del sistema: análisis multimodal en tiempo real desde una GUI unificada.

## 2. Module Descriptions (una sección por arquitectura)

### 2.1 ① MLP — Clasificador de acción
- Arquitectura: 3 capas FC con BN + Dropout + ReLU. Diagrama → `docs/mlp.svg`.
- Loss: CrossEntropy. Optimizer: Adam(lr=1e-3). Epochs: 40.
- Métrica: accuracy + matriz de confusión sobre 5 clases.

### 2.2 ② CNN — Heatmap de pelota
- Arquitectura: encoder-decoder propio con skip connections.
- Loss: focal-weighted MSE.  Optimizer: Adam(lr=1e-3, wd=5e-4). Epochs: 60.
- Métrica: distancia píxel pico-GT, % frames donde la pelota se localiza correctamente.

### 2.3 ③ Object Detector — VolleyDetector
- Arquitectura propia (NO Ultralytics) — single-shot grid detector.
- Loss multi-task estilo YOLOv1 (coord + obj + noobj + cls).
- NMS y class-aware NMS escritos a mano.
- Métricas: mAP@0.5 y mAP@0.5:0.95 calculadas con scripts propios sobre el split test.

### 2.4 ④ LSTM — Trayectoria + táctica
- Encoder-decoder LSTM auto-regresivo + cabeza de clasificación táctica.
- Loss combinada: MSE coords + CE táctica (α=1.0, β=0.5).
- Métricas: ADE/FDE (regresión) y accuracy táctica.

### 2.5 ⑤ VLM — Qwen2.5-VL-3B-Instruct
- Pre-entrenado (permitido por enunciado).
- Prompt engineering con contexto de detecciones del detector + táctica del LSTM.
- Evaluación cualitativa: 10 clips comentados manualmente.

## 3. Design Decisions & Justification

- **Por qué voleibol y no fútbol-11/baloncesto:** restricción del enunciado.
- **Por qué dataset Roboflow:** disponible públicamente, anotado en formato YOLO,
  3 clases relevantes, ya equilibrado (70/20/10 splits).
- **Por qué red propia y no Ultralytics:** la asignatura exige "design, train and
  evaluate" — uso de YOLO librería requiere aprobación previa del profesor.
- **Por qué Qwen2.5-VL:** cabe en T4 free, calidad descriptiva alta en español,
  Apache-2.0. Alternativas valoradas: LLaVA, MiniCPM-V, Gemma-3.
- **Por qué PyQt6 y no Streamlit:** "feel like a real application, not a collection
  of notebooks" según enunciado. PyQt6 da mejor UX y opta a "best tool +10%".

### Repos consultados como inspiración (no copiados)
- `asigatchov/vball-net` (TrackNetV4) — inspiración para la arquitectura de heatmap.
- `volleyIEEE/VolleyStats` (PathFinder) — inspiración para LSTM táctica.
- `TheSmike/VolleyCourt-mapping` — patrón de homografía cancha 2D.
- `masouduut94/volleyball_analytics` — patrón end-to-end y dashboard.
- `shukkkur/VolleyVision`, `jadidimohammad/volleyball-tracking`, `openvolley/ovml`,
  `mbird1258/Body-World-Eye-Mapping`, `asigatchov/fast-volleyball-tracking-inference`
  — exploración inicial del estado del arte.

### Uso de IA para programar
Ver [`ai_usage_log.md`](ai_usage_log.md) para el detalle. Resumen:
- Claude Opus 4.7 — pair programming: scaffold del repo, primeras versiones de cada
  módulo (loss YOLO, NMS, encoder-decoder), revisión de código.
- Validación: cada función pasa por al menos un test manual con tensores sintéticos
  antes de aceptarla. La defensa oral demuestra entendimiento real.

## 4. Conclusions & Lessons Learned

- Lo que funcionó: arrancar primero detector + GUI nos permitió iterar en paralelo.
- Lo que costó: equilibrar la pérdida del detector (coord vs noobj).
- Qué haríamos diferente: anotar trayectorias reales antes en lugar de usar sintéticas.

### Contribución individual

| Persona | Módulos |
|---------|---------|
| P1 — Vision/Detection | dataset preprocesado, CNN ②, detector ③, métricas |
| P2 — Sequences/Tracking | sequence dataset, LSTM ④, evaluación temporal |
| P3 — Integration/VLM | GUI PyQt6, VLM ⑤, integración E2E, README, demo |
