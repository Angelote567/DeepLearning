# AI Usage Log

> Política del enunciado (slide 9): "You must document every instance of AI use and
> explain exactly how it was applied".

Cada entrada lleva fecha, herramienta, tarea, prompt resumido, qué se aceptó / rechazó
y cómo se validó.

| Fecha | Herramienta | Persona | Tarea | Prompt resumido | Output aceptado | Validación |
|-------|-------------|---------|-------|------------------|------------------|------------|
| 2026-04-27 | Claude Opus 4.7 | P3 (Ángel) | Scaffold del repo + 5 módulos en PyTorch desde 0 | "Genera estructura del proyecto Volleyball AI siguiendo el enunciado, sin librería YOLO" | Estructura de carpetas, README, los 5 modelos en PyTorch puro, GUI PyQt6, notebooks Colab | Revisión manual de cada archivo, lectura del código clase a clase, tests con tensores sintéticos planeados |

## Plantilla para añadir entradas nuevas

```
| YYYY-MM-DD | Claude / ChatGPT / Copilot | Pn | tarea concreta | resumen del prompt | qué se aceptó | cómo se validó (test, lectura, ejecución) |
```

## Reglas internas del equipo

1. Nada de IA copia-pega sin entender. Si no sabemos defenderlo en oral, lo
   reescribimos a mano.
2. Cada función generada por IA tiene que tener al menos 1 test manual o uso real.
3. Estas mismas reglas se incluyen en el report PDF como "AI validation policy".
