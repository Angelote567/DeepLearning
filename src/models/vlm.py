"""⑤ VLM — Vision-Language Model para descripción de escena y Q&A.

A diferencia de los otros 4 módulos, el enunciado dice "incorporate a VLM" — está
permitido usar un modelo pre-entrenado. Elegimos **Qwen2.5-VL-3B-Instruct** porque:
* Cabe en una GPU T4 free de Colab (~6 GB con `torch_dtype=torch.float16`).
* Acepta imágenes + texto, devuelve respuesta libre.
* Apache-2.0 — sin problemas de licencia para entrega.

Alternativas valoradas (mencionar en el report PDF):
* LLaVA-1.5 — más antiguo, peor en Q&A.
* Gemma-3-4B — buen rendimiento pero requiere acceso de HuggingFace.
* MiniCPM-V — más liviano pero peor calidad descriptiva.

Uso: el wrapper acepta un frame anotado (con bbox dibujadas) + un prompt en español
y devuelve descripción de la escena o respuesta a una pregunta.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image


class SceneVLM:
    """Wrapper minimal sobre Qwen2.5-VL. Carga perezosa para que la GUI arranque sin GPU."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def describe(self, image: Image.Image | str | Path,
                 prompt: str = "Describe la escena de voleibol en español. ¿Qué jugada se está ejecutando?",
                 max_new_tokens: int = 256) -> str:
        self._load()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def ask(self, image: Image.Image | str | Path, question: str) -> str:
        return self.describe(image, prompt=question)


def build_context_prompt(detections: list[dict], tactic: str | None = None) -> str:
    """Construye un prompt con el contexto de detecciones para enriquecer la respuesta del VLM."""
    lines = ["Contexto del análisis automático:"]
    n_players = sum(1 for d in detections if d.get("class") == "persona")
    has_ball = any(d.get("class") == "pelota" for d in detections)
    lines.append(f"- {n_players} jugadores detectados.")
    lines.append(f"- Pelota visible: {'sí' if has_ball else 'no'}.")
    if tactic:
        lines.append(f"- Táctica predicha por el LSTM: {tactic}.")
    lines.append("\nDescribe brevemente lo que está pasando en la imagen (1-2 frases).")
    return "\n".join(lines)
