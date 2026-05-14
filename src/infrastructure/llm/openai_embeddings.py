"""Adapter dla OpenAI Embeddings API.

`text-embedding-3-small` zwraca wektory 1536-wymiarowe — dokładnie tyle,
ile ma kolumna `embedding VECTOR(1536)` w `prediction_logs` (pgvector).
"""

from __future__ import annotations

from openai import OpenAI

from src.application.ports import EmbeddingPort

DEFAULT_MODEL = "text-embedding-3-small"  # 1536 wymiarów, tani, szybki


class OpenAIEmbeddingAdapter(EmbeddingPort):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        # Pusty tekst → zerowy wektor (AV nie zwrócił newsów dla tego symbolu).
        if not text or not text.strip():
            return []
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)
