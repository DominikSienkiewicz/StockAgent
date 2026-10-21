"""Wyłuskanie i parsowanie JSON-a z odpowiedzi LLM — wspólne dla adapterów.

Model potrafi owinąć JSON w ```json … ```, nawet w trybie JSON. Failover (NIEZ-44)
przełącza analizę między dostawcami, więc ta sama odpowiedź musi dać ten sam dict
u obu adapterów (testy kontraktowe `test_llm_contract.py`).
"""

from __future__ import annotations

import json
import re
from typing import Any

# Bez otaczających `\s*` — `\s*(.*?)\s*` na wejściu z dużą ilością białych
# znaków bez domknięcia fence'a backtrackuje wielomianowo (ReDoS na output LLM).
# Otaczające białe znaki obcina `strip_code_block` przez `.strip()`.
_CODE_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


def strip_code_block(text: str) -> str:
    """Treść pierwszego bloku ```…``` bez białych znaków na brzegach; tekst bez
    domkniętego bloku wraca niezmieniony."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text


def parse_json_object(text: str) -> dict[str, Any]:
    """`json.loads`, który przyjmuje wyłącznie obiekt na najwyższym poziomie.

    ARCH-01: lista, liczba albo `null` to poprawny JSON, ale nie werdykt — bez
    tej bramki wracały jako „dict” i padały dopiero na `.get` w grafie.

    Raises:
        json.JSONDecodeError: tekst nie jest poprawnym JSON-em (caller może
            jeszcze spróbować wyłuskać obiekt z prozy).
        ValueError: poprawny JSON, ale nie obiekt — tu żaden fallback nie pomoże.
    """
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(
            f"LLM returned a JSON {type(value).__name__}, expected a JSON object."
        )
    return value
