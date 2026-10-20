"""Wyłuskanie JSON-a z bloku kodu markdown — wspólne dla adapterów LLM.

Model potrafi owinąć JSON w ```json … ```, nawet w trybie JSON. Failover (NIEZ-44)
przełącza analizę między dostawcami, więc ta sama odpowiedź musi dać ten sam dict
u obu adapterów (testy kontraktowe `test_llm_contract.py`).
"""

from __future__ import annotations

import re

# Bez otaczających `\s*` — `\s*(.*?)\s*` na wejściu z dużą ilością białych
# znaków bez domknięcia fence'a backtrackuje wielomianowo (ReDoS na output LLM).
# Otaczające białe znaki obcina `strip_code_block` przez `.strip()`.
_CODE_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


def strip_code_block(text: str) -> str:
    """Treść pierwszego bloku ```…``` bez białych znaków na brzegach; tekst bez
    domkniętego bloku wraca niezmieniony."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text
