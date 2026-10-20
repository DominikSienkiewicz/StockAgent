"""FIN-02: liczenie płatnych wywołań per REALNY request do dostawcy LLM.

Rada doradcza robi w jednym przebiegu N wywołań person, chairmana i (w trybie
debaty) rundę rewizji. Węzeł grafu widzi z tego jedno `council_port.analyze`,
więc licznik w węźle zaniżał radę 8–15 razy. Dekorator portu LLM liczy każde
wywołanie u źródła — zanim oddamy je dostawcy, więc request, który potem padnie
albo nie zdąży przed limitem czasu rady, też jest policzony (poszedł i kosztuje).
"""

from __future__ import annotations

from typing import Any

from src.application.cycle_telemetry import CycleTelemetry
from src.application.ports import LLMPort


class MeteredLLMPort(LLMPort):
    """Port LLM, który rejestruje każde wywołanie w telemetrii cyklu pod
    kluczem cennika `source` (np. `council_llm`) i deleguje do `inner`."""

    def __init__(self, inner: LLMPort, telemetry: CycleTelemetry, source: str) -> None:
        self._inner = inner
        self._telemetry = telemetry
        self._source = source

    def analyze(self, prompt: str) -> dict[str, Any]:
        self._telemetry.record_paid_call(self._source)
        return self._inner.analyze(prompt)

    def analyze_mistake(self, prompt: str) -> str:
        self._telemetry.record_paid_call(self._source)
        return self._inner.analyze_mistake(prompt)
