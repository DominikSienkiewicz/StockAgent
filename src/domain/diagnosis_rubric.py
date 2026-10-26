"""EXP-10 — rubryka sędziego diagnoz błędnych predykcji (LLM-as-judge na próbce).

Diagnoza LLM („lessons learned”) wraca do kolejnych promptów przez reflection i analogi
RAG, więc ładnie brzmiący szum zatruwałby predykcje. Sędzia ocenia próbkę diagnoz
trzema kryteriami tak/nie; wynik to liczba spełnionych (0–3), a diagnoza poniżej
`MIN_PASSING_RUBRIC_SCORE` nie trafia do lessons ani RAG.

Czysta logika (stdlib + domena): parse odpowiedzi sędziego i reguła progu.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Kryteria rubryki — klucze JSON odpowiedzi sędziego, w kolejności promptu.
RUBRIC_CRITERIA: tuple[str, ...] = ("factor_in_evidence", "falsifiable", "specific")
# Ile kryteriów musi spełnić diagnoza, żeby zostać lekcją. Próg do kalibracji przez
# właściciela na 20 jego ocenach (EXP-10) — do tego czasu 2 z 3.
MIN_PASSING_RUBRIC_SCORE = 2


@dataclass(frozen=True)
class RubricVerdict:
    """Ocena jednej diagnozy: (1) czynnik obecny w newsach lub cenach, (2) teza
    falsyfikowalna, (3) nie ogólnik."""

    factor_in_evidence: bool
    falsifiable: bool
    specific: bool

    @property
    def score(self) -> int:
        return sum((self.factor_in_evidence, self.falsifiable, self.specific))

    @property
    def passes(self) -> bool:
        return self.score >= MIN_PASSING_RUBRIC_SCORE


def parse_rubric_verdict(raw: str) -> RubricVerdict | None:
    """Odpowiedź sędziego → ocena. Pierwszy obiekt JSON w tekście z trzema
    kryteriami typu bool; brak JSON-a albo kryterium bez wartości tak/nie → None
    (ocena niepewna nie odrzuca diagnozy, wiersz wraca do kolejnej próbki)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        # Wycinek zaczyna się od „{” i kończy „}”, więc poprawny JSON to zawsze obiekt.
        payload: dict[str, object] = json.loads(raw[start : end + 1])
    except ValueError:
        return None
    values = [payload.get(criterion) for criterion in RUBRIC_CRITERIA]
    if not all(isinstance(value, bool) for value in values):
        return None
    factor, falsifiable, specific = (bool(value) for value in values)
    return RubricVerdict(factor, falsifiable, specific)
