"""WIA-57 — taksonomia przyczyn błędnych predykcji.

Panel lessons wybierał „powracający motyw” jako najczęstsze słowo diagnoz, więc
dla trzech realistycznych diagnoz zwracał „zignorowałem” — słowo, nie przyczynę.
Diagnoza LLM podaje teraz kategorię z zamkniętego zbioru obok tekstu, a raport
liczy histogram kategorii („3 z 5 błędów: makro”).

Port `LLMPort.analyze_mistake` dalej zwraca tekst. `parse_mistake_diagnosis`
wyciąga z niego JSON `{category, insight}` i ma fallback: odpowiedź bez
poprawnego JSON-a to OTHER z całym tekstem jako diagnozą, więc stary prompt,
atrapa lokalna i model, który zignoruje format, niczego nie psują.

Czysta logika (stdlib + domena).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class MistakeCategory(StrEnum):
    """Przyczyna błędu predykcji — wartości kolumny `correction_category` (046)."""

    MACRO = "MACRO"
    EARNINGS = "EARNINGS"
    STALE_NEWS = "STALE_NEWS"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    SENTIMENT_TRAP = "SENTIMENT_TRAP"
    NOISE = "NOISE"
    OTHER = "OTHER"


# Etykiety do maila (rzeczownik po „błędów:”).
MISTAKE_CATEGORY_LABELS_PL: dict[MistakeCategory, str] = {
    MistakeCategory.MACRO: "makro",
    MistakeCategory.EARNINGS: "wyniki spółki",
    MistakeCategory.STALE_NEWS: "nieaktualne newsy",
    MistakeCategory.SECTOR_ROTATION: "rotacja sektorowa",
    MistakeCategory.SENTIMENT_TRAP: "pułapka sentymentu",
    MistakeCategory.NOISE: "szum rynkowy",
    MistakeCategory.OTHER: "inne",
}


@dataclass(frozen=True)
class MistakeDiagnosis:
    """Diagnoza jednej błędnej predykcji: kategoria i tekst do `correction_insights`."""

    category: MistakeCategory
    insight: str


def parse_mistake_category(value: object) -> MistakeCategory | None:
    """Kategoria z danych nieufnych (JSON LLM, kolumna bazy) bez rozróżniania
    wielkości liter; brak albo wartość spoza zbioru → None."""
    if not isinstance(value, str):
        return None
    key = value.strip().upper()
    return MistakeCategory(key) if key in MistakeCategory.__members__ else None


def _json_object(text: str) -> dict[str, object] | None:
    """Pierwszy obiekt JSON w tekście (także w bloku ```json) albo None."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def parse_mistake_diagnosis(raw: str) -> MistakeDiagnosis:
    """Odpowiedź `analyze_mistake` → kategoria + diagnoza.

    JSON z kategorią spoza zbioru → OTHER z jego `insight`. JSON bez niepustego
    `insight` albo tekst bez JSON-a → cała odpowiedź (po strip) jako diagnoza,
    żeby nie zgubić treści, za którą zapłaciliśmy."""
    text = raw.strip()
    payload = _json_object(text)
    if payload is None:
        return MistakeDiagnosis(MistakeCategory.OTHER, text)
    category = parse_mistake_category(payload.get("category")) or MistakeCategory.OTHER
    insight = payload.get("insight")
    if not isinstance(insight, str) or not insight.strip():
        return MistakeDiagnosis(category, text)
    return MistakeDiagnosis(category, insight.strip())


def count_categories(
    categories: Iterable[MistakeCategory | None],
) -> dict[MistakeCategory, int]:
    """Histogram kategorii malejąco po liczności (remis: kolejność taksonomii);
    błędy bez kategorii (wiersze sprzed migracji 046) pomijamy."""
    counts = Counter(category for category in categories if category is not None)
    order = list(MistakeCategory)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], order.index(item[0]))))
