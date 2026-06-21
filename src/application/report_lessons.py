"""Panel 'lessons learned' (T4 — Mistake-Diagnosis Audit Trail).

Czysta logika aplikacyjna (zero I/O, zero portów). Destyluje z zamkniętych
predykcji wyłącznie BŁĘDY, którym towarzyszy diagnoza po fakcie (insight),
i wykrywa powracający motyw — najczęstsze nietrywialne słowo w diagnozach.
Renderer raportu konsumuje gotowy LessonsReport bez dalszego liczenia."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.application.report_models import ResolvedPrediction

# Polskie stopwordy (oraz krótkie spójniki) — nie mogą wygrać jako motyw.
# Niezależnie od listy odrzucamy też słowa krótsze niż _MIN_KEYWORD_LEN.
_STOPWORDS: frozenset[str] = frozenset(
    {"i", "w", "na", "się", "że", "nie", "to", "z", "o", "a", "do", "po"}
)

# Minimalna długość słowa kandydującego na motyw (jak w specyfikacji: min 4).
_MIN_KEYWORD_LEN = 4

# Tokenizer słów: sekwencje znaków alfanumerycznych w danym locale (Unicode).
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class LessonItem:
    """Pojedyncza 'lekcja' — błędna predykcja z diagnozą i faktycznym ruchem."""

    symbol: str
    predicted_trend: str
    insight: str
    actual_change_pct: Decimal | None


@dataclass(frozen=True)
class LessonsReport:
    """Skondensowany panel diagnoz: lekcje (z limitem), łączna liczba błędów
    i powracający motyw (lub None, gdy brak wyraźnego zwycięzcy)."""

    items: tuple[LessonItem, ...]
    total_mistakes: int
    recurring_theme: str | None


def _is_mistake_with_insight(prediction: ResolvedPrediction) -> bool:
    """Kwalifikuje predykcję jako 'lekcję': błąd + niepusta diagnoza."""
    if prediction.is_correct:
        return False
    insight = prediction.insight
    return insight is not None and bool(insight.strip())


def _detect_recurring_theme(insights: Sequence[str]) -> str | None:
    """Najczęstsze nietrywialne słowo w diagnozach (case-insensitive).

    Pomija stopwordy i słowa krótsze niż _MIN_KEYWORD_LEN. Zwraca None, gdy
    żadne słowo nie powtarza się (brak wyraźnego zwycięzcy: częstość < 2)."""
    counts: Counter[str] = Counter()
    for insight in insights:
        for word in _WORD_RE.findall(insight.lower()):
            # Odrzucamy zbyt krótkie, stopwordy oraz tokeny czysto cyfrowe
            # (np. rok "2024") — szum numeryczny nie jest powracającym motywem.
            if (
                len(word) < _MIN_KEYWORD_LEN
                or word in _STOPWORDS
                or word.isdigit()
            ):
                continue
            counts[word] += 1

    if not counts:
        return None

    word, frequency = counts.most_common(1)[0]
    # Brak powtórzenia → brak motywu (pojedyncze wystąpienie to nie trend).
    if frequency < 2:
        return None
    return word


def build_lessons(
    resolved: Sequence[ResolvedPrediction], limit: int = 8
) -> LessonsReport:
    """Buduje panel 'lessons learned' z zamkniętych predykcji.

    Zachowuje wyłącznie błędy z niepustą diagnozą (insight), w kolejności
    wejścia (most-recent-first). `total_mistakes` liczy wszystkie kwalifikujące
    się błędy PRZED przycięciem do `limit`. `recurring_theme` to najczęstszy
    nietrywialny keyword w diagnozach (lub None)."""
    mistakes = [p for p in resolved if _is_mistake_with_insight(p)]

    items = tuple(
        LessonItem(
            symbol=p.symbol,
            predicted_trend=p.predicted_trend,
            # insight nie jest None po filtrze _is_mistake_with_insight.
            insight=p.insight,  # type: ignore[arg-type]
            actual_change_pct=p.actual_change_pct,
        )
        for p in mistakes[:limit]
    )

    recurring_theme = _detect_recurring_theme(
        [p.insight for p in mistakes if p.insight is not None]
    )

    return LessonsReport(
        items=items,
        total_mistakes=len(mistakes),
        recurring_theme=recurring_theme,
    )
