"""Strażnik twierdzeń w docs i komentarzach, które kod już obalił (DOC-20).

Od migracji 012 trening XGBoost uczy się na `target_return` z widoku
`ml_feature_store`, a `accuracy_score` (bliskość ceny do celu) liczy się dalej,
ale trenować już nie służy. Komentarze w `ports.py` i `agent_graph.py` oraz
`docs/how-it-works.md` długo twierdziły coś przeciwnego — recenzent, który
przechodzi z opisu do kodu treningu, wyłapuje taki rozjazd w minutę.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

CLAIM_SOURCES = ("README.md", "docs/*.md", "src/**/*.py", "main_*.py", "config.toml")

# `accuracy_score`, a w ciągu kolejnych ~160 znaków (także przez koniec linii
# komentarza) „napędza trening” albo „feeds XGBoost training”. Okno zamiast jednej
# linii, bo komentarz w agent_graph.py zawijał to twierdzenie na dwie linie.
ACCURACY_FEEDS_TRAINING = re.compile(
    r"accuracy_score[\s\S]{0,160}?"
    r"(?:(?:napędza|zasila|karmi)\s+trening|feeds?\s+(?:the\s+)?(?:XGBoost\s+)?training)",
    re.IGNORECASE,
)


def _claim_files() -> list[Path]:
    return sorted(
        path for pattern in CLAIM_SOURCES for path in ROOT.glob(pattern) if path.is_file()
    )


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def test_docs_do_not_claim_accuracy_score_feeds_training() -> None:
    claims = [
        f"{path.relative_to(ROOT)}:{_line_of(text, found.start())}"
        for path in _claim_files()
        for text in [path.read_text(encoding="utf-8")]
        for found in ACCURACY_FEEDS_TRAINING.finditer(text)
    ]

    assert claims == [], (
        "accuracy_score nie napędza treningu — uczy się na `target_return` z "
        f"ml_feature_store: {claims}"
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "`accuracy_score` (how close the price landed — feeds XGBoost training)",
        "# Domena liczy accuracy_score (0.0-1.0) — cena do celu,\n# to napędza trening XGBoost.",
        "`accuracy_score` (bliskość ceny do\n celu) napędza trening;",
    ],
)
def test_accuracy_claim_pattern_matches_both_languages_and_wrapped_lines(sentence: str) -> None:
    assert ACCURACY_FEEDS_TRAINING.search(sentence)


def test_accuracy_claim_pattern_ignores_the_corrected_wording() -> None:
    corrected = (
        "`accuracy_score` feeds the weekly recap; training uses `target_return` "
        "from `ml_feature_store`"
    )

    assert not ACCURACY_FEEDS_TRAINING.search(corrected)
