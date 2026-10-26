"""MAIL-38: persony rady w publicznych wyjściach tylko jako archetypy.

Decyzja właściciela (2026-10-23, opcja c): nazwiska person zostają w prywatnym mailu,
a każde publiczne wyjście pokazuje archetyp („Innowacja i wzrost”), żeby nie przypisywać
żyjącym osobom rekomendacji KUP/SPRZEDAJ. Publiczne wyjścia to dziś przykładowy mail
w `docs/img/` (z `sample_data`, renderowany `main_tools.py render-sample-report`),
strona i `scorecard.json` publicznego scorecardu. Ledger atestacji niesie tylko werdykt
rady (rekomendacja, siła konsensusu) — bez głosów person. Klucz persony w bazie
(`council_votes`, ranking) zostaje nazwą, więc mapę daje `archetypes.toml`.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src.application.report_evidence import PaperCosts
from src.application.report_models import ResolvedPrediction
from src.application.report_public_scorecard import (
    render_public_scorecard_html,
    render_public_scorecard_json,
)
from src.domain.equity_curve import equity_curve
from src.infrastructure.persona_loader import (
    DEFAULT_PERSONAS_DIR,
    load_council_personas,
    load_persona_archetypes,
)
from src.tools.render_sample_report import render_sample_report

ROOT = Path(__file__).parent.parent
PUBLISHED_SAMPLE_EMAIL = ROOT / "docs" / "img" / "report-email.html"


def _persona_names() -> list[str]:
    return [persona.name for persona in load_council_personas(DEFAULT_PERSONAS_DIR)]


def _named_personas(text: str) -> list[str]:
    """Persony, których imię i nazwisko albo samo nazwisko stoi w tekście jako słowo."""
    found = []
    for name in _persona_names():
        surname = name.split()[-1]
        if re.search(rf"\b({re.escape(name)}|{re.escape(surname)})\b", text):
            found.append(name)
    return found


def test_every_persona_has_an_archetype_without_a_name() -> None:
    archetypes = load_persona_archetypes(DEFAULT_PERSONAS_DIR)

    assert sorted(archetypes) == sorted(_persona_names())
    assert len(set(archetypes.values())) == len(archetypes), "dwie persony, jeden archetyp"
    assert [label for label in archetypes.values() if _named_personas(label)] == []


def test_sample_email_shows_archetypes_not_persona_names() -> None:
    sample = render_sample_report()
    archetypes = set(load_persona_archetypes(DEFAULT_PERSONAS_DIR).values())

    assert _named_personas(sample.html) == []
    assert _named_personas(sample.text) == []
    assert any(label in sample.html for label in archetypes)


def test_published_sample_email_names_no_persona() -> None:
    assert _named_personas(PUBLISHED_SAMPLE_EMAIL.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("name", ["Warren Buffett", "George Soros"])
def test_public_scorecard_never_carries_a_persona_name(name: str) -> None:
    # Uzasadnienie predykcji bywa tekstem LLM-a, który może cytować radę — publiczna
    # strona i JSON nie mogą go przenieść.
    resolved = [
        ResolvedPrediction(
            symbol="NVDA",
            predicted_trend="BULLISH",
            is_correct=True,
            reasoning=f"{name} głosował KUP.",
            price_at_prediction=Decimal("100"),
            actual_price=Decimal("103"),
        )
    ]
    curve = equity_curve([("BULLISH", 0.03)])
    samples = [(0.7, True)]

    html = render_public_scorecard_html(resolved, curve, samples)
    payload = json.dumps(
        render_public_scorecard_json(
            resolved,
            curve,
            samples,
            generated_on=date(2026, 10, 25),
            costs=PaperCosts(cost_bps=10.0, crypto_cost_bps=25.0),
        ),
        ensure_ascii=False,
    )

    assert html != ""
    assert _named_personas(html) == []
    assert _named_personas(payload) == []
