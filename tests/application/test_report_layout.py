"""Układ maila (MAIL-17): strefa „Na dziś” przed separatorem „Zaplecze”.

Gmail przycina wiadomość powyżej ~102 KB i chowa jej dół, więc akcyjna część
(lead, portfel, zmiany od wczoraj, sygnały, predykcje, krypto) stoi na górze,
a sekcje meta (ryzyko, Track Record, model, rada, koszt, pominięte, błędy) za
separatorem. Kolejność sprawdzamy na pełnym cyklu z golden snapshotu (TEST-02),
który niesie dane każdej sekcji.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import ReportContext, SymbolResult, build_html_report
from src.domain.cycle_maturity import SkipReason
from tests.application.test_report_builder_snapshot import (
    DURATION_SECONDS,
    STARTED,
    _full_context,
    _full_cycle,
)

BACKOFFICE = "🗄️ Zaplecze"

TODAY_HTML = (
    "⚡ Najważniejsze w tym cyklu",
    "Sygnały ponad próg",
    "Wiarygodność modelu",
    "💼 Twój portfel",
    "📊 Zamknięte predykcje",
    "🔄 Zmiany nastawienia",
    "🎯 Najsilniejsze sygnały",
    "🔮 Wygenerowane predykcje",
    "💡 Uzasadnienia",
    "🪙 Krypto",
)

BACKOFFICE_HTML = (
    "🚨 Risk Watch",
    "🛰️ Alpha Signals",
    "📈 Track Record",
    "🩺 Karta kondycji modelu",
    "🏆 Rada — ranking wiarygodności",
    "🗳️ Rada — historia głosów",
    "💸 Koszt cyklu",
    "⏸ Pominięte",
    "⚠️ Błędy",
)

TODAY_TEXT = (
    "NAJWAŻNIEJSZE W TYM CYKLU",
    "Sygnały ponad próg:",
    "Wiarygodność modelu:",
    "=== 💼 Twój portfel",
    "ZAMKNIĘTE PREDYKCJE",
    "NAJSILNIEJSZE SYGNAŁY",
    "\nPREDYKCJE\n",
    "\nKRYPTO\n",
)

BACKOFFICE_TEXT = (
    "Risk Watch",
    "PORTFOLIO RADAR",
    "HISTORIA TRAFNOŚCI",
    "POMINIĘTE",
    "BŁĘDY",
)


def _positions(document: str, markers: tuple[str, ...]) -> list[int]:
    return [document.index(marker) for marker in markers]


def test_html_today_zone_comes_first_in_the_listed_order() -> None:
    html, _ = _full_cycle()

    today = _positions(html, TODAY_HTML)
    assert today == sorted(today)
    assert today[-1] < html.index(BACKOFFICE)


def test_html_backoffice_zone_follows_the_separator_in_the_listed_order() -> None:
    html, _ = _full_cycle()

    backoffice = _positions(html, BACKOFFICE_HTML)
    assert backoffice == sorted(backoffice)
    assert html.index(BACKOFFICE) < backoffice[0]
    assert backoffice[-1] < html.index("nie stanowi rekomendacji")


def test_plain_text_follows_the_same_two_zones() -> None:
    _, text = _full_cycle()

    today = _positions(text, TODAY_TEXT)
    backoffice = _positions(text, BACKOFFICE_TEXT)
    assert today == sorted(today)
    assert backoffice == sorted(backoffice)
    assert today[-1] < text.index("ZAPLECZE") < backoffice[0]


def test_credibility_bar_gives_directional_hit_rate_and_n_on_the_first_screen() -> None:
    # 23 trafienia na 40 prognoz kierunkowych w oknie 30 dni (`_full_context`).
    html, text = _full_cycle()

    bar = "trafność kierunkowa 57,5% (n=40, ostatnie 30 dni)"
    assert bar in html
    assert html.index(bar) < html.index("🎯 Najsilniejsze sygnały")
    assert f"Wiarygodność modelu: {bar}" in text
    assert text.index(bar) < text.index("NAJSILNIEJSZE SYGNAŁY")


def test_credibility_bar_is_hidden_without_directional_counts() -> None:
    context = dataclasses.replace(
        _full_context(), accuracy_stats={"mean_accuracy": 0.6, "sample_count": 10,
                                         "correct_count": 6, "days_window": 30}
    )
    html, text = build_html_report([], STARTED, DURATION_SECONDS, context)

    assert "Wiarygodność modelu" not in html
    assert "Wiarygodność modelu" not in text


def test_credibility_bar_without_directional_calls_says_so() -> None:
    context = ReportContext(
        accuracy_stats={
            "mean_accuracy": 1.0, "sample_count": 3, "correct_count": 3,
            "days_window": 30, "directional_count": 0, "directional_correct_count": 0,
        }
    )
    html, text = build_html_report([], STARTED, DURATION_SECONDS, context)

    assert "trafność kierunkowa — (brak prognoz kierunkowych, ostatnie 30 dni)" in html
    assert "trafność kierunkowa — (brak prognoz kierunkowych, ostatnie 30 dni)" in text


def test_new_reference_point_is_a_change_since_yesterday_in_the_today_zone() -> None:
    stale = SymbolResult(
        symbol="AAPL", status="ignored", delta=Decimal("0.12"),
        current_price=Decimal("250"), skip_reason=SkipReason.STALE_REFERENCE,
        reference_at=datetime(2026, 9, 19, 5, 30, tzinfo=UTC),
    )
    html, text = build_html_report([stale], STARTED, DURATION_SECONDS)

    assert html.index("🔄 Nowy punkt odniesienia") < html.index(BACKOFFICE)
    assert text.index("NOWY PUNKT ODNIESIENIA") < text.index("ZAPLECZE")

