"""MAIL-23: układ mobilny — poranny mail na telefonie (~360 px) to podstawowy scenariusz.

Tabela predykcji ma 5 kolumn (sentyment, newsy i sygnał modelu idą do bloku
uzasadnień), boxy pierwszego ekranu i karty sygnałów to tabele z komórkami
`width:33%` zamiast `display:flex` (Outlook flexa nie zna), a media query
`max-width:480px` zamienia wiersze predykcji na bloki.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import SymbolResult, build_html_report
from src.application.report_styles import EMAIL_STYLESHEET

AT = datetime(2026, 5, 14, tzinfo=UTC)
SAVED = SymbolResult(
    symbol="AAPL",
    status="saved",
    delta=Decimal("0.025"),
    current_price=Decimal("298.87"),
    trend="BULLISH",
    target_price=Decimal("305.00"),
    confidence_score=0.85,
    reasoning="Strong macro tailwinds + positive earnings expectations.",
    sentiment_score=0.42,
    sentiment_label="Somewhat-Bullish",
    news_volume=12,
)


def _html() -> str:
    html, _ = build_html_report([SAVED], AT, 1.0)
    return html


def test_predictions_table_keeps_company_move_direction_target_confidence() -> None:
    table = _html().split("🔮 Wygenerowane predykcje")[1].split("</table>")[0]

    headers = re.findall(r"<th\b[^>]*>(.*?)</th>", table)
    assert headers == ["Spółka", "Zmiana (cykl)", "Kierunek", "Prognoza (nast. cykl)", "Pewność"]


def test_sentiment_news_and_model_signal_move_to_the_reasoning_block() -> None:
    reasoning = _html().split("💡 Uzasadnienia")[1]

    assert "Sentyment: Lekko pozytywny (+0,42)" in reasoning
    assert "Newsy (24 h): 12" in reasoning
    assert "KUP" in reasoning


def test_no_flex_layout_anywhere_in_the_mail() -> None:
    html = _html()

    assert "display: flex" not in html
    assert "display:flex" not in html
    assert "flex: 1" not in html


def test_first_screen_boxes_and_signal_cards_are_third_width_table_cells() -> None:
    html = _html()

    first_screen = html.split("Sygnały ponad próg")[0]
    assert "width: 33%" in first_screen.split("<table")[-1]
    cards = html.split("🎯 Najsilniejsze sygnały")[1].split("<h2")[0]
    assert cards.count("width: 33%") == 3


def test_narrow_screen_media_query_turns_prediction_rows_into_blocks() -> None:
    assert "@media (max-width:480px)" in EMAIL_STYLESHEET
    query = EMAIL_STYLESHEET.split("@media (max-width:480px)")[1]

    assert ".pt tr" in query
    assert "display:block" in query
