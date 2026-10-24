"""WIA-56: skrót kwitu decyzyjnego pod chybieniem w sekcji zamkniętych predykcji.

Post-mortem mówił, co się stało i dlaczego teza nie wyszła, ale nie — na czym
predykcja stała. Jedna linia z kwitu (`decision_receipts`, migracja 020): analogi
RAG z podobieństwem i próg efektywny po mnożnikach reżimu i earnings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.application.decision_receipt import read_decision_receipt, receipt_line_pl
from src.application.report_builder import (
    ReportContext,
    build_html_report,
    parse_resolved_predictions,
)

_RECEIPT: dict[str, Any] = {
    "schema_version": 1,
    "effective_threshold": "0.03",
    "precedents": [
        {"id": "3f2a1b9c-0d4e-4f5a-9b8c-7d6e5f4a3b2c", "similarity": 0.91},
        {"id": "77aa0e12-5555-4444-8888-999999999999", "similarity": 0.873},
        {"summary": "analog bez id", "similarity": 0.5},
    ],
    "feature_attribution": [],
}
_STARTED_AT = datetime(2026, 10, 22, 5, 30, tzinfo=UTC)


def _row(correct: bool, receipt: Any = _RECEIPT) -> dict[str, Any]:
    return {
        "symbol": "NVDA",
        "predicted_trend": "BULLISH",
        "is_trend_correct": correct,
        "price_at_prediction": "100",
        "actual_price_after_12h": "97",
        "decision_receipts": receipt,
    }


def test_line_names_the_analogs_with_similarity_and_the_threshold() -> None:
    receipt = read_decision_receipt(_RECEIPT)
    assert receipt is not None

    assert receipt_line_pl(receipt) == (
        "stała na analogach #3f2a1b9c (0,91), #77aa0e12 (0,87) przy progu 3,0%"
    )


def test_line_without_analogs_says_so() -> None:
    receipt = read_decision_receipt({**_RECEIPT, "precedents": []})
    assert receipt is not None

    assert receipt_line_pl(receipt) == "bez analogów RAG, przy progu 3,0%"


def test_parsed_resolved_prediction_carries_its_receipt() -> None:
    [resolved] = parse_resolved_predictions([_row(False)])

    assert resolved.receipt is not None
    assert resolved.receipt.effective_threshold == "0.03"


def test_unknown_or_missing_receipt_parses_to_none() -> None:
    unknown, missing = parse_resolved_predictions(
        [_row(False, {**_RECEIPT, "schema_version": 9}), _row(False, None)]
    )

    assert unknown.receipt is None
    assert missing.receipt is None


def test_miss_shows_one_receipt_line_in_html_and_text() -> None:
    context = ReportContext(resolved_predictions=parse_resolved_predictions([_row(False)]))

    html, text = build_html_report([], _STARTED_AT, 1.0, context)

    line = "stała na analogach #3f2a1b9c (0,91), #77aa0e12 (0,87) przy progu 3,0%"
    assert f"Kwit:</strong> {line}" in html
    assert f"kwit: {line}" in text


def test_hit_has_no_receipt_line() -> None:
    context = ReportContext(resolved_predictions=parse_resolved_predictions([_row(True)]))

    html, text = build_html_report([], _STARTED_AT, 1.0, context)

    assert "stała na analogach" not in html
    assert "stała na analogach" not in text
