"""WIA-56: czytnik kwitów decyzyjnych — „dlaczego agent tak przewidział?”.

`decision_receipts` (migracja 020) był write-only: jedynym miejscem w `src`, które
go dotykało, był zapis w grafie. Audit trail, którego nikt nie czyta, nie jest
dowodem. Narzędzie składa z kwitu i głosów rady odpowiedź dla jednej predykcji
(po id) albo dla wszystkich predykcji symbolu z jednego dnia UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import Mock

from src.application.ports import RepositoryPort, StoredReceipt
from src.domain.council import InvestorOpinion
from src.tools.explain import explain_prediction, explain_symbol_day

_ANALOG_ID = "3f2a1b9c-0d4e-4f5a-9b8c-7d6e5f4a3b2c"
_V1: dict[str, Any] = {
    "schema_version": 1,
    "effective_threshold": "0.03",
    "precedents": [
        {
            "id": _ANALOG_ID,
            "similarity": 0.91,
            "predicted_trend": "BEARISH",
            "is_trend_correct": True,
            "summary": "Fed hawkish",
        }
    ],
    "feature_attribution": [{"feature": "price_delta", "contribution": 0.0123}],
}


def _stored(payload: Any, *, prediction_id: str = "pred-1") -> StoredReceipt:
    return StoredReceipt(
        prediction_id=prediction_id,
        symbol="NVDA",
        predicted_at=datetime(2026, 8, 14, 5, 31, tzinfo=UTC),
        predicted_trend="BULLISH",
        is_trend_correct=False,
        payload=payload,
    )


def _repo(stored: StoredReceipt | None, votes: list[InvestorOpinion] | None = None) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_receipt.return_value = stored
    repo.get_council_votes_for_prediction.return_value = votes or []
    return repo


def test_v1_receipt_shows_the_effective_threshold_and_analog_ids() -> None:
    out = explain_prediction(_repo(_stored(_V1)), "pred-1")

    assert "Próg efektywny" in out
    assert "3,0%" in out
    assert _ANALOG_ID in out
    assert "0,91" in out
    assert "price_delta" in out
    assert "NVDA" in out
    assert "chybiona" in out


def test_unknown_schema_version_is_a_message_not_an_exception() -> None:
    out = explain_prediction(_repo(_stored({**_V1, "schema_version": 3})), "pred-1")

    assert "nieznana wersja schematu kwitu (schema_version=3)" in out
    assert "Próg efektywny" not in out


def test_council_votes_are_listed() -> None:
    vote = InvestorOpinion(
        investor_name="Buffett",
        recommendation="HOLD",
        confidence=0.7,
        reasoning="Wycena wysoka.",
        key_factors=("P/E",),
    )

    out = explain_prediction(_repo(_stored(_V1), [vote]), "pred-1")

    assert "Buffett: HOLD (pewność 70%)" in out


def test_prediction_without_a_receipt_says_so() -> None:
    out = explain_prediction(_repo(_stored(None)), "pred-1")

    assert "brak kwitu decyzyjnego" in out


def test_unknown_prediction_says_so() -> None:
    assert "nie znaleziono predykcji pred-404" in explain_prediction(_repo(None), "pred-404")


def test_symbol_and_date_explain_every_prediction_of_that_utc_day() -> None:
    repo = _repo(None)
    repo.find_receipts.return_value = [
        _stored(_V1, prediction_id="pred-1"),
        _stored({**_V1, "effective_threshold": "0.045"}, prediction_id="pred-2"),
    ]

    out = explain_symbol_day(repo, "NVDA", date(2026, 8, 14))

    repo.find_receipts.assert_called_once_with("NVDA", date(2026, 8, 14))
    assert "pred-1" in out
    assert "pred-2" in out
    assert "4,5%" in out


def test_symbol_and_date_without_predictions_says_so() -> None:
    repo = _repo(None)
    repo.find_receipts.return_value = []

    out = explain_symbol_day(repo, "NVDA", date(2026, 8, 14))

    assert "brak predykcji NVDA z 2026-08-14 (UTC)" in out


def test_routed_receipt_names_the_model_that_answered() -> None:
    # FIN-21: przy routingu modeli kwit v2 mówi, który model faktycznie odpowiedział.
    payload = {**_V1, "schema_version": 2, "model_used": "anthropic:claude-sonnet-4-6"}

    out = explain_prediction(_repo(_stored(payload)), "pred-1")

    assert "Model, który odpowiedział: anthropic:claude-sonnet-4-6" in out
