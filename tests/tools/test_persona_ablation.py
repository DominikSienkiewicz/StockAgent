"""WIA-46 — ablacja ważenia person prequential: werdykt rady z wagami z chwili
predykcji (tylko głosy rozliczone wcześniej) kontra werdykt bez wag."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.domain.persona_track_record import PersonaVoteOutcome
from src.tools.persona_ablation import format_ablation, replay_persona_weighting

_T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _vote(
    prediction_id: str,
    investor: str,
    recommendation: str,
    confidence: float,
    *,
    at: datetime,
    actual: str = "110",
    symbol: str = "AAPL",
) -> PersonaVoteOutcome:
    return PersonaVoteOutcome(
        prediction_id=prediction_id,
        investor_name=investor,
        recommendation=recommendation,
        confidence=confidence,
        symbol=symbol,
        price_at_prediction=Decimal("100"),
        actual_price=Decimal(actual),
        predicted_at=at,
        settled_at=at + timedelta(days=1),
    )


def _two_cycles() -> list[PersonaVoteOutcome]:
    # Oba cykle: Wood BUY 0.6 kontra Burry SELL 0.7, a cena rośnie. Bez wag wygrywa
    # SELL; po pierwszym rozliczeniu Wood waży 1.5, a Burry 0.5, więc wygrywa BUY.
    t1 = _T0 + timedelta(days=2)
    return [
        _vote("p-1", "Wood", "BUY", 0.6, at=_T0),
        _vote("p-1", "Burry", "SELL", 0.7, at=_T0),
        _vote("p-2", "Wood", "BUY", 0.6, at=t1),
        _vote("p-2", "Burry", "SELL", 0.7, at=t1),
    ]


def test_weights_come_only_from_votes_settled_before_each_prediction() -> None:
    report = replay_persona_weighting(_two_cycles(), window_days=90)

    # Wagi „od dziś” znałyby wynik p-1 i odwróciłyby też jej werdykt (2 trafienia).
    assert report.evaluated == 2
    assert report.without_weights == 1
    assert (report.weighted_hits, report.unweighted_hits) == (1, 0)
    assert (report.weighted_only, report.unweighted_only) == (1, 0)


def test_since_limits_the_scored_predictions_but_not_the_history() -> None:
    report = replay_persona_weighting(_two_cycles(), since=_T0 + timedelta(days=1), window_days=90)

    assert report.evaluated == 1
    assert report.without_weights == 0
    assert report.weighted_hits == 1


def test_hold_on_crypto_is_scored_with_the_crypto_band() -> None:
    outcomes = [
        _vote("p-1", "Wood", "HOLD", 0.9, at=_T0, actual="102", symbol="BTC"),
        _vote("p-1", "Burry", "BUY", 0.1, at=_T0, actual="102", symbol="BTC"),
    ]

    report = replay_persona_weighting(outcomes, crypto_symbols={"BTC"})

    assert report.unweighted_hits == 1


def test_sign_test_needs_discordant_verdicts() -> None:
    report = replay_persona_weighting(_two_cycles()[:2])

    assert report.p_value is None


def test_format_ties_the_flag_to_the_out_of_sample_result() -> None:
    text = format_ablation(replay_persona_weighting(_two_cycles()), days=30, window_days=90)

    assert "prequential" in text
    assert "persona_weighting_enabled" in text
    assert "n=2" in text
