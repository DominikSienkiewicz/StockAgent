"""WIA-43 — leave-one-out persony na zapisanych głosach rady (offline, domena)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.domain.council_ablation import CouncilAblation, PersonaMarginal, leave_one_out
from src.domain.persona_track_record import PersonaVoteOutcome

_T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _vote(
    prediction_id: str,
    investor: str,
    recommendation: str,
    confidence: float,
    *,
    at: datetime = _T0,
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


def _marginal(report: CouncilAblation, name: str) -> PersonaMarginal:
    found = [p for p in report.personas if p.investor_name == name]
    assert found, f"{name} is missing from the report"
    return found[0]


def _pivotal_council(prediction_id: str = "p-1", at: datetime = _T0) -> list[PersonaVoteOutcome]:
    # BUY 0.9 + BUY 0.5 przeważa SELL 0.6; cena rośnie, więc rada trafia. Bez Alfy
    # BUY 0.5 przegrywa z SELL 0.6 — Alfa przesądza o trafnym werdykcie.
    return [
        _vote(prediction_id, "Alfa", "BUY", 0.9, at=at),
        _vote(prediction_id, "Beta", "BUY", 0.5, at=at),
        _vote(prediction_id, "Gamma", "SELL", 0.6, at=at),
    ]


def test_removing_pivotal_persona_flips_outcome() -> None:
    report = leave_one_out(_pivotal_council())

    alfa = _marginal(report, "Alfa")
    assert alfa.flips == 1
    assert alfa.accuracy_with == 1.0
    assert alfa.accuracy_without == 0.0
    assert alfa.accuracy_delta == -1.0
    assert alfa.removal_candidate is False
    # Usunięcie Bety albo Gammy zostawia werdykt BUY.
    assert _marginal(report, "Beta").flips == 0
    assert _marginal(report, "Gamma").flips == 0


def test_redundant_persona_has_zero_marginal_contribution() -> None:
    votes = [
        _vote("p-1", "Alfa", "BUY", 0.8),
        _vote("p-1", "Beta", "BUY", 0.8),
        _vote("p-1", "Gamma", "BUY", 0.8),
    ]

    report = leave_one_out(votes)

    for name in ("Alfa", "Beta", "Gamma"):
        marginal = _marginal(report, name)
        assert marginal.flips == 0
        assert marginal.accuracy_delta == 0.0


def test_a_persona_whose_removal_repairs_the_verdict_is_a_removal_candidate() -> None:
    # Beta (SELL 0.9) przesądza o chybionym werdykcie SELL; bez niej BUY trafia.
    votes = [
        _vote("p-1", "Alfa", "BUY", 0.5),
        _vote("p-1", "Beta", "SELL", 0.9),
        _vote("p-1", "Gamma", "BUY", 0.3),
    ]

    beta = _marginal(leave_one_out(votes), "Beta")

    assert beta.accuracy_delta == 1.0
    assert beta.removal_candidate is True


def test_brier_scores_the_consensus_strength_against_the_hit() -> None:
    report = leave_one_out(_pivotal_council())

    alfa = _marginal(report, "Alfa")
    # Pełna rada: BUY 1.4 z 2.0 → siła 0.7, trafienie → (0.7 − 1)² = 0.09.
    assert alfa.brier_with == pytest.approx(0.09)
    # Bez Alfy: SELL 0.6 z 1.1 → siła ≈ 0.545, pudło → 0.545².
    assert alfa.brier_without == pytest.approx((0.6 / 1.1) ** 2)
    assert report.brier == pytest.approx(0.09)
    assert report.accuracy == 1.0


def test_predictions_with_a_single_vote_are_skipped() -> None:
    report = leave_one_out([_vote("p-1", "Alfa", "BUY", 0.9)])

    assert report.predictions == 0
    assert report.personas == ()
    assert report.accuracy is None


def test_since_scores_only_later_predictions() -> None:
    early = _pivotal_council("p-early", at=_T0)
    late = _pivotal_council("p-late", at=_T0 + timedelta(days=10))

    report = leave_one_out([*early, *late], since=_T0 + timedelta(days=5))

    assert report.predictions == 1


def test_crypto_hold_is_scored_with_the_wider_sideways_band() -> None:
    # HOLD na BTC +2% mieści się w paśmie ±2.5% krypto (±0.5% dla akcji).
    votes = [
        _vote("p-1", "Alfa", "HOLD", 0.9, actual="102", symbol="BTC"),
        _vote("p-1", "Beta", "HOLD", 0.8, actual="102", symbol="BTC"),
    ]

    assert leave_one_out(votes, crypto_symbols={"BTC"}).accuracy == 1.0
    assert leave_one_out(votes).accuracy == 0.0


def test_block_interval_needs_enough_cycles() -> None:
    few = leave_one_out(_pivotal_council())
    assert _marginal(few, "Alfa").accuracy_interval is None

    votes = [
        vote
        for day in range(25)
        for vote in _pivotal_council(f"p-{day}", at=_T0 + timedelta(days=day))
    ]
    many = leave_one_out(votes, n_boot=200)

    interval = _marginal(many, "Alfa").accuracy_interval
    assert interval is not None
    assert interval[0] <= -1.0 + 1e-9
    assert interval[1] <= 0.0


def test_personas_are_listed_removal_candidates_first() -> None:
    votes = [
        _vote("p-1", "Alfa", "BUY", 0.5),
        _vote("p-1", "Beta", "SELL", 0.9),
        _vote("p-1", "Gamma", "BUY", 0.3),
    ]

    names = [p.investor_name for p in leave_one_out(votes).personas]

    assert names[:1] == ["Beta"]


def test_unparsable_recommendations_and_duplicate_votes_do_not_change_the_council() -> None:
    votes = [
        *_pivotal_council(),
        _vote("p-1", "Alfa", "SELL", 0.99),  # duplikat persony: liczy się pierwszy głos
        _vote("p-1", "Delta", "STRONG_BUY", 0.99),  # spoza BUY/SELL/HOLD: odpada
    ]

    report = leave_one_out(votes)

    assert sorted(p.investor_name for p in report.personas) == ["Alfa", "Beta", "Gamma"]
    assert _marginal(report, "Alfa").flips == 1


def test_naive_timestamps_are_read_as_utc_cycles() -> None:
    naive = _T0.replace(tzinfo=None)
    votes = [
        vote
        for day in range(25)
        for vote in _pivotal_council(f"p-{day}", at=naive + timedelta(days=day))
    ]

    report = leave_one_out(votes, n_boot=50)

    assert _marginal(report, "Alfa").accuracy_interval is not None
