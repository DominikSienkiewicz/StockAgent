"""WIA-45 — redundancja rady: kappa Cohena i efektywna liczba niezależnych głosów."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.domain.council import Recommendation
from src.domain.council_redundancy import (
    EFFECTIVELY_SINGLE_VOTER_N_EFF,
    LLM_RATER,
    CouncilVoteSample,
    assess_council_redundancy,
    cohen_kappa,
    effective_voters,
    raw_agreement,
)

_CYCLE: tuple[Recommendation, ...] = ("BUY", "SELL", "HOLD", "BUY", "SELL", "HOLD", "BUY")


def _votes(
    investor: str, recommendations: Sequence[Recommendation], offset: int = 0
) -> list[CouncilVoteSample]:
    return [
        CouncilVoteSample(f"p-{index + offset}", investor, rec)
        for index, rec in enumerate(recommendations)
    ]


def _varied(length: int = 30) -> list[Recommendation]:
    return [_CYCLE[index % len(_CYCLE)] for index in range(length)]


def test_identical_voters_give_effective_n_one() -> None:
    votes = [vote for name in ("Graham", "Buffett", "Marks") for vote in _votes(name, _varied())]

    report = assess_council_redundancy(votes)

    assert report is not None
    assert report.mean_kappa == pytest.approx(1.0)
    assert report.n_eff == pytest.approx(1.0)
    assert report.is_effectively_single_voter is True
    assert report.personas == 3


def test_kappa_zero_for_independent_votes() -> None:
    # Pełny iloczyn kartezjański etykiet: zgodność 1/3 równa się oczekiwanej z przypadku.
    labels: list[Recommendation] = ["BUY", "SELL", "HOLD"]
    first = [a for a in labels for _ in labels]
    second = [b for _ in labels for b in labels]

    assert raw_agreement(first, second) == pytest.approx(1 / 3)
    assert cohen_kappa(first, second) == pytest.approx(0.0)


def test_independent_voters_keep_all_their_votes() -> None:
    labels: list[Recommendation] = ["BUY", "SELL", "HOLD"]
    first = [a for a in labels for _ in labels] * 2
    second = [b for _ in labels for b in labels] * 2
    votes = _votes("Alfa", first) + _votes("Beta", second)

    report = assess_council_redundancy(votes)

    assert report is not None
    assert report.mean_kappa == pytest.approx(0.0)
    assert report.n_eff == pytest.approx(2.0)
    assert report.is_effectively_single_voter is False


def test_opposite_voters_are_not_credited_more_than_n_votes() -> None:
    # Kappa ujemna (systematyczna niezgoda) nie daje N_eff > N.
    assert effective_voters(2, -0.5) == pytest.approx(2.0)


def test_effective_voters_formula() -> None:
    # 7 person, ρ̄ = 0,5 → 7 / (1 + 6 · 0,5) = 1,75.
    assert effective_voters(7, 0.5) == pytest.approx(1.75)
    assert effective_voters(7, 0.0) == pytest.approx(7.0)
    assert effective_voters(1, 0.9) == pytest.approx(1.0)


def test_kappa_is_unstable_under_class_dominance_so_the_raw_agreement_is_reported() -> None:
    # Paradoks kappa: zgodność 95%, a prawie wszystko to HOLD → kappa wyraźnie niższa.
    first = ["HOLD"] * 19 + ["BUY"]
    second = ["HOLD"] * 19 + ["SELL"]
    votes = _votes("Alfa", first) + _votes("Beta", second)  # type: ignore[arg-type]

    report = assess_council_redundancy(votes)

    assert report is not None
    (pair,) = report.pairs
    assert pair.agreement == pytest.approx(0.95)
    assert pair.kappa < 0.7


def test_two_constant_identical_voters_agree_fully() -> None:
    assert cohen_kappa(["HOLD"] * 12, ["HOLD"] * 12) == 1.0


def test_kappa_needs_pairs() -> None:
    assert cohen_kappa([], []) is None
    assert raw_agreement([], []) is None


def test_pairs_with_too_little_overlap_are_left_out() -> None:
    votes = _votes("Alfa", _varied(5)) + _votes("Beta", _varied(5))

    assert assess_council_redundancy(votes) is None


def test_pairs_use_only_the_predictions_both_personas_voted_on() -> None:
    # Beta głosowała na predykcjach 5..34, Alfa na 0..29: wspólne okno to 25.
    votes = _votes("Alfa", _varied(30)) + _votes("Beta", _varied(30), offset=5)

    report = assess_council_redundancy(votes)

    assert report is not None
    assert report.pairs[0].n == 25
    assert report.predictions == 35


def test_most_redundant_pair_comes_first() -> None:
    shifted = _varied()[1:] + _varied()[:1]
    votes = _votes("Graham", _varied()) + _votes("Buffett", _varied()) + _votes("Wood", shifted)

    report = assess_council_redundancy(votes)

    assert report is not None
    first = report.pairs[0]
    assert {first.first, first.second} == {"Graham", "Buffett"}
    assert first.kappa == pytest.approx(1.0)


def test_persona_agreement_with_the_llm_trend() -> None:
    recommendations = _varied()
    votes = _votes("Alfa", recommendations) + _votes("Beta", recommendations)
    trends = {
        f"p-{index}": {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "SIDEWAYS"}[rec]
        for index, rec in enumerate(recommendations)
    }

    report = assess_council_redundancy(votes, trends)

    assert report is not None
    assert {pair.first for pair in report.llm_pairs} == {"Alfa", "Beta"}
    assert all(pair.second == LLM_RATER for pair in report.llm_pairs)
    assert all(pair.kappa == pytest.approx(1.0) for pair in report.llm_pairs)


def test_single_voter_threshold_is_between_one_and_two() -> None:
    assert 1.0 < EFFECTIVELY_SINGLE_VOTER_N_EFF < 2.0


def test_personas_without_shared_predictions_make_no_pair() -> None:
    votes = _votes("Alfa", _varied(12)) + _votes("Beta", _varied(12), offset=100)

    assert assess_council_redundancy(votes, min_overlap=0) is None


def test_persona_without_shared_predictions_with_the_llm_is_left_out() -> None:
    recommendations = _varied(12)
    votes = _votes("Alfa", recommendations) + _votes("Beta", recommendations)
    trends = {f"p-{index + 500}": "BULLISH" for index in range(len(recommendations))}

    report = assess_council_redundancy(votes, trends, min_overlap=0)

    assert report is not None
    assert report.llm_pairs == ()
