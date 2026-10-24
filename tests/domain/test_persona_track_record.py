"""Testy domeny rankingu wiarygodności person rady (#3).

Kontrakt zamrożony tu wprost odpowiada "największemu ryzyku" pozycji: szumowy
ranking przy małej próbce. Stąd dwie twarde asercje: próg `min_votes` żyje
w domenie (nie w SQL, nie w rendererze) i liczba głosów jest zawsze niesiona
obok hit-rate'u, żeby warstwa prezentacji mogła ją pokazać.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.domain.persona_track_record import (
    DEFAULT_MIN_VOTES,
    PersonaTrackRecord,
    PersonaVoteOutcome,
    assess_against_track_record,
    assess_persona_significance,
    outcome_known_at,
    persona_accuracy_weights,
    persona_accuracy_weights_as_of,
    persona_hit_stats,
    persona_vote_correct,
    rank_personas,
    sort_personas,
)
from src.domain.track_record import HitRateEvidence, ScoredCall, assess_directional_hit_rate
from src.domain.value_objects import AssetType


class TestRankPersonas:
    def test_empty_stats_yield_empty_ranking(self) -> None:
        assert rank_personas({}) == []

    def test_ranks_personas_by_hit_rate_descending(self) -> None:
        ranking = rank_personas(
            {"Wood": (0.41, 18), "Buffett": (0.68, 22), "Lynch": (0.55, 10)}
        )

        assert [r.investor_name for r in ranking] == ["Buffett", "Lynch", "Wood"]

    def test_record_carries_hit_rate_and_vote_count(self) -> None:
        (record,) = rank_personas({"Buffett": (0.68, 22)})

        assert record == PersonaTrackRecord(
            investor_name="Buffett", hit_rate=0.68, votes=22
        )

    def test_persona_below_min_votes_is_excluded(self) -> None:
        # Persona z jednym trafionym głosem miałaby 100% hit-rate i stała na
        # czele rankingu — próg odcina ten szum, zanim dojdzie do renderera.
        ranking = rank_personas(
            {"Nowicjusz": (1.0, 1), "Buffett": (0.68, 22)}, min_votes=5
        )

        assert [r.investor_name for r in ranking] == ["Buffett"]

    def test_persona_exactly_at_min_votes_is_included(self) -> None:
        ranking = rank_personas({"Graham": (0.6, 5)}, min_votes=5)

        assert [r.investor_name for r in ranking] == ["Graham"]

    def test_default_min_votes_is_five(self) -> None:
        assert DEFAULT_MIN_VOTES == 5
        assert rank_personas({"Nowicjusz": (1.0, 4)}) == []

    def test_min_votes_zero_keeps_every_persona_with_a_vote(self) -> None:
        ranking = rank_personas({"Nowicjusz": (1.0, 1)}, min_votes=0)

        assert [r.investor_name for r in ranking] == ["Nowicjusz"]

    def test_persona_without_votes_is_always_excluded(self) -> None:
        # Nawet przy min_votes=0 zerowa próbka nie ma hit-rate'u do pokazania.
        assert rank_personas({"Duch": (0.0, 0)}, min_votes=0) == []

    def test_negative_vote_count_is_excluded(self) -> None:
        assert rank_personas({"Uszkodzony": (0.9, -3)}, min_votes=0) == []

    def test_ties_broken_by_vote_count_then_name(self) -> None:
        # Remis hit-rate'u: wygrywa większa próbka (mocniejszy dowód), a przy
        # równej próbce alfabet — ranking musi być deterministyczny.
        ranking = rank_personas(
            {
                "Soros": (0.5, 10),
                "Dalio": (0.5, 30),
                "Marks": (0.5, 10),
            }
        )

        assert [r.investor_name for r in ranking] == ["Dalio", "Marks", "Soros"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1.4, 1.0), (-0.2, 0.0), (0.0, 0.0), (1.0, 1.0)],
    )
    def test_hit_rate_is_clamped_to_unit_interval(
        self, raw: float, expected: float
    ) -> None:
        (record,) = rank_personas({"Buffett": (raw, 22)}, min_votes=0)

        assert record.hit_rate == expected

    def test_ranking_is_a_new_list_not_a_view_of_input(self) -> None:
        stats = {"Buffett": (0.68, 22), "Wood": (0.41, 18)}
        ranking = rank_personas(stats)
        ranking.clear()

        assert len(stats) == 2


class TestPersonaTrackRecord:
    def test_is_frozen(self) -> None:
        record = PersonaTrackRecord(investor_name="Buffett", hit_rate=0.68, votes=22)

        with pytest.raises(FrozenInstanceError):
            record.hit_rate = 0.9  # type: ignore[misc]

    def test_hit_rate_pct_rounds_to_whole_percent(self) -> None:
        record = PersonaTrackRecord(investor_name="Buffett", hit_rate=0.684, votes=22)

        assert record.hit_rate_pct == 68


class TestWilsonRanking:
    """WIA-30: leaderboard po gołym hit-rate nagradza małe próby — 5/5 stoi przed
    40/50. Dolna granica przedziału Wilsona karze małą próbę."""

    def test_small_perfect_sample_ranks_below_large_strong_sample(self) -> None:
        ranking = rank_personas({"A": (1.0, 5), "B": (0.8, 50)}, rank_by="wilson")

        assert [r.investor_name for r in ranking] == ["B", "A"]

    def test_default_ranking_stays_by_hit_rate(self) -> None:
        ranking = rank_personas({"A": (1.0, 5), "B": (0.8, 50)})

        assert [r.investor_name for r in ranking] == ["A", "B"]

    def test_wilson_lower_bound_uses_n_eff_when_known(self) -> None:
        raw = PersonaTrackRecord(investor_name="A", hit_rate=0.8, votes=50)
        clustered = PersonaTrackRecord(investor_name="A", hit_rate=0.8, votes=50, n_eff=10.0)

        assert clustered.wilson_lower < raw.wilson_lower

    def test_sort_personas_reorders_by_the_chosen_key(self) -> None:
        records = [
            PersonaTrackRecord(investor_name="A", hit_rate=1.0, votes=5),
            PersonaTrackRecord(investor_name="B", hit_rate=0.8, votes=50),
        ]

        assert [r.investor_name for r in sort_personas(records, "wilson")] == ["B", "A"]
        assert [r.investor_name for r in sort_personas(records, "hit_rate")] == ["A", "B"]


class TestPersonaSignificance:
    """WIA-22: werdykt istotności każdej persony względem najlepszego naiwnego
    baseline'u, test dwumianowy na n_eff i korekta BH-FDR po personach — 7 person
    to 7 porównań naraz, więc „najlepsza persona” bywa selekcją szumu."""

    @staticmethod
    def _records(stats: dict[str, tuple[float, int]]) -> list[PersonaTrackRecord]:
        return rank_personas(stats, min_votes=0)

    def test_persona_clearly_above_the_baseline_is_significant(self) -> None:
        records = self._records(
            {"Buffett": (0.8, 100), "Wood": (0.5, 100), "Soros": (0.52, 100)}
        )

        assessed = assess_persona_significance(records, baseline=0.5)

        by_name = {r.investor_name: r for r in assessed}
        assert by_name["Buffett"].significant is True
        assert by_name["Wood"].significant is False
        assert by_name["Buffett"].verdict is not None
        assert by_name["Buffett"].verdict.startswith("istotnie lepsze")

    def test_bh_correction_cancels_a_lone_raw_win_among_seven_personas(self) -> None:
        # 32/50 przy p0 = 0.5: surowe p ≈ 0.03 < 0.05, ale BH przy 7 personach
        # wymaga p ≤ 0.10/7 ≈ 0.014 dla najmniejszej p-wartości.
        stats = {"Lucky": (0.64, 50)} | {f"P{i}": (0.5, 50) for i in range(6)}

        assessed = assess_persona_significance(self._records(stats), baseline=0.5)

        lucky = next(r for r in assessed if r.investor_name == "Lucky")
        assert lucky.p_value is not None and lucky.p_value < 0.05
        assert lucky.significant is False
        assert lucky.verdict is not None
        assert lucky.verdict.startswith("nieodróżnialne od rzutu monetą")

    def test_design_effect_shrinks_the_effective_sample(self) -> None:
        records = self._records({"Buffett": (0.6, 100)})

        (assessed,) = assess_persona_significance(records, baseline=0.5, design_effect=4.0)

        assert assessed.n_eff == pytest.approx(25.0)
        assert assessed.verdict is not None
        assert "n=100" in assessed.verdict
        assert "n_eff≈25" in assessed.verdict

    def test_order_and_counts_are_preserved(self) -> None:
        records = self._records({"Buffett": (0.68, 22), "Wood": (0.41, 18)})

        assessed = assess_persona_significance(records, baseline=0.5)

        assert [(r.investor_name, r.votes) for r in assessed] == [("Buffett", 22), ("Wood", 18)]

    def test_unassessed_record_has_no_verdict(self) -> None:
        assert PersonaTrackRecord(investor_name="Buffett", hit_rate=0.68, votes=22).verdict is None


class TestAssessAgainstTrackRecord:
    """WIA-22: p0 to najlepszy naiwny baseline z okna scorecardu, a n_eff person
    przybliża efekt projektu predykcji tego okna (głosy nie mają własnych cykli)."""

    @staticmethod
    def _evidence() -> HitRateEvidence:
        # 4 cykle po 10 identycznych wyników → n_eff = 4, efekt projektu 10.
        calls = [
            ScoredCall(
                cycle=f"2026-09-0{day + 1}",
                predicted_trend="BULLISH",
                is_correct=day % 2 == 0,
                realized_change=0.01 if day % 2 == 0 else -0.01,
            )
            for day in range(4)
            for _ in range(10)
        ]
        return assess_directional_hit_rate(calls, n_boot=0)

    def test_uses_the_best_naive_baseline_and_the_design_effect(self) -> None:
        records = rank_personas({"Buffett": (0.9, 100)}, min_votes=0)

        (assessed,) = assess_against_track_record(records, self._evidence())

        assert assessed.n_eff == pytest.approx(10.0)
        assert assessed.p_value is not None

    def test_without_baseline_records_stay_unassessed(self) -> None:
        records = rank_personas({"Buffett": (0.9, 100)}, min_votes=0)
        empty = assess_directional_hit_rate([], n_boot=0)

        assert assess_against_track_record(records, empty) == records


_T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _outcome(
    *,
    investor: str = "Buffett",
    recommendation: str = "HOLD",
    symbol: str = "AAPL",
    price: str = "100",
    actual: str = "102",
    predicted_at: datetime = _T0,
    settled_at: datetime | None = None,
    prediction_id: str = "p-1",
    confidence: float = 0.8,
) -> PersonaVoteOutcome:
    return PersonaVoteOutcome(
        prediction_id=prediction_id,
        investor_name=investor,
        recommendation=recommendation,
        confidence=confidence,
        symbol=symbol,
        price_at_prediction=Decimal(price),
        actual_price=Decimal(actual),
        predicted_at=predicted_at,
        settled_at=settled_at,
    )


class TestPersonaVoteCorrect:
    """WIA-51 (G45): reguła trafności głosu persony żyje w domenie, ze stałymi
    z `prediction.py` — SQL z 015/018 oceniał HOLD na krypto pasmem ±0.5%."""

    def test_hold_on_crypto_uses_crypto_tolerance(self) -> None:
        # +2% mieści się w paśmie krypto ±2.5%, a w paśmie akcji ±0.5% już nie.
        assert persona_vote_correct("HOLD", Decimal("100"), Decimal("102"), AssetType.CRYPTO)

    def test_hold_on_stock_keeps_the_stock_tolerance(self) -> None:
        assert not persona_vote_correct("HOLD", Decimal("100"), Decimal("102"))
        assert persona_vote_correct("HOLD", Decimal("100"), Decimal("100.4"))

    def test_buy_and_sell_follow_the_direction_of_the_move(self) -> None:
        assert persona_vote_correct("BUY", Decimal("100"), Decimal("101"))
        assert not persona_vote_correct("BUY", Decimal("100"), Decimal("99"))
        assert persona_vote_correct("SELL", Decimal("100"), Decimal("99"))
        assert not persona_vote_correct("SELL", Decimal("100"), Decimal("101"))

    def test_recommendation_is_parsed_like_the_council_parses_it(self) -> None:
        assert persona_vote_correct(" buy ", Decimal("100"), Decimal("101"))

    def test_unknown_recommendation_is_a_miss(self) -> None:
        # Jak `ELSE 0.0` w SQL: głos się liczy, ale nie jest trafieniem.
        assert not persona_vote_correct("STRONG_BUY", Decimal("100"), Decimal("101"))


class TestPersonaHitStats:
    def test_crypto_tolerance_is_chosen_per_symbol(self) -> None:
        outcomes = [
            _outcome(symbol="BTC", prediction_id="p-btc"),
            _outcome(symbol="AAPL", prediction_id="p-aapl"),
        ]

        stats = persona_hit_stats(outcomes, crypto_symbols=frozenset({"BTC"}))

        assert stats == {"Buffett": (0.5, 2)}

    def test_stats_count_every_settled_vote_per_persona(self) -> None:
        outcomes = [
            _outcome(investor="Wood", recommendation="BUY", actual="110"),
            _outcome(investor="Wood", recommendation="BUY", actual="90", prediction_id="p-2"),
            _outcome(investor="Burry", recommendation="SELL", actual="90"),
        ]

        assert persona_hit_stats(outcomes) == {"Wood": (0.5, 2), "Burry": (1.0, 1)}

    def test_weights_use_the_same_rule_as_the_stats(self) -> None:
        # Waga konsensusu jak w 015: 0.5 + trafność, w [0.5, 1.5].
        outcomes = [
            _outcome(symbol="BTC", prediction_id="p-btc"),
            _outcome(symbol="AAPL", prediction_id="p-aapl"),
            _outcome(investor="Wood", recommendation="BUY", actual="90"),
        ]

        weights = persona_accuracy_weights(outcomes, crypto_symbols=frozenset({"BTC"}))

        assert weights == {"Buffett": pytest.approx(1.0), "Wood": pytest.approx(0.5)}


class TestPersonaWeightsAsOf:
    """WIA-46: ocena prequential — waga z chwili predykcji zna wyłącznie głosy
    rozliczone PRZED nią, a nie całe okno liczone od dziś."""

    def test_weights_as_of_ignore_votes_settled_after_cutoff(self) -> None:
        cutoff = _T0 + timedelta(days=2)
        outcomes = [
            _outcome(investor="Wood", recommendation="BUY", actual="110",
                     settled_at=_T0 + timedelta(days=1)),
            _outcome(investor="Burry", recommendation="BUY", actual="110",
                     settled_at=cutoff + timedelta(hours=1), prediction_id="p-2"),
        ]

        weights = persona_accuracy_weights_as_of(outcomes, as_of=cutoff, window_days=90)

        assert weights == {"Wood": pytest.approx(1.5)}

    def test_missing_or_floor_settlement_counts_from_the_horizon(self) -> None:
        # Bez stempla albo z dolną granicą backfillu 029 (`settled_at = timestamp`)
        # wynik znamy najwcześniej po horyzoncie prognozy (12 h).
        floor = _outcome(settled_at=_T0)
        missing = _outcome(settled_at=None)

        assert outcome_known_at(floor) == _T0 + timedelta(hours=12)
        assert outcome_known_at(missing) == _T0 + timedelta(hours=12)
        assert persona_accuracy_weights_as_of(
            [floor], as_of=_T0 + timedelta(hours=11), window_days=90
        ) == {}

    def test_window_counts_back_from_the_cutoff(self) -> None:
        cutoff = _T0 + timedelta(days=30)
        old = _outcome(predicted_at=_T0, settled_at=_T0 + timedelta(hours=13))

        assert persona_accuracy_weights_as_of([old], as_of=cutoff, window_days=7) == {}
        assert persona_accuracy_weights_as_of([old], as_of=cutoff, window_days=31) != {}
