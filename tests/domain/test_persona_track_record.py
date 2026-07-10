"""Testy domeny rankingu wiarygodności person rady (#3).

Kontrakt zamrożony tu wprost odpowiada "największemu ryzyku" pozycji: szumowy
ranking przy małej próbce. Stąd dwie twarde asercje: próg `min_votes` żyje
w domenie (nie w SQL, nie w rendererze) i liczba głosów jest zawsze niesiona
obok hit-rate'u, żeby warstwa prezentacji mogła ją pokazać.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.domain.persona_track_record import (
    DEFAULT_MIN_VOTES,
    PersonaTrackRecord,
    rank_personas,
)


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
