# tests/domain/test_consensus_shift.py
from src.domain.consensus_shift import (
    CONSENSUS_DROP_THRESHOLD,
    SENTIMENT_JUMP_THRESHOLD,
    STALE_GAP_HOURS,
    ConsensusShift,
    ShiftKind,
    detect_shift,
)
from src.domain.council import CouncilVerdict, InvestorOpinion


def _opinion(name: str, rec: str) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=rec,
        confidence=0.8,
        reasoning="...",
        key_factors=(),
    )


def _verdict(
    rec: str,
    consensus: float,
    opinions: tuple[InvestorOpinion, ...] = (),
) -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation=rec,
        consensus_strength=consensus,
        summary="...",
        dissenting_views=(),
        investor_opinions=opinions,
    )


class TestThresholdConstants:
    """Progi są zamrożone jako stałe modułu — warstwa wyżej i testy dzielą jedną prawdę."""

    def test_frozen_values(self):
        assert SENTIMENT_JUMP_THRESHOLD == 0.2
        assert CONSENSUS_DROP_THRESHOLD == 0.25
        assert STALE_GAP_HOURS == 48.0


class TestFlip:
    """FLIP = rada odwróciła kierunek (BUY↔SELL) — najbardziej actionable sygnał dnia."""

    def test_buy_to_sell_is_flip(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("SELL", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.FLIP
        assert result.previous_recommendation == "BUY"
        assert result.current_recommendation == "SELL"

    def test_sell_to_buy_is_flip(self):
        prev = _verdict("SELL", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.FLIP


class TestStaleGate:
    """NAJWIĘKSZE RYZYKO: fałszywa dramaturgia ze stęchłych danych. Dane, które
    normalnie dałyby FLIP, przy przeterminowanej luce MUSZĄ zdegradować do
    STALE_COMPARISON — próg wieku to wymaganie, nie opcja."""

    def test_flip_data_becomes_stale_at_72h(self):
        prev = _verdict("BUY", 0.9)
        curr = _verdict("SELL", 0.9)
        result = detect_shift(
            prev, curr, previous_sentiment=0.5, current_sentiment=-0.5, gap_hours=72.0
        )
        assert result.kind is ShiftKind.STALE_COMPARISON

    def test_boundary_exactly_48h_is_not_stale(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("SELL", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=48.0
        )
        assert result.kind is ShiftKind.FLIP

    def test_boundary_48_1h_is_stale(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("SELL", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=48.1
        )
        assert result.kind is ShiftKind.STALE_COMPARISON

    def test_stale_still_carries_raw_deltas_for_upper_layer(self):
        # Nawet gdy porównanie jest stęchłe, wynik niesie surowe dane, żeby
        # renderer mógł napisać "porównanie z cyklu sprzed N dni".
        prev = _verdict("BUY", 0.9)
        curr = _verdict("SELL", 0.6)
        result = detect_shift(
            prev, curr, previous_sentiment=0.4, current_sentiment=-0.1, gap_hours=72.0
        )
        assert result.previous_recommendation == "BUY"
        assert result.current_recommendation == "SELL"
        assert result.gap_hours == 72.0


class TestSoftening:
    """SOFTENING = osłabienie przekonania bez pełnego odwrócenia: spadek
    konsensusu, skok sentymentu albo zmiana rekomendacji do/z HOLD."""

    def test_consensus_drop_above_threshold(self):
        prev = _verdict("BUY", 0.5)
        curr = _verdict("BUY", 0.24)  # spadek 0.26 > 0.25
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.SOFTENING

    def test_consensus_drop_exactly_at_threshold_is_stable(self):
        prev = _verdict("BUY", 0.5)
        curr = _verdict("BUY", 0.25)  # spadek dokładnie 0.25, próg jest ostry (>)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.STABLE

    def test_consensus_increase_is_not_softening(self):
        prev = _verdict("BUY", 0.4)
        curr = _verdict("BUY", 0.9)  # wzrost konsensusu — nie osłabienie
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.STABLE

    def test_sentiment_jump_at_threshold(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.2, gap_hours=1.0
        )
        assert result.kind is ShiftKind.SOFTENING

    def test_sentiment_jump_below_threshold_is_stable(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.19, gap_hours=1.0
        )
        assert result.kind is ShiftKind.STABLE

    def test_negative_sentiment_swing_counts_by_magnitude(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.3, current_sentiment=-0.3, gap_hours=1.0
        )
        assert result.kind is ShiftKind.SOFTENING

    def test_recommendation_change_to_hold_is_softening_not_flip(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("HOLD", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert result.kind is ShiftKind.SOFTENING


class TestStable:
    """STABLE = nic nie przekroczyło progów, rada trzyma kurs."""

    def test_no_change_is_stable(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.1, current_sentiment=0.1, gap_hours=1.0
        )
        assert result.kind is ShiftKind.STABLE


class TestVotesChanged:
    """Reużycie CouncilVerdict.opinion_shift() do policzenia zmienionych głosów."""

    def test_counts_changed_votes_via_opinion_shift(self):
        prev = _verdict(
            "BUY",
            0.8,
            (_opinion("Burry", "SELL"), _opinion("Wood", "BUY"), _opinion("Marks", "HOLD")),
        )
        curr = _verdict(
            "SELL",
            0.8,
            (_opinion("Burry", "SELL"), _opinion("Wood", "SELL"), _opinion("Marks", "SELL")),
        )
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        # Wood BUY→SELL i Marks HOLD→SELL zmienili zdanie; Burry został przy SELL.
        assert result.votes_changed == 2


class TestCarriedFields:
    """Wynik niesie surowe dane dla warstwy prezentacji ('sprzed N dni', delty)."""

    def test_sentiment_and_consensus_deltas_are_signed(self):
        prev = _verdict("BUY", 0.9)
        curr = _verdict("SELL", 0.6)
        result = detect_shift(
            prev, curr, previous_sentiment=0.5, current_sentiment=0.0, gap_hours=24.0
        )
        assert result.sentiment_delta == -0.5
        assert abs(result.consensus_delta + 0.3) < 1e-9

    def test_gap_days_derives_from_hours(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=72.0
        )
        assert result.gap_days == 3.0

    def test_result_is_frozen_dataclass(self):
        prev = _verdict("BUY", 0.8)
        curr = _verdict("BUY", 0.8)
        result = detect_shift(
            prev, curr, previous_sentiment=0.0, current_sentiment=0.0, gap_hours=1.0
        )
        assert isinstance(result, ConsensusShift)
