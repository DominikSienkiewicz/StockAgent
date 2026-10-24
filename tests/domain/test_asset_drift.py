"""ALR-10 — bramka volatility widzi powolny zjazd: dryf od ostatniej predykcji.

10 cykli po -1,9% to -17,5% bez żadnej analizy, bo każdy krok jest pod progiem.
Dryf mierzy ruch od ceny ostatniej predykcji (albo od początku okna
`DRIFT_WINDOW_DAYS`) i otwiera bramkę tym samym progiem co ruch cyklu."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.domain.asset import DRIFT_WINDOW_DAYS, Asset, PriceDelta, drift_reference
from src.domain.value_objects import Threshold

NOW = datetime(2026, 10, 21, 13, 30, tzinfo=UTC)
GATE = Threshold(Decimal("0.02"))
CYCLE = PriceDelta(Decimal("-0.019"))


class TestEvaluateVolatilityWithDrift:
    def test_cumulative_drift_opens_the_gate_below_the_cycle_move(self) -> None:
        assert Asset("AAPL").evaluate_volatility(
            CYCLE, GATE, drift=PriceDelta(Decimal("-0.175"))
        ) is True

    def test_small_drift_keeps_the_gate_closed(self) -> None:
        assert Asset("AAPL").evaluate_volatility(
            CYCLE, GATE, drift=PriceDelta(Decimal("0.0199"))
        ) is False

    def test_without_drift_the_gate_is_the_cycle_move(self) -> None:
        assert Asset("AAPL").evaluate_volatility(CYCLE, GATE) is False
        assert Asset("AAPL").evaluate_volatility(PriceDelta(Decimal("0.02")), GATE) is True

    def test_stale_reference_closes_the_gate_despite_the_drift(self) -> None:
        # FIN-14: po pauzie najpierw re-baseline — dryf nie omija tej reguły.
        assert Asset("AAPL").evaluate_volatility(
            CYCLE, GATE, reference_stale=True, drift=PriceDelta(Decimal("-0.3"))
        ) is False


class TestDriftReference:
    @staticmethod
    def _days_ago(days: float) -> datetime:
        return NOW - timedelta(days=days)

    def test_window_is_ten_days(self) -> None:
        assert DRIFT_WINDOW_DAYS == 10

    def test_prediction_inside_the_window_is_the_reference(self) -> None:
        history = [(self._days_ago(9), Decimal("130")), (self._days_ago(1), Decimal("101"))]

        reference = drift_reference((self._days_ago(3), Decimal("118.9")), history, NOW)

        assert reference == Decimal("118.9")

    def test_prediction_older_than_the_window_falls_back_to_the_window_start(self) -> None:
        # Predykcja sprzed pauzy to nie punkt odniesienia — inaczej wznowienie
        # otworzyłoby bramkę każdemu symbolowi, który ruszył się od tamtej pory.
        history = [(self._days_ago(9.5), Decimal("125")), (self._days_ago(1), Decimal("101"))]

        reference = drift_reference((self._days_ago(30), Decimal("80")), history, NOW)

        assert reference == Decimal("125")

    def test_without_a_prediction_the_first_price_of_the_window_is_the_reference(self) -> None:
        history = [(self._days_ago(8), Decimal("120")), (self._days_ago(2), Decimal("105"))]

        assert drift_reference(None, history, NOW) == Decimal("120")

    def test_prices_before_the_window_are_ignored(self) -> None:
        history = [(self._days_ago(15), Decimal("150")), (self._days_ago(4), Decimal("110"))]

        assert drift_reference(None, history, NOW) == Decimal("110")

    def test_nothing_inside_the_window_gives_no_reference(self) -> None:
        assert drift_reference((self._days_ago(20), Decimal("90")), [], NOW) is None

    def test_non_positive_prices_give_no_reference(self) -> None:
        history = [(self._days_ago(5), Decimal("0"))]

        assert drift_reference((self._days_ago(2), Decimal("0")), history, NOW) is None

    def test_naive_times_are_read_as_utc(self) -> None:
        naive_prediction = (NOW.replace(tzinfo=None) - timedelta(days=2), Decimal("99"))

        assert drift_reference(naive_prediction, [], NOW) == Decimal("99")
