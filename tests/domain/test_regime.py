"""Testy domeny detektora reżimu rynkowego.

Reżim agreguje najgorszy alert makro (`MacroAlertLevel` z macro_risk /
drawdown, `MacroStressLevel` z polish_macro — ta sama skala NORMAL/
ELEVATED/CRITICAL) oraz liczbę/severity sygnałów risk-off i mapuje to na
jeden z trzech stanów: RISK_ON / NEUTRAL / RISK_OFF.

Niezmiennik FinOps: mnożnik progu zwracany przez `threshold_multiplier`
NIGDY nie schodzi poniżej 1.0 — reżim może tylko ZACIEŚNIĆ bramkę
volatility (podnieść próg), nigdy jej poluzować.
"""

from __future__ import annotations

import pytest

from src.domain.macro_risk import MacroAlertLevel
from src.domain.polish_macro import MacroStressLevel
from src.domain.regime import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
    apply_hysteresis,
)


class TestClassify:
    def test_clean_inputs_are_risk_on(self) -> None:
        detector = RegimeDetector()
        regime = detector.classify(
            MacroAlertLevel.NORMAL,
            [MacroAlertLevel.NORMAL, MacroAlertLevel.NORMAL],
        )
        assert regime is MarketRegime.RISK_ON

    def test_critical_macro_alert_is_risk_off(self) -> None:
        detector = RegimeDetector()
        regime = detector.classify(
            MacroAlertLevel.CRITICAL,
            [MacroAlertLevel.NORMAL],
        )
        assert regime is MarketRegime.RISK_OFF

    def test_many_elevated_risk_signals_is_risk_off(self) -> None:
        detector = RegimeDetector()
        # Dwa lub więcej ELEVATED bez CRITICAL makro → też risk-off.
        regime = detector.classify(
            MacroAlertLevel.NORMAL,
            [MacroAlertLevel.ELEVATED, MacroAlertLevel.ELEVATED],
        )
        assert regime is MarketRegime.RISK_OFF

    def test_single_elevated_is_neutral(self) -> None:
        detector = RegimeDetector()
        regime = detector.classify(
            MacroAlertLevel.ELEVATED,
            [MacroAlertLevel.NORMAL],
        )
        assert regime is MarketRegime.NEUTRAL

    def test_one_elevated_risk_signal_is_neutral(self) -> None:
        detector = RegimeDetector()
        regime = detector.classify(
            MacroAlertLevel.NORMAL,
            [MacroAlertLevel.ELEVATED, MacroAlertLevel.NORMAL],
        )
        assert regime is MarketRegime.NEUTRAL

    def test_single_critical_risk_signal_is_risk_off(self) -> None:
        detector = RegimeDetector()
        # Pojedynczy CRITICAL wśród sygnałów risk-off wystarcza do risk-off.
        regime = detector.classify(
            MacroAlertLevel.NORMAL,
            [MacroAlertLevel.CRITICAL],
        )
        assert regime is MarketRegime.RISK_OFF

    def test_empty_signals_with_normal_macro_is_risk_on(self) -> None:
        detector = RegimeDetector()
        assert detector.classify(MacroAlertLevel.NORMAL, []) is MarketRegime.RISK_ON

    def test_none_inputs_are_neutral(self) -> None:
        detector = RegimeDetector()
        # Brak danych makro = brak informacji → neutralnie, nie risk-on.
        assert detector.classify(None, None) is MarketRegime.NEUTRAL

    def test_none_macro_with_clean_signals_is_neutral(self) -> None:
        detector = RegimeDetector()
        assert (
            detector.classify(None, [MacroAlertLevel.NORMAL])
            is MarketRegime.NEUTRAL
        )

    def test_accepts_polish_stress_level(self) -> None:
        detector = RegimeDetector()
        # Skala MacroStressLevel jest wymienna z MacroAlertLevel.
        regime = detector.classify(
            MacroStressLevel.CRITICAL,
            [MacroStressLevel.NORMAL],
        )
        assert regime is MarketRegime.RISK_OFF


class TestThresholdMultiplier:
    def test_risk_off_returns_max_multiplier(self) -> None:
        detector = RegimeDetector()
        assert detector.threshold_multiplier(MarketRegime.RISK_OFF, 2.0) == pytest.approx(2.0)

    def test_neutral_returns_one(self) -> None:
        detector = RegimeDetector()
        assert detector.threshold_multiplier(MarketRegime.NEUTRAL, 2.0) == pytest.approx(1.0)

    def test_risk_on_returns_one(self) -> None:
        detector = RegimeDetector()
        assert detector.threshold_multiplier(MarketRegime.RISK_ON, 2.0) == pytest.approx(1.0)

    def test_never_below_one_even_with_sub_one_max(self) -> None:
        detector = RegimeDetector()
        # Niezmiennik: nawet absurdalny max < 1.0 nie poluzuje bramki.
        assert detector.threshold_multiplier(MarketRegime.RISK_OFF, 0.5) >= 1.0
        assert detector.threshold_multiplier(MarketRegime.NEUTRAL, 0.5) >= 1.0
        assert detector.threshold_multiplier(MarketRegime.RISK_ON, 0.5) >= 1.0

    def test_risk_off_clamped_to_one_when_max_below_one(self) -> None:
        detector = RegimeDetector()
        assert detector.threshold_multiplier(MarketRegime.RISK_OFF, 0.5) == pytest.approx(1.0)

    @pytest.mark.parametrize("max_mult", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_ordering_risk_off_ge_neutral_ge_risk_on(
        self, max_mult: float
    ) -> None:
        detector = RegimeDetector()
        off = detector.threshold_multiplier(MarketRegime.RISK_OFF, max_mult)
        neutral = detector.threshold_multiplier(MarketRegime.NEUTRAL, max_mult)
        on = detector.threshold_multiplier(MarketRegime.RISK_ON, max_mult)
        assert off >= neutral >= on
        assert on >= 1.0

    def test_risk_off_respects_max_multiplier_ceiling(self) -> None:
        detector = RegimeDetector()
        # Mnożnik nigdy nie przekracza max_multiplier (clamp górny).
        assert detector.threshold_multiplier(MarketRegime.RISK_OFF, 3.0) == pytest.approx(3.0)


class TestLabel:
    def test_labels_are_distinct_polish_strings(self) -> None:
        detector = RegimeDetector()
        labels = {
            detector.label(MarketRegime.RISK_ON),
            detector.label(MarketRegime.NEUTRAL),
            detector.label(MarketRegime.RISK_OFF),
        }
        assert len(labels) == 3
        for label in labels:
            assert isinstance(label, str)
            assert label


# ---------------------------------------------------------------------------
# ALR-03 — histereza reżimu: wejście w RISK_OFF natychmiast, wyjście dopiero
# po N kolejnych surowych odczytach poza RISK_OFF. `classify()` jest bezstanowe,
# więc sygnał oscylujący przy progu przełączałby mnożnik progu co cykl.
# ---------------------------------------------------------------------------


def _walk(
    start: MarketRegime, raws: list[MarketRegime], exit_confirmations: int = 2
) -> list[RegimeState]:
    """Przepuszcza kolejne surowe odczyty przez histerezę od stanu `start`."""
    state = RegimeState(regime=start)
    states: list[RegimeState] = []
    for raw in raws:
        state = apply_hysteresis(state, raw, exit_confirmations)
        states.append(state)
    return states


class TestApplyHysteresis:
    def test_oscillating_signal_keeps_risk_off(self) -> None:
        states = _walk(
            MarketRegime.RISK_OFF,
            [MarketRegime.RISK_ON, MarketRegime.RISK_OFF, MarketRegime.RISK_ON],
        )

        assert [s.regime for s in states] == [MarketRegime.RISK_OFF] * 3

    def test_two_consecutive_readings_exit_on_the_second(self) -> None:
        first, second = _walk(
            MarketRegime.RISK_OFF, [MarketRegime.RISK_ON, MarketRegime.RISK_ON]
        )

        assert first == RegimeState(
            regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1
        )
        assert second == RegimeState(regime=MarketRegime.RISK_ON)

    def test_entering_risk_off_is_immediate(self) -> None:
        state = apply_hysteresis(
            RegimeState(regime=MarketRegime.RISK_ON), MarketRegime.RISK_OFF
        )

        assert state == RegimeState(regime=MarketRegime.RISK_OFF)

    @pytest.mark.parametrize("raw", list(MarketRegime))
    def test_no_history_returns_the_raw_regime(self, raw: MarketRegime) -> None:
        assert apply_hysteresis(None, raw) == RegimeState(regime=raw)

    def test_a_risk_off_reading_resets_the_streak(self) -> None:
        *_, last = _walk(
            MarketRegime.RISK_OFF,
            [MarketRegime.RISK_ON, MarketRegime.RISK_OFF, MarketRegime.RISK_ON],
        )

        assert last.streak == 1

    def test_neutral_and_risk_on_both_count_towards_the_exit(self) -> None:
        # Kierunek = wyjście z RISK_OFF; reżim po wyjściu to ostatni surowy odczyt.
        first, second = _walk(
            MarketRegime.RISK_OFF, [MarketRegime.NEUTRAL, MarketRegime.RISK_ON]
        )

        assert first.regime is MarketRegime.RISK_OFF
        assert second == RegimeState(regime=MarketRegime.RISK_ON)

    def test_moves_outside_risk_off_are_not_delayed(self) -> None:
        state = apply_hysteresis(
            RegimeState(regime=MarketRegime.NEUTRAL), MarketRegime.RISK_ON
        )

        assert state == RegimeState(regime=MarketRegime.RISK_ON)

    def test_three_confirmations_need_three_readings(self) -> None:
        states = _walk(MarketRegime.RISK_OFF, [MarketRegime.RISK_ON] * 3, 3)

        assert [s.regime for s in states] == [
            MarketRegime.RISK_OFF,
            MarketRegime.RISK_OFF,
            MarketRegime.RISK_ON,
        ]

    @pytest.mark.parametrize("confirmations", [1, 0, -3])
    def test_one_or_fewer_confirmations_exit_at_once(self, confirmations: int) -> None:
        # N ≤ 1 = zachowanie sprzed ALR-03 (fail-safe na zły config, bez wyjątku).
        state = apply_hysteresis(
            RegimeState(regime=MarketRegime.RISK_OFF), MarketRegime.RISK_ON, confirmations
        )

        assert state == RegimeState(regime=MarketRegime.RISK_ON)


_PREVIOUS_STATES: list[RegimeState | None] = [
    None,
    *(RegimeState(regime=regime) for regime in MarketRegime),
    RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1),
    RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.NEUTRAL, streak=5),
]


class TestHysteresisFinOps:
    """Niezmiennik FinOps: histereza może tylko PRZEDŁUŻYĆ RISK_OFF (wyższy próg),
    nigdy obniżyć progu względem surowego reżimu z `classify()`."""

    @pytest.mark.parametrize("previous", _PREVIOUS_STATES)
    @pytest.mark.parametrize("raw", list(MarketRegime))
    @pytest.mark.parametrize("confirmations", [1, 2, 3])
    def test_never_lowers_the_threshold_below_the_raw_regime(
        self,
        previous: RegimeState | None,
        raw: MarketRegime,
        confirmations: int,
    ) -> None:
        detector = RegimeDetector()

        effective = apply_hysteresis(previous, raw, confirmations).regime

        assert detector.threshold_multiplier(
            effective, 2.0
        ) >= detector.threshold_multiplier(raw, 2.0)


class TestMacroSeriesVotes:
    """DAT-22 — percentyl VIX z roku i zmiana spreadu HY to stabilniejsze zmienne
    stanu niż dzienna zmiana VIXY; wchodzą do `RegimeDetector` jako głosy."""

    def test_vix_percentile_above_90_maps_to_critical_input(self) -> None:
        from src.domain.macro_series import vix_percentile_alert_level

        level = vix_percentile_alert_level(92.0)

        assert level is MacroAlertLevel.CRITICAL
        assert RegimeDetector().classify(MacroAlertLevel.NORMAL, [level]) is (
            MarketRegime.RISK_OFF
        )

    @pytest.mark.parametrize(
        ("percentile", "expected"),
        [
            (90.0, MacroAlertLevel.ELEVATED),
            (75.0, MacroAlertLevel.ELEVATED),
            (74.9, MacroAlertLevel.NORMAL),
        ],
    )
    def test_vix_percentile_bands(self, percentile: float, expected: MacroAlertLevel) -> None:
        from src.domain.macro_series import vix_percentile_alert_level

        assert vix_percentile_alert_level(percentile) is expected

    def test_a_single_elevated_macro_vote_only_moves_to_neutral(self) -> None:
        from src.domain.macro_series import credit_spread_alert_level

        level = credit_spread_alert_level(0.30)

        assert level is MacroAlertLevel.ELEVATED
        assert RegimeDetector().classify(MacroAlertLevel.NORMAL, [level]) is (
            MarketRegime.NEUTRAL
        )
