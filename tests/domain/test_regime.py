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
from src.domain.regime import MarketRegime, RegimeDetector


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
