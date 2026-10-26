from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.domain.asset import (
    REFERENCE_MAX_AGE_HOURS,
    Asset,
    PriceDelta,
    is_near_threshold,
    is_reference_stale,
)
from src.domain.value_objects import Money, Ratio, Threshold


class TestPriceDelta:
    def test_calculates_negative_delta_when_price_drops(self) -> None:
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("95.0")))
        # Pole `fraction` trzyma UŁAMEK (-0.05), nie procent (-5.0).
        assert delta.fraction == Decimal("-0.05")

    def test_calculates_positive_delta_when_price_rises(self) -> None:
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert delta.fraction == Decimal("0.10")

    def test_returns_zero_when_prices_equal(self) -> None:
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("100.0")))
        assert delta.fraction == Decimal("0")

    def test_returns_zero_when_previous_price_is_zero(self) -> None:
        # Guard przed dzieleniem przez zero (np. brak danych historycznych)
        delta = PriceDelta.calculate(Money(Decimal("0")), Money(Decimal("100.0")))
        assert delta.fraction == Decimal("0")

    def test_percent_property_is_fraction_times_100(self) -> None:
        # Audyt z czerwca 2026: `percent` to wygodny przelicznik fraction*100. Ułamek
        # 0.10 → 10.0 procenta. To naprawia 100x pułapkę nazewnictwa.
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert delta.percent == Decimal("10.0")

    def test_percentage_alias_is_removed(self) -> None:
        # ARCH-14: alias `percentage` zwracał ułamek mimo nazwy (100x pułapka).
        # Ułamek to `fraction`, procent to `percent` — trzeciej drogi nie ma.
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert not hasattr(delta, "percentage")


class TestAssetEvaluateVolatility:
    def test_triggers_analysis_when_price_drops_below_threshold(self) -> None:
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("95.0")))  # -5%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_triggers_analysis_when_price_rises_above_threshold(self) -> None:
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("105.0")))  # +5%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_ignores_minor_fluctuations(self) -> None:
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("99.0")))  # -1%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is False

    def test_triggers_at_exact_threshold(self) -> None:
        # Próg jest inkluzywny (>= threshold)
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("98.0")))  # -2%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_gates_on_fraction_not_percent(self) -> None:
        # Audyt z czerwca 2026: bramka volatility porównuje UŁAMEK (0.05) z progiem
        # (0.02), nie procent (5.0). Gdyby porównywała percent, próg 0.02
        # zawsze przepuszczałby ruch — co byłoby 100x błędem.
        asset = Asset(symbol="AAPL")
        delta = PriceDelta(Decimal("0.05"))  # ułamek = 5%
        assert delta.fraction == Decimal("0.05")
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True


class TestStaleReference:
    """FIN-14: referencja ceny sprzed pauzy nie jest „poprzednim cyklem”.

    Ruch liczony od snapshotu sprzed tygodni to skumulowana zmiana wielu dni —
    otworzyłby bramkę dla wszystkich symboli naraz. Decyzja 2026-10-19:
    re-baseline (cykl zapisuje nowy punkt odniesienia bez analizy)."""

    _NOW = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def test_max_age_is_frozen_at_96_hours(self) -> None:
        # Decyzja właściciela (FIN-14): zmiana progu ma świadomie łamać ten test.
        assert REFERENCE_MAX_AGE_HOURS == 96

    def test_reference_from_30_days_ago_is_stale(self) -> None:
        assert is_reference_stale(self._NOW - timedelta(days=30), self._NOW) is True

    def test_reference_from_previous_cycle_is_fresh(self) -> None:
        assert is_reference_stale(self._NOW - timedelta(hours=24), self._NOW) is False

    def test_weekend_gap_is_still_fresh(self) -> None:
        # Piątek wieczór → poniedziałek rano: ~60 h, to nadal poprzedni cykl.
        assert is_reference_stale(self._NOW - timedelta(hours=60), self._NOW) is False

    def test_exactly_max_age_is_fresh_and_one_minute_more_is_stale(self) -> None:
        limit = timedelta(hours=REFERENCE_MAX_AGE_HOURS)
        assert is_reference_stale(self._NOW - limit, self._NOW) is False
        assert is_reference_stale(self._NOW - limit - timedelta(minutes=1), self._NOW) is True

    def test_missing_reference_is_not_stale(self) -> None:
        # Cold-start (brak snapshotu) ma własną ścieżkę — reguła go nie dotyka.
        assert is_reference_stale(None, self._NOW) is False

    def test_naive_reference_is_read_as_utc(self) -> None:
        naive = (self._NOW - timedelta(days=30)).replace(tzinfo=None)
        assert is_reference_stale(naive, self._NOW) is True

    def test_stale_reference_closes_the_gate_despite_a_big_move(self) -> None:
        asset = Asset(symbol="AAPL")
        delta = PriceDelta(Decimal("-0.10"))
        threshold = Threshold(Decimal("0.02"))

        assert asset.evaluate_volatility(delta, threshold) is True
        assert asset.evaluate_volatility(delta, threshold, reference_stale=True) is False


class TestEffectiveThreshold:
    """FIN-08: jedna reguła progu bramki w domenie — wybór progu klasy aktywa,
    potem iloczyn mnożników reżimu i earnings, nigdy poniżej progu bazowego."""

    def test_stock_without_multipliers_keeps_the_base(self) -> None:
        from src.domain.asset import effective_threshold
        from src.domain.value_objects import AssetType

        base = Threshold(Decimal("0.02"))

        assert effective_threshold(base, AssetType.STOCK, Threshold(Decimal("0.05"))) is base

    def test_crypto_takes_its_override(self) -> None:
        from src.domain.asset import effective_threshold
        from src.domain.value_objects import AssetType

        chosen = effective_threshold(
            Threshold(Decimal("0.02")), AssetType.CRYPTO, Threshold(Decimal("0.05"))
        )

        assert chosen.value == Decimal("0.05")

    def test_crypto_without_override_falls_back_to_the_base(self) -> None:
        from src.domain.asset import effective_threshold
        from src.domain.value_objects import AssetType

        chosen = effective_threshold(Threshold(Decimal("0.02")), AssetType.CRYPTO, None)

        assert chosen.value == Decimal("0.02")

    def test_regime_and_earnings_multipliers_compound_on_the_chosen_threshold(self) -> None:
        from src.domain.asset import effective_threshold
        from src.domain.value_objects import AssetType

        chosen = effective_threshold(
            Threshold(Decimal("0.02")),
            AssetType.CRYPTO,
            Threshold(Decimal("0.05")),
            regime_multiplier=2.0,
            earnings_multiplier=1.5,
        )

        assert chosen.value == Decimal("0.150")

    def test_a_multiplier_below_one_never_loosens_the_gate(self) -> None:
        from src.domain.asset import effective_threshold

        base = Threshold(Decimal("0.02"))

        assert effective_threshold(base, None, None, regime_multiplier=0.5).value == Decimal("0.02")


class TestNearThreshold:
    """MAIL-18: ruch od 75% progu efektywnego, ale pod nim, jest „blisko progu”."""

    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (Decimal("0.015"), True),
            (Decimal("-0.019"), True),
            (Decimal("0.0149"), False),
            (Decimal("0.02"), False),
            (Decimal("0"), False),
        ],
    )
    def test_move_from_three_quarters_of_the_threshold_is_near(
        self, delta: Decimal, expected: bool
    ) -> None:
        assert is_near_threshold(Ratio(delta), Ratio(Decimal("0.02"))) is expected

    def test_zero_threshold_is_never_near(self) -> None:
        assert is_near_threshold(Ratio(Decimal("0.01")), Ratio(Decimal("0"))) is False
