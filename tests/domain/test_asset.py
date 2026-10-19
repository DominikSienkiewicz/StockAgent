from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.domain.asset import (
    REFERENCE_MAX_AGE_HOURS,
    Asset,
    PriceDelta,
    is_reference_stale,
)
from src.domain.value_objects import Money, Threshold


class TestPriceDelta:
    def test_calculates_negative_delta_when_price_drops(self):
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("95.0")))
        # Pole `fraction` trzyma UŁAMEK (-0.05), nie procent (-5.0).
        assert delta.fraction == Decimal("-0.05")

    def test_calculates_positive_delta_when_price_rises(self):
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert delta.fraction == Decimal("0.10")

    def test_returns_zero_when_prices_equal(self):
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("100.0")))
        assert delta.fraction == Decimal("0")

    def test_returns_zero_when_previous_price_is_zero(self):
        # Guard przed dzieleniem przez zero (np. brak danych historycznych)
        delta = PriceDelta.calculate(Money(Decimal("0")), Money(Decimal("100.0")))
        assert delta.fraction == Decimal("0")

    def test_percent_property_is_fraction_times_100(self):
        # Finding #33: `percent` to wygodny przelicznik fraction*100. Ułamek
        # 0.10 → 10.0 procenta. To naprawia 100x pułapkę nazewnictwa.
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert delta.percent == Decimal("10.0")

    def test_percentage_alias_still_reads_fraction(self):
        # Deprecated alias `percentage` musi nadal zwracać UŁAMEK (nie procent),
        # bo zewnętrzny czytelnik (agent_graph.py) czyta `delta.percentage`
        # i traktuje go jak ułamek. Zmiana wartości złamałaby ten kontrakt.
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("110.0")))
        assert delta.percentage == Decimal("0.10")
        assert delta.percentage == delta.fraction


class TestAssetEvaluateVolatility:
    def test_triggers_analysis_when_price_drops_below_threshold(self):
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("95.0")))  # -5%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_triggers_analysis_when_price_rises_above_threshold(self):
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("105.0")))  # +5%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_ignores_minor_fluctuations(self):
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("99.0")))  # -1%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is False

    def test_triggers_at_exact_threshold(self):
        # Próg jest inkluzywny (>= threshold)
        asset = Asset(symbol="AAPL")
        delta = PriceDelta.calculate(Money(Decimal("100.0")), Money(Decimal("98.0")))  # -2%
        assert asset.evaluate_volatility(delta, Threshold(Decimal("0.02"))) is True

    def test_gates_on_fraction_not_percent(self):
        # Finding #33: bramka volatility porównuje UŁAMEK (0.05) z progiem
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

    def test_max_age_is_frozen_at_96_hours(self):
        # Decyzja właściciela (FIN-14): zmiana progu ma świadomie łamać ten test.
        assert REFERENCE_MAX_AGE_HOURS == 96

    def test_reference_from_30_days_ago_is_stale(self):
        assert is_reference_stale(self._NOW - timedelta(days=30), self._NOW) is True

    def test_reference_from_previous_cycle_is_fresh(self):
        assert is_reference_stale(self._NOW - timedelta(hours=24), self._NOW) is False

    def test_weekend_gap_is_still_fresh(self):
        # Piątek wieczór → poniedziałek rano: ~60 h, to nadal poprzedni cykl.
        assert is_reference_stale(self._NOW - timedelta(hours=60), self._NOW) is False

    def test_exactly_max_age_is_fresh_and_one_minute_more_is_stale(self):
        limit = timedelta(hours=REFERENCE_MAX_AGE_HOURS)
        assert is_reference_stale(self._NOW - limit, self._NOW) is False
        assert is_reference_stale(self._NOW - limit - timedelta(minutes=1), self._NOW) is True

    def test_missing_reference_is_not_stale(self):
        # Cold-start (brak snapshotu) ma własną ścieżkę — reguła go nie dotyka.
        assert is_reference_stale(None, self._NOW) is False

    def test_naive_reference_is_read_as_utc(self):
        naive = (self._NOW - timedelta(days=30)).replace(tzinfo=None)
        assert is_reference_stale(naive, self._NOW) is True

    def test_stale_reference_closes_the_gate_despite_a_big_move(self):
        asset = Asset(symbol="AAPL")
        delta = PriceDelta(Decimal("-0.10"))
        threshold = Threshold(Decimal("0.02"))

        assert asset.evaluate_volatility(delta, threshold) is True
        assert asset.evaluate_volatility(delta, threshold, reference_stale=True) is False
