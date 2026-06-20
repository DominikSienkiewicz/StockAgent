from decimal import Decimal

from src.domain.asset import Asset, PriceDelta
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
