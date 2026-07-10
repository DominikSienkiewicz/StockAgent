from datetime import date
from decimal import Decimal

from src.domain.asset import Asset, PriceDelta
from src.domain.shock import (
    CRYPTO_SHOCK_THRESHOLD,
    SHOCK_THRESHOLD_MULTIPLIER,
    SNAPSHOT_MAX_AGE_HOURS,
    ShockAlert,
    ShockDirection,
    detect_shock,
    should_emit,
)
from src.domain.value_objects import AssetType, Threshold

# Bramka volatility Fast Loopa (typowy próg 2%). Szok = 2× tej wartości = 4%.
FAST_LOOP_THRESHOLD = Threshold(Decimal("0.02"))
FRESH_SNAPSHOT_HOURS = 1.0


class TestDetectShock:
    def test_emits_alert_when_equity_delta_exceeds_double_threshold(self):
        # 5% spadek akcji przekracza próg szoku (2× 2% = 4%).
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.05"))
        alert = detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert == ShockAlert(
            symbol="NVDA", delta=delta, direction=ShockDirection.DOWN
        )

    def test_no_alert_when_equity_delta_below_shock_threshold(self):
        # 3% mieści się w bramce volatility × 2 (4%) → to nie jest szok.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("0.03"))
        assert (
            detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
            is None
        )

    def test_upward_shock_reports_up_direction(self):
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("0.06"))
        alert = detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert is not None
        assert alert.direction is ShockDirection.UP

    def test_crypto_needs_higher_threshold_than_equity_for_same_delta(self):
        # Ta sama delta 4%: akcje dają alert, BTC nie (krypto rusza się natywnie
        # 3–5% dziennie, więc ma osobny, wyższy próg).
        delta = PriceDelta(Decimal("0.04"))
        stock = Asset("NVDA", AssetType.STOCK)
        crypto = Asset("BTC", AssetType.CRYPTO)

        stock_alert = detect_shock(
            stock, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS
        )
        crypto_alert = detect_shock(
            crypto, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS
        )

        assert stock_alert is not None
        assert crypto_alert is None

    def test_crypto_emits_alert_above_its_own_threshold(self):
        # 8% na BTC przekracza osobny próg krypto → alert.
        crypto = Asset("BTC", AssetType.CRYPTO)
        delta = PriceDelta(Decimal("-0.08"))
        alert = detect_shock(crypto, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert is not None
        assert alert.symbol == "BTC"
        assert alert.direction is ShockDirection.DOWN

    def test_stale_snapshot_suppresses_alert_despite_large_delta(self):
        # Stęchły punkt odniesienia (48h) daje fałszywy alarm — brak alertu
        # mimo dużej delty 10%.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.10"))
        assert detect_shock(asset, delta, FAST_LOOP_THRESHOLD, 48.0) is None

    def test_snapshot_exactly_at_limit_is_still_fresh(self):
        # Wiek dokładnie na granicy nie jest jeszcze stęchły — alert dozwolony.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.10"))
        alert = detect_shock(
            asset, delta, FAST_LOOP_THRESHOLD, SNAPSHOT_MAX_AGE_HOURS
        )
        assert alert is not None

    def test_zero_delta_never_shocks(self):
        asset = Asset("NVDA", AssetType.STOCK)
        assert (
            detect_shock(
                asset, PriceDelta(Decimal("0")), FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS
            )
            is None
        )


class TestShouldEmit:
    def test_first_alert_of_the_day_is_emitted(self):
        assert should_emit("NVDA", date(2026, 7, 10), set()) is True

    def test_second_alert_same_symbol_same_day_is_suppressed(self):
        # Twardy debounce: drugi szok tego samego symbolu tego samego dnia
        # NIE jest emitowany (spam uczy ignorować kanał).
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("NVDA", date(2026, 7, 10), already_sent) is False

    def test_same_symbol_next_day_is_emitted_again(self):
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("NVDA", date(2026, 7, 11), already_sent) is True

    def test_other_symbol_same_day_is_independent(self):
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("BTC", date(2026, 7, 10), already_sent) is True


class TestConstants:
    def test_shock_multiplier_is_double(self):
        assert Decimal("2") == SHOCK_THRESHOLD_MULTIPLIER

    def test_crypto_threshold_is_higher_than_doubled_typical_equity_gate(self):
        # Krypto musi mieć wyższy próg niż podwojona typowa bramka equity (4%).
        assert Decimal("0.02") * SHOCK_THRESHOLD_MULTIPLIER < CRYPTO_SHOCK_THRESHOLD
