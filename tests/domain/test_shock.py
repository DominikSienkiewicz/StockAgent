from datetime import UTC, date, datetime
from decimal import Decimal

from src.domain.asset import Asset, PriceDelta
from src.domain.shock import (
    CRYPTO_SHOCK_THRESHOLD,
    MAX_SHOCK_ALERTS_PER_DAY,
    PORTFOLIO_ALERT_SYMBOL,
    SHOCK_THRESHOLD_MULTIPLIER,
    SNAPSHOT_MAX_AGE_HOURS,
    SentShockLevel,
    ShockAlert,
    ShockDirection,
    classify_shock,
    detect_portfolio_shock,
    detect_shock,
    is_reference_stale,
    next_shock_level,
    should_emit,
    should_escalate,
)
from src.domain.value_objects import AssetType, Threshold

# Bramka volatility Fast Loopa (typowy próg 2%). Szok = 2× tej wartości = 4%.
FAST_LOOP_THRESHOLD = Threshold(Decimal("0.02"))
FRESH_SNAPSHOT_HOURS = 1.0


class TestDetectShock:
    def test_emits_alert_when_equity_delta_exceeds_double_threshold(self) -> None:
        # 5% spadek akcji przekracza próg szoku (2× 2% = 4%).
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.05"))
        alert = detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert == ShockAlert(
            symbol="NVDA", delta=delta, direction=ShockDirection.DOWN
        )

    def test_no_alert_when_equity_delta_below_shock_threshold(self) -> None:
        # 3% mieści się w bramce volatility × 2 (4%) → to nie jest szok.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("0.03"))
        assert (
            detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
            is None
        )

    def test_upward_shock_reports_up_direction(self) -> None:
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("0.06"))
        alert = detect_shock(asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert is not None
        assert alert.direction is ShockDirection.UP

    def test_crypto_needs_higher_threshold_than_equity_for_same_delta(self) -> None:
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

    def test_crypto_emits_alert_above_its_own_threshold(self) -> None:
        # 8% na BTC przekracza osobny próg krypto → alert.
        crypto = Asset("BTC", AssetType.CRYPTO)
        delta = PriceDelta(Decimal("-0.08"))
        alert = detect_shock(crypto, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS)
        assert alert is not None
        assert alert.symbol == "BTC"
        assert alert.direction is ShockDirection.DOWN

    def test_stale_snapshot_suppresses_alert_despite_large_delta(self) -> None:
        # Stęchły punkt odniesienia (48h) daje fałszywy alarm — brak alertu
        # mimo dużej delty 10%.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.10"))
        assert detect_shock(asset, delta, FAST_LOOP_THRESHOLD, 48.0) is None

    def test_snapshot_exactly_at_limit_is_still_fresh(self) -> None:
        # Wiek dokładnie na granicy nie jest jeszcze stęchły — alert dozwolony.
        asset = Asset("NVDA", AssetType.STOCK)
        delta = PriceDelta(Decimal("-0.10"))
        alert = detect_shock(
            asset, delta, FAST_LOOP_THRESHOLD, SNAPSHOT_MAX_AGE_HOURS
        )
        assert alert is not None

    def test_zero_delta_never_shocks(self) -> None:
        asset = Asset("NVDA", AssetType.STOCK)
        assert (
            detect_shock(
                asset, PriceDelta(Decimal("0")), FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS
            )
            is None
        )


class TestReferenceStaleness:
    def test_reference_older_than_max_age_is_stale(self) -> None:
        # Ta sama granica co w `detect_shock`: dokładnie limit jest jeszcze
        # świeży, wszystko powyżej to stęchły punkt odniesienia.
        assert is_reference_stale(30.01) is True
        assert is_reference_stale(SNAPSHOT_MAX_AGE_HOURS) is False

    def test_fresh_reference_is_not_stale(self) -> None:
        assert is_reference_stale(FRESH_SNAPSHOT_HOURS) is False


class TestShouldEmit:
    def test_first_alert_of_the_day_is_emitted(self) -> None:
        assert should_emit("NVDA", date(2026, 7, 10), set()) is True

    def test_second_alert_same_symbol_same_day_is_suppressed(self) -> None:
        # Twardy debounce: drugi szok tego samego symbolu tego samego dnia
        # NIE jest emitowany (spam uczy ignorować kanał).
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("NVDA", date(2026, 7, 10), already_sent) is False

    def test_same_symbol_next_day_is_emitted_again(self) -> None:
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("NVDA", date(2026, 7, 11), already_sent) is True

    def test_other_symbol_same_day_is_independent(self) -> None:
        already_sent = {("NVDA", date(2026, 7, 10))}
        assert should_emit("BTC", date(2026, 7, 10), already_sent) is True


class TestShouldEscalate:
    """ALR-16: drabina eskalacji — drugi alert dnia tylko przy ruchu ≥ 2× progu szoku
    w tym samym kierunku albo przy odwróceniu kierunku ponad próg; najwyżej 2 dziennie."""

    NVDA = Asset("NVDA", AssetType.STOCK)

    def _alert(self, fraction: str) -> ShockAlert:
        alert = classify_shock(self.NVDA, PriceDelta(Decimal(fraction)), FAST_LOOP_THRESHOLD)
        assert alert is not None
        return alert

    def test_escalates_once_when_move_doubles(self) -> None:
        # Poziom 1 poszedł przy -4,5% (próg szoku 4%). -9,5% przekracza 2× progu (8%)
        # → drugi szczebel; -6% to wciąż ten sam szok → nic; po drugim szczeblu
        # trzecia próba nie przechodzi nawet przy -15%.
        first = SentShockLevel(level=1, direction=self._alert("-0.045").direction)

        assert should_escalate(self.NVDA, self._alert("-0.095"), FAST_LOOP_THRESHOLD, [first])
        assert not should_escalate(self.NVDA, self._alert("-0.06"), FAST_LOOP_THRESHOLD, [first])
        second = SentShockLevel(level=2, direction=ShockDirection.DOWN)
        assert not should_escalate(
            self.NVDA, self._alert("-0.15"), FAST_LOOP_THRESHOLD, [first, second]
        )

    def test_reversal_above_threshold_escalates(self) -> None:
        # Rano -5%, po południu +4,5%: odwrócenie kierunku ponad próg to nowa informacja,
        # choć ruch nie sięga 2× progu.
        first = SentShockLevel(level=1, direction=ShockDirection.DOWN)
        assert should_escalate(self.NVDA, self._alert("0.045"), FAST_LOOP_THRESHOLD, [first])

    def test_first_alert_of_the_day_needs_no_escalation(self) -> None:
        assert should_escalate(self.NVDA, self._alert("-0.045"), FAST_LOOP_THRESHOLD, [])

    def test_crypto_escalates_from_its_own_threshold(self) -> None:
        # Krypto: próg szoku 6%, więc drugi szczebel dopiero od 12%.
        btc = Asset("BTC", AssetType.CRYPTO)
        first = [SentShockLevel(level=1, direction=ShockDirection.DOWN)]
        below = classify_shock(btc, PriceDelta(Decimal("-0.10")), FAST_LOOP_THRESHOLD)
        above = classify_shock(btc, PriceDelta(Decimal("-0.12")), FAST_LOOP_THRESHOLD)
        assert below is not None and above is not None
        assert not should_escalate(btc, below, FAST_LOOP_THRESHOLD, first)
        assert should_escalate(btc, above, FAST_LOOP_THRESHOLD, first)

    def test_next_level_follows_the_sent_rungs(self) -> None:
        assert next_shock_level([]) == 1
        assert next_shock_level([SentShockLevel(level=1, direction=ShockDirection.UP)]) == 2
        assert MAX_SHOCK_ALERTS_PER_DAY == 2


class TestConstants:
    def test_shock_multiplier_is_double(self) -> None:
        assert Decimal("2") == SHOCK_THRESHOLD_MULTIPLIER

    def test_crypto_threshold_is_higher_than_doubled_typical_equity_gate(self) -> None:
        # Krypto musi mieć wyższy próg niż podwojona typowa bramka equity (4%).
        assert Decimal("0.02") * SHOCK_THRESHOLD_MULTIPLIER < CRYPTO_SHOCK_THRESHOLD


class TestClassifyShock:
    """ALR-02: sama klasyfikacja ruchu, bez wieku referencji — dzienna zmiana
    z /quote nie ma snapshotu, więc jej świeżość ocenia osobny predykat."""

    def test_equity_move_above_double_gate_is_a_shock(self) -> None:
        from src.domain.shock import classify_shock

        delta = PriceDelta(Decimal("-0.09"))
        alert = classify_shock(Asset("AAPL", AssetType.STOCK), delta, FAST_LOOP_THRESHOLD)
        assert alert == ShockAlert(symbol="AAPL", delta=delta, direction=ShockDirection.DOWN)

    def test_crypto_keeps_its_own_higher_threshold(self) -> None:
        from src.domain.shock import classify_shock

        delta = PriceDelta(Decimal("0.05"))
        assert classify_shock(Asset("BTC", AssetType.CRYPTO), delta, FAST_LOOP_THRESHOLD) is None

    def test_detect_shock_matches_classify_shock_on_a_fresh_reference(self) -> None:
        from src.domain.shock import classify_shock

        asset = Asset("NVDA", AssetType.STOCK)
        for raw in ("-0.10", "-0.04", "0.03", "0.06"):
            delta = PriceDelta(Decimal(raw))
            assert detect_shock(
                asset, delta, FAST_LOOP_THRESHOLD, FRESH_SNAPSHOT_HOURS
            ) == classify_shock(asset, delta, FAST_LOOP_THRESHOLD)


class TestDailyChangeFreshness:
    """ALR-02: `dp` Finnhuba w sobotę to ruch z piątku. Odczyt spoza bieżącego
    dnia UTC (dnia klucza debounce) nie może alarmować — inaczej piątkowy szok
    wróciłby w sobotę jako drugi alert pod nowym kluczem `alert_date`."""

    def test_reading_from_the_same_utc_day_is_fresh(self) -> None:
        from datetime import UTC, datetime

        from src.domain.shock import is_daily_change_fresh

        now = datetime(2026, 10, 16, 19, 7, tzinfo=UTC)
        assert is_daily_change_fresh(datetime(2026, 10, 16, 19, 0, tzinfo=UTC), now) is True

    def test_saturday_with_friday_close_is_stale(self) -> None:
        from datetime import UTC, datetime

        from src.domain.shock import is_daily_change_fresh

        friday_close = datetime(2026, 10, 16, 20, 0, tzinfo=UTC)
        saturday = datetime(2026, 10, 17, 9, 7, tzinfo=UTC)
        assert is_daily_change_fresh(friday_close, saturday) is False

    def test_friday_evening_after_utc_midnight_is_stale(self) -> None:
        # 20:07 ET w piątek to już sobota w UTC: debounce liczy nowy dzień, więc
        # świeżość musi liczyć ten sam dzień — inaczej sesja dałaby dwa alerty.
        from datetime import UTC, datetime

        from src.domain.shock import is_daily_change_fresh

        friday_close = datetime(2026, 10, 16, 20, 0, tzinfo=UTC)
        after_midnight = datetime(2026, 10, 17, 0, 7, tzinfo=UTC)
        assert is_daily_change_fresh(friday_close, after_midnight) is False

    def test_naive_timestamps_are_read_as_utc(self) -> None:
        from datetime import UTC, datetime

        from src.domain.shock import is_daily_change_fresh

        assert is_daily_change_fresh(
            datetime(2026, 10, 16, 23, 0), datetime(2026, 10, 16, 23, 30, tzinfo=UTC)
        ) is True


class TestRollingWindowCooldown:
    """ALR-02, decyzja 2: zmiana 24 h CoinGecko wciąż zawiera ruch, za który alert
    wyszedł mniej niż 24 h temu — kolejny alert z niej byłby tym samym ruchem."""

    _NOW = datetime(2026, 10, 17, 0, 7, tzinfo=UTC)

    def test_alert_from_23_hours_ago_is_still_in_the_window(self) -> None:
        from datetime import timedelta

        from src.domain.shock import is_move_already_alerted

        assert is_move_already_alerted(self._NOW - timedelta(hours=23), self._NOW) is True

    def test_alert_from_25_hours_ago_left_the_window(self) -> None:
        from datetime import timedelta

        from src.domain.shock import is_move_already_alerted

        assert is_move_already_alerted(self._NOW - timedelta(hours=25), self._NOW) is False

    def test_no_previous_alert_never_blocks(self) -> None:
        from src.domain.shock import is_move_already_alerted

        assert is_move_already_alerted(None, self._NOW) is False


class TestPortfolioShock:
    """ALR-07: próg per symbol (2× bramki) nie łapie skoordynowanego spadku wielu
    pozycji; portfel liczy Σ waga × Δ i ma własny próg."""

    THRESHOLD = Decimal("0.03")

    @staticmethod
    def _equal_weights(*symbols: str) -> dict[str, Decimal]:
        return {symbol: Decimal(1) / len(symbols) for symbol in symbols}

    def test_portfolio_shock_fires_on_correlated_small_moves(self) -> None:
        # 4 pozycje po 25%, każda −3,5%: żadna nie przekracza progu symbolu (4%),
        # a portfel traci 3,5% ≥ 3%.
        symbols = ("NVDA", "AMD", "AVGO", "TSM")
        weights = self._equal_weights(*symbols)
        deltas = dict.fromkeys(symbols, Decimal("-0.035"))
        for symbol in symbols:
            assert (
                detect_shock(
                    Asset(symbol, AssetType.STOCK),
                    PriceDelta(deltas[symbol]),
                    FAST_LOOP_THRESHOLD,
                    FRESH_SNAPSHOT_HOURS,
                )
                is None
            )

        shock = detect_portfolio_shock(weights, deltas, self.THRESHOLD)

        assert shock is not None
        assert shock.delta.fraction == Decimal("-0.035")
        assert shock.direction is ShockDirection.DOWN
        assert shock.measured_weight == Decimal(1)

    def test_move_below_the_portfolio_threshold_is_not_a_shock(self) -> None:
        weights = self._equal_weights("NVDA", "AMD")
        deltas = {"NVDA": Decimal("-0.06"), "AMD": Decimal("0.02")}  # Σ = −2%

        assert detect_portfolio_shock(weights, deltas, self.THRESHOLD) is None

    def test_a_big_mover_with_a_small_weight_does_not_move_the_portfolio(self) -> None:
        weights = {"NVDA": Decimal("0.1"), "VOO": Decimal("0.9")}
        deltas = {"NVDA": Decimal("-0.06"), "VOO": Decimal("0.0")}  # −0,6%

        assert detect_portfolio_shock(weights, deltas, self.THRESHOLD) is None

    def test_the_threshold_itself_fires(self) -> None:
        weights = {"NVDA": Decimal(1)}

        shock = detect_portfolio_shock(weights, {"NVDA": Decimal("-0.03")}, self.THRESHOLD)

        assert shock is not None

    def test_a_rise_is_a_shock_too(self) -> None:
        weights = self._equal_weights("NVDA", "AMD")
        deltas = {"NVDA": Decimal("0.05"), "AMD": Decimal("0.03")}

        shock = detect_portfolio_shock(weights, deltas, self.THRESHOLD)

        assert shock is not None
        assert shock.direction is ShockDirection.UP

    def test_movers_are_the_largest_contributors_in_the_direction_of_the_shock(self) -> None:
        weights = {
            "NVDA": Decimal("0.5"), "AMD": Decimal("0.3"),
            "AVGO": Decimal("0.1"), "TSM": Decimal("0.1"),
        }
        deltas = {
            "NVDA": Decimal("-0.08"), "AMD": Decimal("-0.05"),
            "AVGO": Decimal("-0.01"), "TSM": Decimal("0.04"),
        }

        shock = detect_portfolio_shock(weights, deltas, self.THRESHOLD, top=2)

        assert shock is not None
        assert shock.movers == (("NVDA", Decimal("-0.08")), ("AMD", Decimal("-0.05")))

    def test_symbols_without_a_move_are_left_out_not_guessed(self) -> None:
        weights = {"NVDA": Decimal("0.5"), "AMD": Decimal("0.5")}

        shock = detect_portfolio_shock(weights, {"NVDA": Decimal("-0.08")}, self.THRESHOLD)

        assert shock is not None
        assert shock.delta.fraction == Decimal("-0.04")
        assert shock.measured_weight == Decimal("0.5")

    def test_nothing_measured_is_no_shock(self) -> None:
        assert detect_portfolio_shock({"NVDA": Decimal(1)}, {}, self.THRESHOLD) is None
        assert detect_portfolio_shock({}, {"NVDA": Decimal("-0.5")}, self.THRESHOLD) is None

    def test_portfolio_alert_is_debounced_once_a_day_under_its_own_key(self) -> None:
        today = date(2026, 10, 26)
        sent = {(PORTFOLIO_ALERT_SYMBOL, today)}

        assert should_emit(PORTFOLIO_ALERT_SYMBOL, today, set()) is True
        assert should_emit(PORTFOLIO_ALERT_SYMBOL, today, sent) is False
        assert should_emit("NVDA", today, sent) is True
