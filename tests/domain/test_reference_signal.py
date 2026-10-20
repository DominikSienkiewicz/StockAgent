"""Testy domeny sygnału referencyjnego WIA-24 (`reference-v1`).

Reguła jest pre-rejestrowana: zamrożony plik protokołu i stała sha256 w domenie
muszą się zgadzać, a liczby okna z domeny muszą stać w tym pliku. Zmiana reguły
to nowa wersja, nie edycja v1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.domain.prediction import TrendDirection
from src.domain.reference_signal import (
    BASELINE_MAX_AGE,
    BASELINE_MIN_AGE,
    REFERENCE_PROTOCOL_PATH,
    REFERENCE_PROTOCOL_SHA256,
    REFERENCE_RECIPE_VERSION,
    ReferenceCall,
    ReferenceOutcome,
    ReferenceRead,
    momentum_call,
    protocol_digest,
    select_baseline,
    settle_call,
)

ROOT = Path(__file__).resolve().parents[2]
SUNDAY = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


def _read(price: str, at: datetime, symbol: str = "AAPL") -> ReferenceRead:
    return ReferenceRead(symbol=symbol, price=Decimal(price), read_at=at)


class TestProtocolHash:
    def test_protocol_file_matches_the_pinned_hash(self) -> None:
        # Edycja pliku protokołu bez nowej wersji (i odwrotnie) pada tutaj.
        text = (ROOT / REFERENCE_PROTOCOL_PATH).read_text(encoding="utf-8")
        assert protocol_digest(text) == REFERENCE_PROTOCOL_SHA256

    def test_digest_ignores_windows_line_endings(self) -> None:
        assert protocol_digest("a\r\nb\n") == protocol_digest("a\nb\n")

    def test_protocol_states_the_domain_window_and_label(self) -> None:
        text = (ROOT / REFERENCE_PROTOCOL_PATH).read_text(encoding="utf-8")
        assert f"at least {BASELINE_MIN_AGE.days} days" in text
        assert f"at most {BASELINE_MAX_AGE.days} days" in text
        assert f"recipe_version = '{REFERENCE_RECIPE_VERSION}'" in text

    def test_label_is_reference_v1(self) -> None:
        assert REFERENCE_RECIPE_VERSION == "reference-v1"


class TestMomentumCall:
    def test_momentum_rule_is_bullish_when_weekly_change_positive(self) -> None:
        call = momentum_call(_read("100", SUNDAY - timedelta(days=7)), _read("103", SUNDAY))

        assert call == ReferenceCall(
            symbol="AAPL",
            trend=TrendDirection.BULLISH,
            price_at_call=Decimal("103"),
            called_at=SUNDAY,
        )

    def test_momentum_rule_is_bearish_when_weekly_change_negative(self) -> None:
        call = momentum_call(_read("100", SUNDAY - timedelta(days=7)), _read("97", SUNDAY))

        assert call is not None
        assert call.trend is TrendDirection.BEARISH

    def test_rule_abstains_without_last_weeks_read(self) -> None:
        # Pierwszy przebieg: nie ma od czego liczyć znaku zmiany.
        assert momentum_call(None, _read("103", SUNDAY)) is None

    def test_rule_abstains_on_unchanged_price(self) -> None:
        assert momentum_call(_read("100", SUNDAY - timedelta(days=7)), _read("100", SUNDAY)) is None

    def test_rule_abstains_on_non_positive_price(self) -> None:
        assert momentum_call(_read("0", SUNDAY - timedelta(days=7)), _read("100", SUNDAY)) is None


class TestSelectBaseline:
    def test_picks_last_weeks_read(self) -> None:
        last_week = _read("100", SUNDAY - timedelta(days=7))
        mid_week = _read("101", SUNDAY - timedelta(days=3))

        assert select_baseline([last_week, mid_week], _read("103", SUNDAY)) == last_week

    def test_read_older_than_the_window_is_no_baseline(self) -> None:
        # Pominięty tydzień: momentum z dwóch tygodni to inna reguła, więc abstynencja.
        two_weeks = _read("100", SUNDAY - timedelta(days=14))

        assert select_baseline([two_weeks], _read("103", SUNDAY)) is None

    def test_window_edges_are_inclusive(self) -> None:
        current = _read("103", SUNDAY)

        assert select_baseline([_read("100", SUNDAY - BASELINE_MIN_AGE)], current) is not None
        assert select_baseline([_read("100", SUNDAY - BASELINE_MAX_AGE)], current) is not None

    def test_other_symbols_reads_are_ignored(self) -> None:
        other = _read("100", SUNDAY - timedelta(days=7), symbol="MSFT")

        assert select_baseline([other], _read("103", SUNDAY)) is None


class TestSettleCall:
    _CALL = ReferenceCall(
        symbol="AAPL",
        trend=TrendDirection.BULLISH,
        price_at_call=Decimal("100"),
        called_at=SUNDAY,
    )

    def test_bullish_call_hits_when_price_rose_by_next_run(self) -> None:
        outcome = settle_call(self._CALL, _read("104", SUNDAY + timedelta(days=7)))
        assert outcome is ReferenceOutcome.HIT

    def test_unchanged_price_is_a_miss(self) -> None:
        # Ta sama reguła kierunku co track record LLM (`Prediction.is_trend_correct`).
        outcome = settle_call(self._CALL, _read("100", SUNDAY + timedelta(days=7)))
        assert outcome is ReferenceOutcome.MISS

    def test_bearish_call_hits_when_price_fell(self) -> None:
        bearish = ReferenceCall(
            symbol="AAPL",
            trend=TrendDirection.BEARISH,
            price_at_call=Decimal("100"),
            called_at=SUNDAY,
        )
        assert settle_call(bearish, _read("96", SUNDAY + timedelta(days=7))) is ReferenceOutcome.HIT

    def test_mid_week_read_leaves_the_call_open(self) -> None:
        outcome = settle_call(self._CALL, _read("104", SUNDAY + timedelta(days=3)))
        assert outcome is ReferenceOutcome.PENDING

    def test_read_after_the_horizon_expires_the_call(self) -> None:
        outcome = settle_call(self._CALL, _read("104", SUNDAY + timedelta(days=14)))
        assert outcome is ReferenceOutcome.EXPIRED
