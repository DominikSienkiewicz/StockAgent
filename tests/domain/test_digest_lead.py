"""Testy domeny Digest Lead — ranking najważniejszych sygnałów cyklu.

Kluczowy invariant zamrożony przez te testy: ranking NIE promuje sensacji
(dużego ruchu ceny) kosztem faktycznie krytycznych sygnałów. Kolejność
priorytetów jest twarda:

    1. CRITICAL QuotaAlert
    2. rada PODZIELONA (is_split_decision) przy dużym |Δ|
    3. największy |Δ| z silnym konsensusem (has_strong_consensus)
    4. werdykt zamkniętej predykcji
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import pytest

from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.digest_lead import (
    LeadItem,
    LeadSignal,
    build_lead,
    lead_headline,
)
from src.domain.quota import QuotaAlert, QuotaSeverity


def _opinion(
    name: str,
    rec: Literal["BUY", "SELL", "HOLD"],
    confidence: float = 0.8,
) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=rec,
        confidence=confidence,
        reasoning="reasoning",
        key_factors=(),
    )


def _split_verdict(strength: float = 0.5) -> CouncilVerdict:
    """Rada z jednoczesnym BUY i SELL → is_split_decision()==True.

    Strength 0.5 (<0.7) gwarantuje, że NIE jest jednocześnie silnym konsensusem.
    """
    return CouncilVerdict(
        final_recommendation="HOLD",
        consensus_strength=strength,
        summary="Rada nie zgadza się co do kierunku.",
        dissenting_views=(),
        investor_opinions=(_opinion("A", "BUY"), _opinion("B", "SELL")),
    )


def _consensus_verdict(
    rec: Literal["BUY", "SELL", "HOLD"] = "BUY",
    strength: float = 0.85,
) -> CouncilVerdict:
    """Rada zgodna (strength ≥ 0.7), bez sprzecznych głosów → strong consensus."""
    return CouncilVerdict(
        final_recommendation=rec,
        consensus_strength=strength,
        summary="Rada zgodna co do kierunku.",
        dissenting_views=(),
        investor_opinions=(_opinion("A", rec), _opinion("B", rec)),
    )


def _critical_alert(source: str = "Alpha Vantage") -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.CRITICAL,
        message="All keys exhausted at 14:32",
        action="Add new API key or wait for reset",
        occurred_at=datetime(2026, 6, 1, 14, 32, tzinfo=UTC),
    )


def _warning_alert(source: str = "Resend") -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.WARNING,
        message="Key rotation succeeded",
        action="Monitor usage",
        occurred_at=datetime(2026, 6, 1, 14, 32, tzinfo=UTC),
    )


class TestEmpty:
    def test_no_signals_no_alerts_returns_empty(self) -> None:
        """Sekcja samosupresująca — brak sygnałów → pusta lista."""
        assert build_lead([], []) == []

    def test_lead_headline_none_on_empty(self) -> None:
        assert lead_headline([]) is None


class TestPriorityOrdering:
    def test_critical_quota_beats_biggest_move(self) -> None:
        """Priorytet 1 > wszystko: CRITICAL QuotaAlert bije największy |Δ|,
        nawet gdy ten ruch ma silny konsensus rady (anti-sensacja).
        """
        big_move = LeadSignal(
            symbol="NVDA",
            price_delta_pct=Decimal("-9.9"),
            council_verdict=_consensus_verdict("SELL"),
        )
        items = build_lead([big_move], [_critical_alert()])

        assert items, "oczekiwano co najmniej jednego elementu"
        assert "Alpha Vantage" in items[0].headline
        # Wielki ruch NVDA schodzi poniżej krytycznego alertu.
        assert any("NVDA" in it.headline for it in items[1:])

    def test_split_beats_strong_consensus_at_smaller_delta(self) -> None:
        """Priorytet 2 > 3: rada PODZIELONA przy MNIEJSZYM Δ bije silny
        konsensus przy WIĘKSZYM Δ. To jądro invariantu anti-sensacja.
        """
        split_sig = LeadSignal(
            symbol="AAA",
            price_delta_pct=Decimal("2.0"),
            council_verdict=_split_verdict(),
        )
        consensus_sig = LeadSignal(
            symbol="BBB",
            price_delta_pct=Decimal("8.0"),
            council_verdict=_consensus_verdict(),
        )
        # Konsensus podany PIERWSZY w wejściu — dowód, że wygrywa ranking,
        # nie kolejność argumentów.
        items = build_lead([consensus_sig, split_sig])

        assert "AAA" in items[0].headline
        assert "PODZIELONA" in items[0].headline
        assert "BBB" in items[1].headline

    def test_strong_consensus_beats_closed_verdict(self) -> None:
        """Priorytet 3 > 4: silny konsensus bije werdykt zamkniętej predykcji."""
        consensus_sig = LeadSignal(
            symbol="AAA",
            price_delta_pct=Decimal("1.0"),
            council_verdict=_consensus_verdict(),
        )
        closed_sig = LeadSignal(
            symbol="BBB",
            price_delta_pct=Decimal("7.0"),
            resolved_label="trafiona predykcja BUY",
        )
        items = build_lead([closed_sig, consensus_sig])

        assert "AAA" in items[0].headline
        assert "BBB" in items[1].headline

    def test_largest_delta_wins_within_strong_consensus(self) -> None:
        """W obrębie tego samego priorytetu decyduje |Δ| — większy ruch wyżej."""
        small = LeadSignal("AAA", Decimal("2.0"), _consensus_verdict())
        large = LeadSignal("BBB", Decimal("-6.0"), _consensus_verdict("SELL"))
        items = build_lead([small, large])

        assert "BBB" in items[0].headline
        assert "AAA" in items[1].headline


class TestAntiSensation:
    def test_big_move_without_qualifying_signal_suppressed(self) -> None:
        """Sam wielki ruch — bez rady i bez zamkniętej predykcji — NIE jest
        leadem. Ranking nie promuje gołej sensacji.
        """
        noise = LeadSignal(symbol="MEME", price_delta_pct=Decimal("42.0"))
        assert build_lead([noise]) == []

    def test_warning_quota_does_not_lead(self) -> None:
        """Tylko CRITICAL QuotaAlert jest priorytetem 1. WARNING ma własny
        banner i nie trafia do leadu.
        """
        assert build_lead([], [_warning_alert()]) == []


class TestAdditiveScoring:
    def test_symbol_without_council_still_scored(self) -> None:
        """Scoring jest ADDYTYWNY: symbol z council_verdict=None, ale z
        werdyktem zamkniętej predykcji, dostaje własny lead. Mnożnikowy
        scoring wyzerowałby go (None → 0).
        """
        sig = LeadSignal(
            symbol="AAA",
            price_delta_pct=Decimal("3.0"),
            council_verdict=None,
            resolved_label="trafiona predykcja SELL",
        )
        items = build_lead([sig])

        assert len(items) == 1
        blob = items[0].headline + items[0].detail
        assert "AAA" in blob
        assert "trafiona predykcja SELL" in blob


class TestCapAndLimits:
    def test_max_three_items(self) -> None:
        """Max 3 pozycje, nawet przy nadmiarze sygnałów."""
        signals = [
            LeadSignal(f"S{i}", Decimal("1.0"), _consensus_verdict())
            for i in range(6)
        ]
        items = build_lead(signals, [_critical_alert(), _critical_alert("OpenAI")])

        assert len(items) == 3

    def test_custom_max_items(self) -> None:
        signals = [
            LeadSignal(f"S{i}", Decimal("1.0"), _consensus_verdict())
            for i in range(4)
        ]
        assert len(build_lead(signals, max_items=2)) == 2


class TestLeadHeadline:
    def test_returns_first_headline(self) -> None:
        items = build_lead([], [_critical_alert()])
        assert lead_headline(items) == items[0].headline


class TestLeadItem:
    def test_is_frozen(self) -> None:
        item = LeadItem(icon="⚠️", headline="h", detail="d")
        with pytest.raises(FrozenInstanceError):
            item.headline = "other"  # type: ignore[misc]
