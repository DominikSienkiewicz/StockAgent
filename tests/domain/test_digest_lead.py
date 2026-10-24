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

from src.domain.consensus_shift import ConsensusShift, ShiftKind
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.digest_lead import (
    LeadItem,
    LeadSignal,
    build_lead,
    lead_headline,
    lead_preview,
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


class TestPortfolioAwareLead:
    """PORT-10: `build_lead` nie wiedział o portfelu, więc największy ruch spółki,
    której nie masz, wygrywał z ruchem 25% Twojego kapitału. Kolejność (decyzja
    2026-10-19): CRITICAL quota > sygnał przeciw Twojej pozycji > ruch Twojej
    pozycji > dotychczasowe tiery."""

    @staticmethod
    def _held(
        symbol: str = "NVDA",
        delta: str = "-4.2",
        weight: str = "0.24",
        verdict: CouncilVerdict | None = None,
        crossed: bool = True,
    ) -> LeadSignal:
        return LeadSignal(
            symbol=symbol,
            price_delta_pct=Decimal(delta),
            council_verdict=verdict,
            position_weight=Decimal(weight),
            crossed_volatility_gate=crossed,
        )

    def test_signal_against_large_position_outranks_strong_consensus(self) -> None:
        against = self._held(verdict=_consensus_verdict("SELL"))
        consensus = LeadSignal("AAA", Decimal("8.0"), _consensus_verdict("BUY"))

        items = build_lead([consensus, against])

        assert "przeciw Twojej pozycji" in items[0].headline
        assert "SPRZEDAJ" in items[0].headline
        assert "NVDA" in items[0].headline
        assert "24%" in items[0].headline
        assert "AAA" in items[1].headline

    def test_critical_quota_still_outranks_a_signal_against_your_position(self) -> None:
        against = self._held(verdict=_consensus_verdict("SELL"))

        items = build_lead([against], [_critical_alert()])

        assert "Alpha Vantage" in items[0].headline
        assert "przeciw Twojej pozycji" in items[1].headline

    def test_signal_against_position_outranks_a_move_of_another_position(self) -> None:
        move = self._held(symbol="AMD", delta="-9.0", weight="0.40")
        against = self._held(verdict=_consensus_verdict("SELL"))

        items = build_lead([move, against])

        assert "przeciw Twojej pozycji" in items[0].headline
        assert items[1].headline == "AMD (40% portfela) -9,0%"

    def test_position_move_outranks_a_split_council(self) -> None:
        move = self._held()
        split = LeadSignal("AAA", Decimal("6.0"), _split_verdict())

        items = build_lead([split, move])

        assert items[0].icon == "💼"
        assert items[0].headline == "NVDA (24% portfela) -4,2%"
        assert "PODZIELONA" in items[1].headline

    def test_position_moves_rank_by_their_impact_on_capital(self) -> None:
        small = self._held(symbol="AMD", delta="-6.0", weight="0.10")
        large = self._held(symbol="NVDA", delta="-3.0", weight="0.30")

        items = build_lead([small, large])

        assert "NVDA" in items[0].headline
        assert "AMD" in items[1].headline

    def test_buy_verdict_on_a_held_position_is_a_move_not_a_warning(self) -> None:
        items = build_lead([self._held(verdict=_consensus_verdict("BUY"))])

        assert items[0].icon == "💼"
        assert "przeciw" not in items[0].headline

    def test_move_below_the_volatility_gate_is_not_a_position_lead(self) -> None:
        # Anti-sensacja zostaje: ruch pod bramką nic nie znaczy, także na pozycji.
        assert build_lead([self._held(crossed=False)]) == []

    def test_small_position_keeps_the_previous_tiers(self) -> None:
        small = self._held(weight="0.05", verdict=_consensus_verdict("SELL"))

        items = build_lead([small])

        assert items[0].headline == "NVDA -4,2%, rada zgodna: SPRZEDAJ"

    def test_without_a_portfolio_the_lead_is_unchanged(self) -> None:
        signal = LeadSignal(
            "NVDA", Decimal("-4.2"), _consensus_verdict("SELL"), crossed_volatility_gate=True
        )

        assert build_lead([signal])[0].headline == "NVDA -4,2%, rada zgodna: SPRZEDAJ"


class TestLeadPreview:
    """MAIL-13: preheader (podgląd w skrzynce) bierze `detail` pierwszej pozycji —
    headline stoi już w temacie maila."""

    def test_returns_the_detail_of_the_first_item(self) -> None:
        items = [
            LeadItem(icon="🔥", headline="NVDA +4.2%, rada zgodna: BUY", detail="Rada: BUY."),
            LeadItem(icon="⚖️", headline="AMD -3.0%, rada PODZIELONA", detail="1 BUY / 1 SELL"),
        ]

        assert lead_preview(items) == "Rada: BUY."

    def test_falls_back_to_the_headline_without_a_detail(self) -> None:
        items = [LeadItem(icon="🔥", headline="NVDA +4.2%, rada zgodna: BUY", detail="  ")]

        assert lead_preview(items) == "NVDA +4.2%, rada zgodna: BUY"

    def test_none_on_empty(self) -> None:
        assert lead_preview([]) is None


def test_strong_consensus_headline_uses_polish_label() -> None:
    # MAIL-15: temat maila powstaje z leadu — „rada zgodna: BUY” mieszało języki.
    signal = LeadSignal("NVDA", Decimal("4.2"), _consensus_verdict("BUY", strength=0.8))

    headline = build_lead([signal])[0].headline

    assert "KUP" in headline
    assert "BUY" not in headline


@pytest.mark.parametrize(
    ("recommendation", "label"),
    [("BUY", "KUP"), ("SELL", "SPRZEDAJ"), ("HOLD", "TRZYMAJ")],
)
def test_strong_consensus_headline_labels_every_verdict_in_polish(
    recommendation: Literal["BUY", "SELL", "HOLD"], label: str
) -> None:
    signal = LeadSignal("AMD", Decimal("-3.0"), _consensus_verdict(recommendation))

    assert build_lead([signal])[0].headline == f"AMD -3,0%, rada zgodna: {label}"


class TestPolishNumbers:
    """MAIL-58: headline leadu idzie do tematu maila i pushu, więc liczby są po
    polsku: przecinek dziesiętny, minus ASCII, zaokrąglenie ROUND_HALF_UP."""

    def test_headline_uses_a_decimal_comma(self) -> None:
        signal = LeadSignal("NVDA", Decimal("-4.2"), _split_verdict())

        assert build_lead([signal])[0].headline == "NVDA -4,2%, rada PODZIELONA"

    def test_delta_rounds_half_up(self) -> None:
        signal = LeadSignal("NVDA", Decimal("4.25"), _split_verdict())

        assert build_lead([signal])[0].headline == "NVDA +4,3%, rada PODZIELONA"

    def test_position_weight_rounds_half_up(self) -> None:
        signal = LeadSignal(
            "NVDA",
            Decimal("-4.2"),
            position_weight=Decimal("0.245"),
            crossed_volatility_gate=True,
        )

        assert build_lead([signal])[0].headline == "NVDA (25% portfela) -4,2%"

    def test_tiny_drop_reads_as_zero_without_a_minus(self) -> None:
        signal = LeadSignal("NVDA", Decimal("-0.04"), resolved_label="trafiona prognoza spadku")

        assert build_lead([signal])[0].headline == "NVDA +0,0%: trafiona prognoza spadku"


class TestLeadItemSymbol:
    """MAIL-06: raport daje symbolom z leadu pełny blok uzasadnień, więc pozycja
    leadu mówi, którego tickera dotyczy; alert limitu nie dotyczy żadnego."""

    def test_signal_item_carries_its_symbol(self) -> None:
        signal = LeadSignal(
            symbol="AAA", price_delta_pct=Decimal("2.0"), council_verdict=_split_verdict()
        )

        assert [item.symbol for item in build_lead([signal])] == ["AAA"]

    def test_quota_item_has_no_symbol(self) -> None:
        assert [item.symbol for item in build_lead([], [_critical_alert()])] == [None]


def _shift(
    kind: ShiftKind = ShiftKind.FLIP,
    previous: str = "BUY",
    current: str = "SELL",
) -> ConsensusShift:
    return ConsensusShift(
        kind=kind,
        previous_recommendation=previous,
        current_recommendation=current,
        votes_changed=4,
        sentiment_delta=-0.4,
        consensus_delta=0.1,
        gap_hours=24.0,
    )


class TestConsensusFlip:
    """MAIL-19: odwrócenie rady (FLIP, BUY↔SELL) to najbardziej actionable zmiana
    nastawienia — ma własny tier leadu między radą podzieloną a silnym konsensusem."""

    def test_flip_outranks_strong_consensus_but_not_split(self) -> None:
        flip = LeadSignal(
            "NVDA", Decimal("1.0"), _consensus_verdict("SELL"), consensus_shift=_shift()
        )
        consensus = LeadSignal("AAA", Decimal("8.0"), _consensus_verdict("BUY"))
        big_flip = LeadSignal(
            "NVDA", Decimal("9.0"), _consensus_verdict("SELL"), consensus_shift=_shift()
        )
        split = LeadSignal("BBB", Decimal("1.0"), _split_verdict())

        assert "NVDA" in build_lead([consensus, flip])[0].headline
        over_flip = build_lead([big_flip, split])
        assert "PODZIELONA" in over_flip[0].headline
        assert "NVDA" in over_flip[1].headline

    def test_flip_headline_names_both_recommendations_in_polish(self) -> None:
        signal = LeadSignal(
            "NVDA", Decimal("-4.2"), _consensus_verdict("SELL"), consensus_shift=_shift()
        )

        item = build_lead([signal])[0]

        assert item.icon == "🔄"
        assert item.headline == "NVDA -4,2%: rada odwróciła się KUP→SPRZEDAJ"
        assert item.detail == "Rada zgodna co do kierunku."
        assert item.symbol == "NVDA"

    def test_stale_comparison_never_leads_as_a_flip(self) -> None:
        signal = LeadSignal(
            "NVDA",
            Decimal("-4.2"),
            _consensus_verdict("SELL"),
            consensus_shift=_shift(ShiftKind.STALE_COMPARISON),
        )

        assert build_lead([signal])[0].headline == "NVDA -4,2%, rada zgodna: SPRZEDAJ"

    def test_softening_is_not_a_flip(self) -> None:
        signal = LeadSignal(
            "NVDA",
            Decimal("-4.2"),
            _consensus_verdict("HOLD", 0.5),
            consensus_shift=_shift(ShiftKind.SOFTENING, current="HOLD"),
        )

        assert build_lead([signal]) == []

    def test_flip_stays_below_a_signal_against_your_position(self) -> None:
        flip = LeadSignal(
            "AMD", Decimal("-9.0"), _consensus_verdict("SELL"), consensus_shift=_shift()
        )
        split = LeadSignal("BBB", Decimal("0.5"), _split_verdict())
        against = LeadSignal(
            "NVDA",
            Decimal("-1.0"),
            _consensus_verdict("SELL"),
            position_weight=Decimal("0.24"),
            crossed_volatility_gate=True,
        )

        items = build_lead([flip, split, against])

        assert [item.symbol for item in items] == ["NVDA", "BBB", "AMD"]

class TestSignificantInsiderBuyTier:
    """DAT-21: znaczący zakup insidera to osobny tier między radą PODZIELONĄ
    a silnym konsensusem — kolejność pozostałych tierów bez zmian."""

    _LABEL = "CEO kupił akcje za 1,2 mln USD"

    def _insider(self, symbol: str, delta: str = "1.0") -> LeadSignal:
        return LeadSignal(
            symbol=symbol, price_delta_pct=Decimal(delta), insider_buy_label=self._LABEL
        )

    def test_insider_buy_sits_between_a_split_council_and_a_strong_consensus(self) -> None:
        split = LeadSignal("AAA", Decimal("0.5"), _split_verdict())
        insider = self._insider("BBB", "0.5")
        # Większy ruch konsensusu nie przeskakuje tieru (anti-sensacja).
        consensus = LeadSignal("CCC", Decimal("9.0"), _consensus_verdict())

        items = build_lead([consensus, insider, split])

        assert [item.symbol for item in items] == ["AAA", "BBB", "CCC"]
        assert items[1].headline == f"BBB +0,5%: {self._LABEL}"
        assert items[1].icon == "🟢"

    def test_insider_buy_takes_the_place_of_the_symbols_strong_consensus(self) -> None:
        signal = LeadSignal(
            "NVDA",
            Decimal("2.0"),
            _consensus_verdict(),
            insider_buy_label=self._LABEL,
        )

        (item,) = build_lead([signal])

        assert self._LABEL in item.headline
        assert item.detail == "Rada zgodna co do kierunku."

    def test_without_a_council_the_detail_names_the_form4_purchase(self) -> None:
        (item,) = build_lead([self._insider("NVDA")])

        assert "Form 4" in item.detail

    def test_critical_quota_and_split_council_stay_above_it(self) -> None:
        split = LeadSignal("AAA", Decimal("0.1"), _split_verdict())
        insider = self._insider("BBB", "9.9")

        items = build_lead([insider, split], [_critical_alert()])

        assert items[0].symbol is None
        assert [item.symbol for item in items[1:]] == ["AAA", "BBB"]


class TestRegimeChangeTier:
    """ALR-05: zmiana reżimu tuż pod CRITICAL quota — dotyczy całego cyklu (próg
    bramki rośnie albo wraca), więc bije sygnały pojedynczych symboli."""

    @staticmethod
    def _against_position() -> LeadSignal:
        return LeadSignal(
            symbol="NVDA",
            price_delta_pct=Decimal("-9.9"),
            council_verdict=_consensus_verdict("SELL"),
            position_weight=Decimal("0.60"),
            crossed_volatility_gate=True,
        )

    def test_entering_risk_off_sits_right_below_a_critical_quota(self) -> None:
        from src.domain.regime import MarketRegime, RegimeChange

        change = RegimeChange(previous=MarketRegime.NEUTRAL, current=MarketRegime.RISK_OFF)

        items = build_lead(
            [self._against_position()], [_critical_alert()], regime_change=change
        )

        assert "Alpha Vantage" in items[0].headline
        assert items[1].headline == "Rynek przeszedł w RISK_OFF, analiza rzadsza"
        assert items[1].icon == "🌧️"
        assert items[1].symbol is None
        assert "Sygnał przeciw Twojej pozycji" in items[2].headline

    def test_leaving_risk_off_names_the_new_regime(self) -> None:
        from src.domain.regime import MarketRegime, RegimeChange

        change = RegimeChange(previous=MarketRegime.RISK_OFF, current=MarketRegime.NEUTRAL)

        (item,) = build_lead([], regime_change=change)

        assert item.headline == "Rynek wyszedł z RISK_OFF (NEUTRAL), analiza częstsza"

    def test_without_a_change_the_lead_is_unchanged(self) -> None:
        signals = [self._against_position()]

        assert build_lead(signals, [_critical_alert()], regime_change=None) == build_lead(
            signals, [_critical_alert()]
        )
