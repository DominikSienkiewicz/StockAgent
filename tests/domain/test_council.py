# tests/domain/test_council.py
from decimal import Decimal

import pytest

from src.domain.council import (
    CouncilInput,
    CouncilVerdict,
    InvestorOpinion,
    derive_consensus,
    vindicated_dissenters,
)
from src.domain.value_objects import AssetType


def _named_opinion(name: str, rec: str) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=rec,
        confidence=0.8,
        reasoning="...",
        key_factors=(),
    )


class TestVindicatedDissenters:
    """Dysydent 'wybroniony' = głosował zgodnie z RZECZYWISTYM ruchem, ale
    niezgodnie z finalną rekomendacją rady (BUY↔UP, SELL↔DOWN, HOLD↔FLAT)."""

    def test_returns_dissenter_who_matched_actual_move(self):
        opinions = (
            _named_opinion("Burry", "SELL"),
            _named_opinion("Wood", "BUY"),
            _named_opinion("Marks", "HOLD"),
        )
        # Rada powiedziała BUY, cena SPADŁA (DOWN) → SELL-owy Burry miał rację.
        result = vindicated_dissenters(opinions, "DOWN", "BUY")
        assert tuple(o.investor_name for o in result) == ("Burry",)

    def test_empty_when_final_recommendation_was_already_right(self):
        opinions = (_named_opinion("Wood", "BUY"), _named_opinion("Burry", "SELL"))
        # Cena wzrosła (UP), rada powiedziała BUY → trafna decyzja = finalna,
        # więc żaden dysydent nie został 'wybroniony'.
        assert vindicated_dissenters(opinions, "UP", "BUY") == ()

    def test_flat_maps_to_hold(self):
        opinions = (_named_opinion("Marks", "HOLD"), _named_opinion("Wood", "BUY"))
        result = vindicated_dissenters(opinions, "FLAT", "BUY")
        assert tuple(o.investor_name for o in result) == ("Marks",)

    def test_empty_opinions(self):
        assert vindicated_dissenters((), "UP", "SELL") == ()


def _make_opinion(rec: str = "BUY", confidence: float = 0.8) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name="Warren Buffett",
        recommendation=rec,
        confidence=confidence,
        reasoning="Silna fosa ekonomiczna.",
        key_factors=["moat", "FCF", "długi horyzont"],
    )


class TestInvestorOpinion:
    def test_frozen(self):
        op = _make_opinion()
        try:
            op.confidence = 0.1  # type: ignore[misc]
            raise AssertionError("should raise")
        except (AttributeError, TypeError):
            pass

    def test_recommendation_stored(self):
        assert _make_opinion("SELL").recommendation == "SELL"

    def test_key_factors_stored(self):
        op = _make_opinion()
        assert "moat" in op.key_factors


class TestCouncilVerdict:
    def test_frozen(self):
        opinions = [_make_opinion()]
        v = CouncilVerdict(
            final_recommendation="BUY",
            consensus_strength=0.73,
            summary="Większość radzi kupować.",
            dissenting_views=["Soros: zbyt ryzykowne"],
            investor_opinions=opinions,
        )
        try:
            v.consensus_strength = 0.1  # type: ignore[misc]
            raise AssertionError("should raise")
        except (AttributeError, TypeError):
            pass

    def test_holds_all_opinions(self):
        ops = [_make_opinion("BUY"), _make_opinion("SELL")]
        v = CouncilVerdict(
            final_recommendation="BUY",
            consensus_strength=0.5,
            summary="Mieszane opinie.",
            dissenting_views=["Livermore: sprzedaj"],
            investor_opinions=ops,
        )
        assert len(v.investor_opinions) == 2


class TestCouncilVerdictBehavior:
    """Metody domenowe — zastępują ad-hoc logikę w grafie/raporcie.

    Zachowanie należy do agregata, nie do warstwy prezentacji."""

    def _verdict(
        self, opinions: list[InvestorOpinion],
        final: str = "BUY", consensus: float = 0.7,
    ) -> CouncilVerdict:
        return CouncilVerdict(
            final_recommendation=final,  # type: ignore[arg-type]
            consensus_strength=consensus,
            summary="Test.",
            dissenting_views=[],
            investor_opinions=opinions,
        )

    # --- has_strong_consensus ---

    def test_strong_consensus_above_threshold(self):
        v = self._verdict([_make_opinion()], consensus=0.85)
        assert v.has_strong_consensus() is True

    def test_strong_consensus_default_threshold_is_0_7(self):
        # Próg 0.7 — typowy "wyraźny konsensus" dla rady.
        v = self._verdict([_make_opinion()], consensus=0.69)
        assert v.has_strong_consensus() is False
        v2 = self._verdict([_make_opinion()], consensus=0.70)
        assert v2.has_strong_consensus() is True

    def test_strong_consensus_custom_threshold(self):
        v = self._verdict([_make_opinion()], consensus=0.55)
        assert v.has_strong_consensus(threshold=0.5) is True
        assert v.has_strong_consensus(threshold=0.6) is False

    # --- is_split_decision ---

    def test_split_decision_when_both_buy_and_sell_present(self):
        ops = [_make_opinion("BUY"), _make_opinion("SELL"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is True

    def test_not_split_when_only_buy_and_hold(self):
        ops = [_make_opinion("BUY"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    def test_not_split_when_only_sell_and_hold(self):
        ops = [_make_opinion("SELL"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    def test_not_split_when_unanimous(self):
        ops = [_make_opinion("BUY"), _make_opinion("BUY"), _make_opinion("BUY")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    # --- vote_distribution ---

    def test_vote_distribution_counts_each_recommendation(self):
        ops = [
            _make_opinion("BUY"),
            _make_opinion("BUY"),
            _make_opinion("SELL"),
            _make_opinion("HOLD"),
        ]
        v = self._verdict(ops)
        dist = v.vote_distribution()
        assert dist == {"BUY": 2, "SELL": 1, "HOLD": 1}

    def test_vote_distribution_missing_recommendations_are_zero(self):
        ops = [_make_opinion("BUY"), _make_opinion("BUY")]
        v = self._verdict(ops)
        # Wszystkie trzy klucze zawsze obecne — łatwiejszy template
        dist = v.vote_distribution()
        assert dist == {"BUY": 2, "SELL": 0, "HOLD": 0}

    def test_vote_distribution_empty_opinions(self):
        v = self._verdict([])
        assert v.vote_distribution() == {"BUY": 0, "SELL": 0, "HOLD": 0}

    # --- dissent_ratio ---

    def test_dissent_ratio_fraction_disagreeing_with_final(self):
        # 2 z 4 nie zgadzają się z finalnym BUY → 0.5
        ops = [
            _make_opinion("BUY"),
            _make_opinion("BUY"),
            _make_opinion("SELL"),
            _make_opinion("HOLD"),
        ]
        v = self._verdict(ops, final="BUY")
        assert v.dissent_ratio() == 0.5

    def test_dissent_ratio_zero_when_unanimous(self):
        ops = [_make_opinion("BUY") for _ in range(5)]
        v = self._verdict(ops, final="BUY")
        assert v.dissent_ratio() == 0.0

    def test_dissent_ratio_zero_when_no_opinions(self):
        v = self._verdict([], final="BUY")
        assert v.dissent_ratio() == 0.0


class TestInvestorOpinionBehavior:
    def test_confidence_label_high(self):
        op = _make_opinion(confidence=0.85)
        assert op.confidence_label() == "HIGH"

    def test_confidence_label_medium(self):
        op = _make_opinion(confidence=0.6)
        assert op.confidence_label() == "MEDIUM"

    def test_confidence_label_low(self):
        op = _make_opinion(confidence=0.3)
        assert op.confidence_label() == "LOW"

    def test_confidence_label_boundary_high(self):
        # ≥ 0.75 = HIGH
        assert _make_opinion(confidence=0.75).confidence_label() == "HIGH"
        assert _make_opinion(confidence=0.749).confidence_label() == "MEDIUM"

    def test_confidence_label_boundary_low(self):
        # < 0.5 = LOW
        assert _make_opinion(confidence=0.5).confidence_label() == "MEDIUM"
        assert _make_opinion(confidence=0.4999).confidence_label() == "LOW"


class TestDeriveConsensus:
    """Czysta funkcja domenowa: rekomendacja + siła konsensusu liczone z
    realnych głosów inwestorów (confidence-weighted), nie z liczby od chairman."""

    def test_unanimous_buy_high_strength(self):
        ops = [_make_opinion("BUY", confidence=0.8) for _ in range(5)]
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        # Jednogłośne BUY → cała "masa" pewności na zwycięskim koszyku → 1.0
        assert strength == 1.0

    def test_three_buy_one_sell_three_hold_deterministic(self):
        # 3×BUY, 1×SELL, 3×HOLD — wszystkie confidence=0.8.
        # Suma confidence = 7 * 0.8 = 5.6; zwycięzca to remis BUY/HOLD po 3.
        # Tie-break preferuje BUY > SELL > HOLD → BUY wygrywa.
        # strength = (3 * 0.8) / (7 * 0.8) = 2.4 / 5.6 = 3/7.
        ops = (
            [_make_opinion("BUY", confidence=0.8) for _ in range(3)]
            + [_make_opinion("SELL", confidence=0.8)]
            + [_make_opinion("HOLD", confidence=0.8) for _ in range(3)]
        )
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        assert strength == pytest.approx(3 / 7)

    def test_confidence_weighted_winner_not_headcount(self):
        # 1×BUY z confidence 0.9 vs 2×SELL z confidence 0.1 każdy.
        # Headcount mówi SELL (2 głosy), ale waga pewności mówi BUY
        # (0.9 vs 0.2). Funkcja musi ważyć pewnością.
        ops = [
            _make_opinion("BUY", confidence=0.9),
            _make_opinion("SELL", confidence=0.1),
            _make_opinion("SELL", confidence=0.1),
        ]
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        assert strength == pytest.approx(0.9 / 1.1)

    def test_empty_opinions_safe_default(self):
        rec, strength = derive_consensus([])
        assert rec == "HOLD"
        assert strength == 0.0

    def test_all_zero_confidence_falls_back_to_headcount(self):
        # Gdy wszyscy mają confidence 0.0, waga jest bezużyteczna — wracamy
        # do liczenia głosów, żeby nie dzielić przez zero.
        ops = [
            _make_opinion("SELL", confidence=0.0),
            _make_opinion("SELL", confidence=0.0),
            _make_opinion("BUY", confidence=0.0),
        ]
        rec, strength = derive_consensus(ops)
        assert rec == "SELL"
        assert strength == pytest.approx(2 / 3)

    def test_strength_in_unit_interval(self):
        ops = [
            _make_opinion("BUY", confidence=0.7),
            _make_opinion("SELL", confidence=0.6),
            _make_opinion("HOLD", confidence=0.55),
        ]
        _, strength = derive_consensus(ops)
        assert 0.0 <= strength <= 1.0


class TestDeriveConsensusWeighted:
    """Feature #3: adaptacyjne ważenie person. Masa głosu = confidence * waga
    person (z historycznej trafności). weights=None → identycznie jak dziś."""

    def _named(self, name: str, rec: str, conf: float = 0.8) -> InvestorOpinion:
        return InvestorOpinion(
            investor_name=name,
            recommendation=rec,  # type: ignore[arg-type]
            confidence=conf,
            reasoning="...",
            key_factors=(),
        )

    def test_weights_none_identical_to_unweighted(self):
        # Brak wag → wynik dokładnie taki jak bez parametru.
        ops = [
            self._named("A", "BUY", 0.9),
            self._named("B", "SELL", 0.1),
            self._named("C", "SELL", 0.1),
        ]
        assert derive_consensus(ops, None) == derive_consensus(ops)

    def test_high_weight_minority_flips_winner(self):
        # Headcount: 2×SELL vs 1×BUY. Confidence równe (0.5 każdy), więc bez
        # wag SELL wygrywa (masa 1.0 vs 0.5). Z mocną wagą dla persony "Star"
        # (BUY) jej głos przeważa i odwraca werdykt na BUY.
        ops = [
            self._named("Star", "BUY", 0.5),
            self._named("B", "SELL", 0.5),
            self._named("C", "SELL", 0.5),
        ]
        # Bez wag: SELL.
        assert derive_consensus(ops)[0] == "SELL"
        # Z wagą: Star ma trafność 5× wyższą → BUY masa 0.5*5=2.5 > SELL 1.0.
        weights = {"Star": 5.0, "B": 1.0, "C": 1.0}
        rec, strength = derive_consensus(ops, weights)
        assert rec == "BUY"
        # strength = 2.5 / (2.5 + 0.5 + 0.5) = 2.5 / 3.5.
        assert strength == pytest.approx(2.5 / 3.5)

    def test_missing_name_defaults_weight_one(self):
        # Persona spoza mapy wag dostaje wagę 1.0 (neutralną).
        ops = [
            self._named("Known", "BUY", 0.5),
            self._named("Unknown", "SELL", 0.5),
        ]
        weights = {"Known": 1.0}  # "Unknown" nie ma wpisu → 1.0
        rec, strength = derive_consensus(ops, weights)
        # Remis mas (0.5 vs 0.5) → tie-break BUY > SELL.
        assert rec == "BUY"
        assert strength == pytest.approx(0.5)

    def test_empty_weights_mapping_acts_like_unweighted(self):
        ops = [self._named("A", "BUY", 0.9), self._named("B", "SELL", 0.1)]
        assert derive_consensus(ops, {}) == derive_consensus(ops)


class TestOpinionShift:
    """Feature #1 (debate mode): ilu inwestorów zmieniło rekomendację między
    rundą 1 a finalnym werdyktem (po rewizji)."""

    def _verdict(self, opinions: list[InvestorOpinion]) -> CouncilVerdict:
        return CouncilVerdict(
            final_recommendation="BUY",
            consensus_strength=0.7,
            summary="Test.",
            dissenting_views=[],
            investor_opinions=opinions,
        )

    def test_counts_investors_who_changed_recommendation(self):
        before = [
            _make_opinion("BUY"),  # Warren Buffett (nazwa z _make_opinion)
        ]
        before[0] = InvestorOpinion(
            investor_name="A", recommendation="SELL", confidence=0.8,
            reasoning="x", key_factors=(),
        )
        after = [
            InvestorOpinion(
                investor_name="A", recommendation="BUY", confidence=0.8,
                reasoning="y", key_factors=(),
            )
        ]
        v = self._verdict(after)
        assert v.opinion_shift(before) == 1

    def test_zero_when_nobody_changed(self):
        before = [
            InvestorOpinion(
                investor_name="A", recommendation="BUY", confidence=0.8,
                reasoning="x", key_factors=(),
            )
        ]
        after = [
            InvestorOpinion(
                investor_name="A", recommendation="BUY", confidence=0.9,
                reasoning="changed reasoning only", key_factors=(),
            )
        ]
        v = self._verdict(after)
        assert v.opinion_shift(before) == 0

    def test_matches_by_investor_name(self):
        before = [
            InvestorOpinion(
                investor_name="A", recommendation="SELL", confidence=0.8,
                reasoning="x", key_factors=(),
            ),
            InvestorOpinion(
                investor_name="B", recommendation="HOLD", confidence=0.8,
                reasoning="x", key_factors=(),
            ),
        ]
        after = [
            InvestorOpinion(
                investor_name="A", recommendation="BUY", confidence=0.8,
                reasoning="x", key_factors=(),
            ),
            InvestorOpinion(
                investor_name="B", recommendation="HOLD", confidence=0.8,
                reasoning="x", key_factors=(),
            ),
        ]
        v = self._verdict(after)
        # Tylko A zmienił (SELL→BUY); B bez zmian.
        assert v.opinion_shift(before) == 1

    def test_unknown_name_in_before_ignored(self):
        # Inwestor obecny tylko w "before" (nie ma go w werdykcie) nie liczy się.
        before = [
            InvestorOpinion(
                investor_name="Ghost", recommendation="SELL", confidence=0.8,
                reasoning="x", key_factors=(),
            )
        ]
        after = [
            InvestorOpinion(
                investor_name="A", recommendation="BUY", confidence=0.8,
                reasoning="x", key_factors=(),
            )
        ]
        v = self._verdict(after)
        assert v.opinion_shift(before) == 0


class TestCouncilInput:
    def test_stores_all_fields(self):
        data = CouncilInput(
            symbol="AAPL",
            current_price=Decimal("180.00"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["Apple beats earnings", "iPhone sales surge"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("185.00"),
        )
        assert data.symbol == "AAPL"
        assert data.ml_price_target == Decimal("185.00")

    def test_asset_type_defaults_to_stock(self):
        # Pole opcjonalne, wstecznie kompatybilne — domyślnie STOCK.
        data = CouncilInput(
            symbol="AAPL",
            current_price=Decimal("180.00"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["x"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("185.00"),
        )
        assert data.asset_type is AssetType.STOCK

    def test_asset_type_can_be_overridden(self):
        data = CouncilInput(
            symbol="BTC",
            current_price=Decimal("60000.00"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["x"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("61000.00"),
            asset_type=AssetType.CRYPTO,
        )
        assert data.asset_type is AssetType.CRYPTO

    def test_news_articles_still_coerced_to_tuple_with_asset_type(self):
        # Regresja: dodanie asset_type nie może zepsuć koercji tuple w __post_init__.
        data = CouncilInput(
            symbol="AAPL",
            current_price=Decimal("180.00"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["a", "b"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("185.00"),
            asset_type=AssetType.ETF,
        )
        assert isinstance(data.news_articles, tuple)
