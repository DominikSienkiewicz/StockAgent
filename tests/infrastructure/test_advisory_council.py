# tests/infrastructure/test_advisory_council.py
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import AdvisoryCouncilPort, LLMPort
from src.domain.council import CouncilInput, CouncilVerdict, InvestorPersona
from src.domain.value_objects import AssetType
from src.infrastructure.adapters.advisory_council import (
    COUNCIL_MAX_PERSONAS,
    LLMAdvisoryCouncil,
    _parse_opinion,
)

ALL_INVESTORS = [
    "Warren Buffett", "Benjamin Graham", "George Soros", "Peter Lynch",
    "Ray Dalio", "Charlie Munger", "Philip Fisher", "Paul Tudor Jones",
    "Bill Gross", "Jesse Livermore",
    "Cathie Wood", "Michael Burry", "Howard Marks",
    "Stanley Druckenmiller", "Joel Greenblatt",
]
COUNCIL_SIZE = len(ALL_INVESTORS)

_TEST_PERSONAS = tuple(
    InvestorPersona(name=name, style=f"Test style for {name}. " * 3)
    for name in ALL_INVESTORS
)


def _make_input() -> CouncilInput:
    return CouncilInput(
        symbol="AAPL",
        current_price=Decimal("180.00"),
        price_delta_pct=Decimal("3.5"),
        sentiment_score=0.6,
        news_articles=["Apple beats earnings"],
        llm_trend="BULLISH",
        llm_confidence=0.82,
        ml_price_target=Decimal("185.00"),
    )


def _opinion_json(rec: str = "BUY") -> dict:
    return {
        "recommendation": rec,
        "confidence": 0.8,
        "reasoning": "Silne fundamenty.",
        "key_factors": ["moat", "FCF", "wzrost"],
    }


def _verdict_json() -> dict:
    return {
        "final_recommendation": "BUY",
        "consensus_strength": 0.73,
        "summary": "Większość radzi kupować.",
        "dissenting_views": ["Soros: zbyt ryzykowne"],
    }


@pytest.fixture
def llm_port() -> Mock:
    mock = Mock(spec=LLMPort)
    mock.analyze.return_value = _opinion_json()
    return mock


@pytest.fixture
def council(llm_port: Mock) -> LLMAdvisoryCouncil:
    # max_personas podniesione ponad rozmiar rady testowej (15 > domyślny cap 12),
    # żeby testy fan-outu sprawdzały wywołanie KAŻDEJ persony bez ucinania.
    return LLMAdvisoryCouncil(
        llm_port=llm_port,
        personas=_TEST_PERSONAS,
        max_personas=COUNCIL_SIZE,
    )


class TestLLMAdvisoryCouncilIsPort:
    def test_implements_port(self, council: LLMAdvisoryCouncil):
        assert isinstance(council, AdvisoryCouncilPort)


class TestAnalyzeCallCount:
    def _side_effect_factory(self, llm_port: Mock):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return _opinion_json() if call_count <= COUNCIL_SIZE else _verdict_json()
        llm_port.analyze.side_effect = side_effect

    def test_calls_llm_once_per_investor_plus_chairman(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        self._side_effect_factory(llm_port)
        council.analyze("AAPL", _make_input())
        assert llm_port.analyze.call_count == COUNCIL_SIZE + 1

    def test_returns_council_verdict(self, council: LLMAdvisoryCouncil, llm_port: Mock):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        assert isinstance(result, CouncilVerdict)
        assert result.final_recommendation == "BUY"

    def test_verdict_has_opinion_per_investor(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == COUNCIL_SIZE

    def test_opinion_investor_names_all_present(self, council: LLMAdvisoryCouncil, llm_port: Mock):
        self._side_effect_factory(llm_port)
        result = council.analyze("AAPL", _make_input())
        names = {op.investor_name for op in result.investor_opinions}
        assert names == set(ALL_INVESTORS)


class TestFallbacks:
    def test_missing_recommendation_defaults_to_hold(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= COUNCIL_SIZE:
                return {"confidence": 0.5, "reasoning": "ok", "key_factors": []}
            return _verdict_json()
        llm_port.analyze.side_effect = side_effect
        result = council.analyze("AAPL", _make_input())
        for op in result.investor_opinions:
            assert op.recommendation == "HOLD"

    def test_missing_final_recommendation_uses_votes_not_chairman(
        self, council: LLMAdvisoryCouncil, llm_port: Mock
    ):
        call_count = 0
        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= COUNCIL_SIZE:
                return _opinion_json()
            return {"consensus_strength": 0.5, "summary": "ok", "dissenting_views": []}
        llm_port.analyze.side_effect = side_effect
        result = council.analyze("AAPL", _make_input())
        # Wszystkie opinie to BUY (domyślny mock) → werdykt z głosów to BUY,
        # niezależnie od tego, co (nie) zwrócił chairman.
        assert result.final_recommendation == "BUY"


class TestConsensusDerivedFromVotes:
    """Finding #3: consensus_strength + final_recommendation muszą pochodzić
    z REALNYCH głosów inwestorów, nie z liczby wymyślonej przez chairman-LLM."""

    def _council_3_personas(self, llm_port: Mock) -> LLMAdvisoryCouncil:
        personas = tuple(
            InvestorPersona(name=n, style=f"Styl testowy dla {n}. " * 3)
            for n in ("Inv A", "Inv B", "Inv C", "Inv D")
        )
        return LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, max_personas=10
        )

    def test_unanimous_buy_survives_chairman_failure(self, llm_port: Mock):
        # 4 inwestorów głosuje BUY; wywołanie chairmana RZUCA wyjątek.
        # Mimo to werdykt musi być BUY z wysokim consensus_strength — NIE 0.5/HOLD.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            raise RuntimeError("chairman down")

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.final_recommendation == "BUY"
        assert result.consensus_strength == pytest.approx(1.0)

    def test_chairman_number_is_ignored_for_decision(self, llm_port: Mock):
        # Chairman zwraca SELL / 0.99, ale wszyscy inwestorzy głosują BUY.
        # Decyzja należy do głosów → BUY, a strength = z głosów (1.0), nie 0.99.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            return {
                "final_recommendation": "SELL",
                "consensus_strength": 0.99,
                "summary": "Chairman się myli.",
                "dissenting_views": [],
            }

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.final_recommendation == "BUY"
        assert result.consensus_strength == pytest.approx(1.0)

    def test_chairman_summary_still_used(self, llm_port: Mock):
        # Tekstowe pola (summary, dissenting_views) NADAL pochodzą od chairmana.
        n_personas = 4
        call_count = 0

        def side_effect(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= n_personas:
                return _opinion_json("BUY")
            return _verdict_json()

        llm_port.analyze.side_effect = side_effect
        council = self._council_3_personas(llm_port)
        result = council.analyze("AAPL", _make_input())

        assert result.summary == "Większość radzi kupować."
        # dissenting_views to teraz tuple (frozen value object — finding #32).
        assert result.dissenting_views == ("Soros: zbyt ryzykowne",)


class TestConfidenceClamp:
    """Finding #20: confidence dostarczane przez LLM musi być przycięte do
    [0,1] na granicy parsowania. Model dryfujący na skalę 0-100 nie może
    podbijać każdej opinii do HIGH (progi confidence_label to 0.75/0.5)."""

    def test_confidence_above_one_clamped_to_one(self) -> None:
        # Dryf na skalę 0-100: confidence=85 → musi spaść do 1.0, nie zostać 85.
        opinion = _parse_opinion("Inv", {"recommendation": "BUY", "confidence": 85})
        assert opinion.confidence == 1.0

    def test_confidence_below_zero_clamped_to_zero(self) -> None:
        opinion = _parse_opinion("Inv", {"recommendation": "BUY", "confidence": -3})
        assert opinion.confidence == 0.0

    def test_confidence_in_range_passes_through(self) -> None:
        opinion = _parse_opinion("Inv", {"recommendation": "BUY", "confidence": 0.7})
        assert opinion.confidence == 0.7

    def test_confidence_missing_defaults_to_half(self) -> None:
        opinion = _parse_opinion("Inv", {"recommendation": "BUY"})
        assert opinion.confidence == 0.5

    def test_high_confidence_label_not_triggered_by_drift(self) -> None:
        # Regresja na sedno findingu: 85 sklamowane do 1.0 → label HIGH legalnie,
        # ale wartość liczbowa nie korumpuje już dalszych obliczeń (np. ważenia).
        opinion = _parse_opinion("Inv", {"recommendation": "BUY", "confidence": 85})
        assert 0.0 <= opinion.confidence <= 1.0


class TestPersonaCap:
    """Finding #11: liczba wywołań LLM ograniczona górnym capem person."""

    def _personas(self, count: int) -> tuple[InvestorPersona, ...]:
        return tuple(
            InvestorPersona(name=f"Inv {i:02d}", style=f"Styl testowy {i}. " * 3)
            for i in range(count)
        )

    def test_truncates_above_cap_and_warns(self, llm_port: Mock, caplog):
        over = COUNCIL_MAX_PERSONAS + 3
        personas = self._personas(over)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)

        with caplog.at_level("WARNING"):
            result = council.analyze("AAPL", _make_input())

        # Tylko `cap` opinii inwestorów + jeden chairman.
        assert llm_port.analyze.call_count == COUNCIL_MAX_PERSONAS + 1
        assert len(result.investor_opinions) == COUNCIL_MAX_PERSONAS
        # Ostrzeżenie wymienia odrzucone persony (te ponad capem).
        warning_text = " ".join(r.getMessage() for r in caplog.records)
        dropped_names = [p.name for p in personas[COUNCIL_MAX_PERSONAS:]]
        for name in dropped_names:
            assert name in warning_text

    def test_at_cap_unchanged_no_warning(self, llm_port: Mock, caplog):
        personas = self._personas(COUNCIL_MAX_PERSONAS)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)

        with caplog.at_level("WARNING"):
            result = council.analyze("AAPL", _make_input())

        assert len(result.investor_opinions) == COUNCIL_MAX_PERSONAS
        assert "persona" not in " ".join(
            r.getMessage().lower() for r in caplog.records
        )

    def test_below_cap_unchanged(self, llm_port: Mock):
        personas = self._personas(5)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == 5

    def test_custom_cap_param(self, llm_port: Mock):
        personas = self._personas(8)
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, max_personas=3
        )
        result = council.analyze("AAPL", _make_input())
        assert len(result.investor_opinions) == 3

    def test_default_constant_value_is_12(self):
        assert COUNCIL_MAX_PERSONAS == 12

    def test_backward_compatible_signature(self, llm_port: Mock):
        # Istniejące wywołanie bez max_personas musi nadal działać.
        personas = self._personas(3)
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        assert isinstance(council, LLMAdvisoryCouncil)


def _personas(names: tuple[str, ...]) -> tuple[InvestorPersona, ...]:
    return tuple(
        InvestorPersona(name=n, style=f"Styl testowy dla {n}. " * 3) for n in names
    )


class TestAdaptivePersonaWeighting:
    """Feature #3: weights_provider waży głosy person ich historyczną
    trafnością. Brak providera → werdykt jak dziś (wagi 1.0)."""

    def _opinion(self, rec: str, conf: float = 0.5) -> dict:
        return {
            "recommendation": rec,
            "confidence": conf,
            "reasoning": "...",
            "key_factors": [],
        }

    def test_provider_called_with_asset_type(self, llm_port: Mock):
        llm_port.analyze.return_value = self._opinion("BUY")
        captured: list[AssetType] = []

        def provider(asset_type: AssetType) -> dict[str, float]:
            captured.append(asset_type)
            return {}

        personas = _personas(("A", "B"))
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, weights_provider=provider
        )
        data = CouncilInput(
            symbol="BTC",
            current_price=Decimal("60000"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["x"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("61000"),
            asset_type=AssetType.CRYPTO,
        )
        council.analyze("BTC", data)
        assert AssetType.CRYPTO in captured

    def test_heavy_persona_swings_verdict(self, llm_port: Mock):
        # Round-1: "Star" → BUY, dwie inne → SELL, równe confidence 0.5.
        # Bez wag: SELL (headcount/masa). Z 5× wagą dla Star: BUY wygrywa.
        def side_effect(prompt: str) -> dict:
            if "Star" in prompt:
                return self._opinion("BUY")
            if "B persona" in prompt or "C persona" in prompt:
                return self._opinion("SELL")
            # chairman
            return _verdict_json()

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("Star", "B persona", "C persona"))

        def provider(asset_type: AssetType) -> dict[str, float]:
            return {"Star": 5.0, "B persona": 1.0, "C persona": 1.0}

        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, weights_provider=provider
        )
        result = council.analyze("AAPL", _make_input())
        assert result.final_recommendation == "BUY"

    def test_no_provider_unchanged(self, llm_port: Mock):
        # Bez weights_provider werdykt wynika z czystej masy confidence.
        def side_effect(prompt: str) -> dict:
            if "Star" in prompt:
                return self._opinion("BUY")
            if "B persona" in prompt or "C persona" in prompt:
                return self._opinion("SELL")
            return _verdict_json()

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("Star", "B persona", "C persona"))
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        result = council.analyze("AAPL", _make_input())
        # 2×SELL vs 1×BUY przy równym confidence → SELL.
        assert result.final_recommendation == "SELL"

    def test_provider_exception_falls_back_to_unweighted(self, llm_port: Mock):
        # Wyjątek w providerze nie może wywalić rady — degradujemy do wag None.
        def side_effect(prompt: str) -> dict:
            if "Star" in prompt:
                return self._opinion("BUY")
            if "B persona" in prompt or "C persona" in prompt:
                return self._opinion("SELL")
            return _verdict_json()

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("Star", "B persona", "C persona"))

        def provider(asset_type: AssetType) -> dict[str, float]:
            raise RuntimeError("weights store down")

        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, weights_provider=provider
        )
        result = council.analyze("AAPL", _make_input())
        # Provider padł → wagi None → headcount/masa wygrywa SELL.
        assert result.final_recommendation == "SELL"


class TestDebateMode:
    """Feature #1: jedna runda rewizji, gdy round-1 jest podzielone
    (jednocześnie BUY i SELL). N round-1 + N rewizji + 1 chairman = 2N+1."""

    def _split_side_effect(self, llm_port: Mock, n: int):
        """Round-1 podzielone: persona 0 → BUY, persona 1 → SELL, reszta → HOLD.
        Po rewizji wszyscy → BUY. Chairman na końcu."""
        round1_calls = {"n": 0}

        def side_effect(prompt: str) -> dict:
            # Rewizja rozpoznawana po bloku peerów (unikalny marker).
            if "<opinie_peerow>" in prompt:
                return _opinion_json("BUY")
            # Chairman rozpoznawany po schemacie werdyktu.
            if "final_recommendation" in prompt:
                return _verdict_json()
            # Round-1 investor.
            idx = round1_calls["n"]
            round1_calls["n"] += 1
            if idx == 0:
                return _opinion_json("BUY")
            if idx == 1:
                return _opinion_json("SELL")
            return _opinion_json("HOLD")

        llm_port.analyze.side_effect = side_effect

    def test_split_round1_triggers_one_revision_round(self, llm_port: Mock):
        n = 3
        self._split_side_effect(llm_port, n)
        personas = _personas(("A", "B", "C"))
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, debate_mode=True
        )
        council.analyze("AAPL", _make_input())
        # N round-1 + N rewizji + 1 chairman = 2N+1.
        assert llm_port.analyze.call_count == 2 * n + 1

    def test_verdict_reflects_revised_votes(self, llm_port: Mock):
        n = 3
        self._split_side_effect(llm_port, n)
        personas = _personas(("A", "B", "C"))
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, debate_mode=True
        )
        result = council.analyze("AAPL", _make_input())
        # Po rewizji wszyscy głosują BUY → werdykt BUY (round-1 był split).
        assert result.final_recommendation == "BUY"
        # Werdykt zawiera zrewidowane opinie (wszystkie BUY), nie round-1.
        assert all(op.recommendation == "BUY" for op in result.investor_opinions)

    def test_consensus_round1_no_revision(self, llm_port: Mock):
        # Round-1 jednomyślne BUY (brak SELL) → brak rewizji: N + 1.
        n = 3

        def side_effect(prompt: str) -> dict:
            if "final_recommendation" in prompt:
                return _verdict_json()
            return _opinion_json("BUY")

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("A", "B", "C"))
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, debate_mode=True
        )
        council.analyze("AAPL", _make_input())
        assert llm_port.analyze.call_count == n + 1

    def test_debate_off_never_revises_even_when_split(self, llm_port: Mock):
        # debate_mode=False: nawet split round-1 nie wywołuje rewizji.
        n = 3
        round1_calls = {"n": 0}

        def side_effect(prompt: str) -> dict:
            if "<opinie_peerow>" in prompt:
                raise AssertionError("revision must not run when debate is off")
            if "final_recommendation" in prompt:
                return _verdict_json()
            idx = round1_calls["n"]
            round1_calls["n"] += 1
            if idx == 0:
                return _opinion_json("BUY")
            if idx == 1:
                return _opinion_json("SELL")
            return _opinion_json("HOLD")

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("A", "B", "C"))
        council = LLMAdvisoryCouncil(
            llm_port=llm_port, personas=personas, debate_mode=False
        )
        council.analyze("AAPL", _make_input())
        # N round-1 + 1 chairman, zero rewizji.
        assert llm_port.analyze.call_count == n + 1

    def test_default_debate_mode_is_off(self, llm_port: Mock):
        # Domyślnie debate_mode wyłączone (wsteczna kompatybilność).
        n = 3
        round1_calls = {"n": 0}

        def side_effect(prompt: str) -> dict:
            if "<opinie_peerow>" in prompt:
                raise AssertionError("revision must not run by default")
            if "final_recommendation" in prompt:
                return _verdict_json()
            idx = round1_calls["n"]
            round1_calls["n"] += 1
            return _opinion_json("BUY" if idx == 0 else "SELL" if idx == 1 else "HOLD")

        llm_port.analyze.side_effect = side_effect
        personas = _personas(("A", "B", "C"))
        council = LLMAdvisoryCouncil(llm_port=llm_port, personas=personas)
        council.analyze("AAPL", _make_input())
        assert llm_port.analyze.call_count == n + 1
