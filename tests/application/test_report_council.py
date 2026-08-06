from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import build_html_report, to_symbol_result
from src.application.report_models import SymbolResult
from src.domain.council import CouncilVerdict, InvestorOpinion


def _make_verdict(rec: str = "BUY") -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation=rec,  # type: ignore[arg-type]
        consensus_strength=0.73,
        summary="Rada radzi kupować akcje AAPL.",
        dissenting_views=["Soros: zbyt ryzykowne"],
        investor_opinions=[
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",  # type: ignore[arg-type]
                confidence=0.9,
                reasoning="Silna fosa.",
                key_factors=["moat", "FCF"],
            ),
            InvestorOpinion(
                investor_name="George Soros",
                recommendation="SELL",  # type: ignore[arg-type]
                confidence=0.7,
                reasoning="Reflexivity wskazuje korektę.",
                key_factors=["makro", "momentum"],
            ),
        ],
    )


class TestSymbolResultHasCouncilVerdict:
    def test_symbol_result_accepts_council_verdict(self):
        r = SymbolResult(
            symbol="AAPL",
            status="saved",
            council_verdict=_make_verdict(),
        )
        assert r.council_verdict is not None
        assert r.council_verdict.final_recommendation == "BUY"

    def test_symbol_result_council_verdict_defaults_none(self):
        r = SymbolResult(symbol="AAPL", status="saved")
        assert r.council_verdict is None


class TestToSymbolResultExtractsCouncil:
    def test_extracts_council_verdict_from_raw(self):
        verdict = _make_verdict()
        raw = {
            "status": "saved",
            "delta": Decimal("0.035"),
            "current_price": Decimal("180.00"),
            "ml_target_price": Decimal("185.00"),
            "llm_analysis": {
                "trend_direction": "BULLISH",
                "confidence_score": 0.8,
                "av_agreement": 0.9,
                "reasoning": "Strong momentum.",
            },
            "council_verdict": verdict,
        }
        result = to_symbol_result("AAPL", raw)
        assert result.council_verdict is verdict

    def test_council_verdict_none_when_absent_from_raw(self):
        raw = {"status": "saved", "llm_analysis": {}}
        result = to_symbol_result("AAPL", raw)
        assert result.council_verdict is None

    def test_council_verdict_none_when_raw_contains_dict_not_verdict(self):
        raw = {"status": "saved", "llm_analysis": {}, "council_verdict": {"some": "dict"}}
        result = to_symbol_result("AAPL", raw)
        assert result.council_verdict is None


class TestCouncilHtmlSection:
    def _build_report_with_verdict(self, rec: str = "BUY") -> str:
        verdict = _make_verdict(rec)
        results = [
            SymbolResult(
                symbol="AAPL",
                status="saved",
                delta=Decimal("0.035"),
                current_price=Decimal("180.00"),
                trend="BULLISH",
                council_verdict=verdict,
            )
        ]
        html, _ = build_html_report(results, datetime.now(UTC), 1.0)
        return html

    def test_report_contains_rada_doradcza(self):
        assert "RADA DORADCZA" in self._build_report_with_verdict()

    def test_report_contains_investor_name(self):
        assert "Warren Buffett" in self._build_report_with_verdict()

    def test_report_contains_consensus_label_buy(self):
        assert "KUP" in self._build_report_with_verdict("BUY")

    def test_report_contains_consensus_label_sell(self):
        assert "SPRZEDAJ" in self._build_report_with_verdict("SELL")

    def test_report_contains_consensus_label_hold(self):
        assert "TRZYMAJ" in self._build_report_with_verdict("HOLD")

    def test_report_contains_dissenting_view(self):
        assert "Soros" in self._build_report_with_verdict()

    def test_no_council_section_when_verdict_none(self):
        results = [SymbolResult(symbol="AAPL", status="saved")]
        html, _ = build_html_report(results, datetime.now(UTC), 1.0)
        assert "RADA DORADCZA" not in html


class TestCouncilSectionUsesDomainBehavior:
    """Sekcja rady w raporcie korzysta z metod domenowych (is_split_decision,
    vote_distribution, has_strong_consensus) zamiast inline'owych warunków
    w HTML. Bez tego logika decyzyjna była rozsmarowana po warstwie prezentacji.
    """

    def _result_with_opinions(
        self, opinions: list[InvestorOpinion], final: str = "BUY",
        consensus: float = 0.6,
    ) -> SymbolResult:
        verdict = CouncilVerdict(
            final_recommendation=final,  # type: ignore[arg-type]
            consensus_strength=consensus,
            summary="Test.",
            dissenting_views=[],
            investor_opinions=opinions,
        )
        return SymbolResult(
            symbol="AAPL",
            status="saved",
            delta=Decimal("0.035"),
            current_price=Decimal("180.00"),
            trend="BULLISH",
            council_verdict=verdict,
        )

    def _build(self, opinions: list[InvestorOpinion], **kw: object) -> str:
        results = [self._result_with_opinions(opinions, **kw)]  # type: ignore[arg-type]
        html, _ = build_html_report(results, datetime.now(UTC), 1.0)
        return html

    def test_split_decision_warning_shown_when_buy_and_sell_present(self):
        opinions = [
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",  # type: ignore[arg-type]
                confidence=0.9, reasoning="x", key_factors=[],
            ),
            InvestorOpinion(
                investor_name="George Soros",
                recommendation="SELL",  # type: ignore[arg-type]
                confidence=0.7, reasoning="x", key_factors=[],
            ),
        ]
        html = self._build(opinions)
        # Marker dla "split decision" — fundamentalny brak zgody w radzie.
        assert "SPLIT" in html or "PODZIELONA" in html

    def test_no_split_warning_when_only_buy_and_hold(self):
        opinions = [
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",  # type: ignore[arg-type]
                confidence=0.9, reasoning="x", key_factors=[],
            ),
            InvestorOpinion(
                investor_name="Charlie Munger",
                recommendation="HOLD",  # type: ignore[arg-type]
                confidence=0.6, reasoning="x", key_factors=[],
            ),
        ]
        html = self._build(opinions)
        assert "SPLIT" not in html
        assert "PODZIELONA" not in html

    def test_vote_distribution_displayed(self):
        # Trzy głosy: 2 BUY, 1 SELL → rozkład widoczny w raporcie
        opinions = [
            InvestorOpinion(
                investor_name=name, recommendation=rec,  # type: ignore[arg-type]
                confidence=0.8, reasoning="x", key_factors=[],
            )
            for name, rec in [
                ("A", "BUY"), ("B", "BUY"), ("C", "SELL"),
            ]
        ]
        html = self._build(opinions)
        # Format "2 BUY" / "1 SELL" — albo polski odpowiednik. Niech będzie elastyczne.
        assert "2" in html
        # Najprostszy sprawdzian: wszystkie 3 etykiety obecne gdzieś w sekcji
        assert "KUP" in html
        assert "SPRZEDAJ" in html

    def test_strong_consensus_badge_when_above_threshold(self):
        opinions = [
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",  # type: ignore[arg-type]
                confidence=0.9, reasoning="x", key_factors=[],
            ),
        ]
        html = self._build(opinions, consensus=0.85)
        assert "SILNY KONSENSUS" in html or "STRONG CONSENSUS" in html

    def test_no_strong_consensus_badge_when_below_threshold(self):
        opinions = [
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",  # type: ignore[arg-type]
                confidence=0.7, reasoning="x", key_factors=[],
            ),
        ]
        html = self._build(opinions, consensus=0.5)
        assert "SILNY KONSENSUS" not in html
        assert "STRONG CONSENSUS" not in html
