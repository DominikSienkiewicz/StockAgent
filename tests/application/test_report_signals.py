from decimal import Decimal

from src.application.report_models import SymbolResult, TradeSignal
from src.application.report_signals import build_trade_signals
from src.domain.council import CouncilVerdict, InvestorOpinion


def _opinion(rec: str, confidence: float = 0.8) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name="X",
        recommendation=rec,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="...",
        key_factors=(),
    )


def _verdict(
    *,
    final: str = "BUY",
    consensus: float = 0.9,
    opinions: tuple[InvestorOpinion, ...] | None = None,
) -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation=final,  # type: ignore[arg-type]
        consensus_strength=consensus,
        summary="...",
        dissenting_views=(),
        investor_opinions=opinions if opinions is not None else (_opinion("BUY"),),
    )


def _saved_result(
    *,
    symbol: str = "AAPL",
    council: CouncilVerdict | None = None,
) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="saved",
        current_price=Decimal("100"),
        target_price=Decimal("110"),
        trend="BULLISH",
        confidence_score=0.8,
        council_verdict=council,
    )


class TestBuildTradeSignalsSizeBand:
    """Q7 — build_trade_signals dokleja sugerowaną bandę pozycji do sygnału."""

    def test_signal_has_no_band_without_council(self):
        # Brak werdyktu rady → brak danych do sizingu → size_band None.
        signals = build_trade_signals([_saved_result(council=None)])
        assert len(signals) == 1
        assert signals[0].size_band is None

    def test_strong_consensus_good_hitrate_gives_full_band(self):
        council = _verdict(consensus=0.9, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals(
            [_saved_result(council=council)], hit_rate=0.7
        )
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "full"

    def test_weak_consensus_gives_starter_band(self):
        council = _verdict(consensus=0.4, opinions=(_opinion("BUY"), _opinion("SELL")))
        signals = build_trade_signals(
            [_saved_result(council=council)], hit_rate=0.7
        )
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "starter"

    def test_unknown_hitrate_is_conservative(self):
        # Cold-start: rada przekonana, ale brak track recordu → nie pełna banda.
        council = _verdict(consensus=0.95, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals([_saved_result(council=council)])
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier != "full"

    def test_high_dissent_pulls_band_down(self):
        # Werdykt BUY, ale połowa rady głosuje SELL → wysoki dissent → starter.
        council = _verdict(
            final="BUY",
            consensus=0.9,
            opinions=(_opinion("BUY"), _opinion("SELL"), _opinion("SELL")),
        )
        signals = build_trade_signals(
            [_saved_result(council=council)], hit_rate=0.7
        )
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "starter"

    def test_default_size_band_is_none_on_dto(self):
        # Wsteczna kompatybilność — ręcznie zbudowany TradeSignal bez bandy.
        sig = TradeSignal(
            symbol="AAPL",
            direction="KUP",
            confidence=0.8,
            expected_change=Decimal("0.1"),
            strength=8.0,
            current_price=Decimal("100"),
            target_price=Decimal("110"),
        )
        assert sig.size_band is None
