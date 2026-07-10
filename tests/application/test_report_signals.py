from decimal import Decimal

import pytest

from src.application.report_models import SymbolResult, TradeSignal
from src.application.report_signals import build_trade_signals
from src.domain.calibration_curve import CalibrationBucket
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


class TestCalibratedConfidenceInSignalRanking:
    """#9 — historyczny hit-rate kubełka pewności koryguje `strength`, które
    steruje rankingiem "🎯 Najsilniejsze sygnały". Render-only, bez persystencji."""

    @staticmethod
    def _saved(symbol: str, confidence: float, change: str) -> SymbolResult:
        return SymbolResult(
            symbol=symbol,
            status="saved",
            trend="BULLISH",
            confidence_score=confidence,
            current_price=Decimal("100"),
            target_price=Decimal("100") * (Decimal("1") + Decimal(change)),
        )

    @staticmethod
    def _overconfident_bucket() -> list[CalibrationBucket]:
        # Kubełek 80-90%: deklarowana pewność 0.85, realny hit-rate 0.55.
        return [
            CalibrationBucket(
                lower=0.8, upper=0.9, count=100, mean_confidence=0.85, hit_rate=0.55
            )
        ]

    def test_without_buckets_strength_uses_raw_confidence(self) -> None:
        (signal,) = build_trade_signals([self._saved("AAPL", 0.85, "0.10")])

        assert signal.strength == pytest.approx(0.85 * 10.0)
        assert signal.calibrated_confidence is None

    def test_buckets_shrink_overconfident_strength(self) -> None:
        (signal,) = build_trade_signals(
            [self._saved("AAPL", 0.85, "0.10")],
            calibration_buckets=self._overconfident_bucket(),
        )

        # count=100, min_count=10 → waga 100/110 ≈ 0.909 → ~0.577
        assert signal.calibrated_confidence == pytest.approx(0.5773, abs=1e-3)
        assert signal.strength == pytest.approx(signal.calibrated_confidence * 10.0)
        assert signal.strength < 0.85 * 10.0

    def test_raw_confidence_is_still_reported_alongside(self) -> None:
        # Raport pokazuje "pewność LLM: 85% → skalibrowana historią: 58%",
        # więc surowa pewność NIE może zniknąć z sygnału.
        (signal,) = build_trade_signals(
            [self._saved("AAPL", 0.85, "0.10")],
            calibration_buckets=self._overconfident_bucket(),
        )

        assert signal.confidence == 0.85

    def test_calibration_can_reorder_the_ranking(self) -> None:
        # Rdzeń pozycji: kalibracja zmienia RANKING, nie tylko etykietę.
        # AAPL deklaruje 0.85 (kubełek przepewny), MSFT 0.65 (kubełek uczciwy).
        buckets = [
            CalibrationBucket(0.8, 0.9, count=100, mean_confidence=0.85, hit_rate=0.30),
            CalibrationBucket(0.6, 0.7, count=100, mean_confidence=0.65, hit_rate=0.70),
        ]
        results = [
            self._saved("AAPL", 0.85, "0.10"),
            self._saved("MSFT", 0.65, "0.10"),
        ]

        raw = build_trade_signals(results)
        calibrated = build_trade_signals(results, calibration_buckets=buckets)

        assert [s.symbol for s in raw] == ["AAPL", "MSFT"]
        assert [s.symbol for s in calibrated] == ["MSFT", "AAPL"]

    def test_empty_buckets_behave_like_no_history(self) -> None:
        (signal,) = build_trade_signals(
            [self._saved("AAPL", 0.85, "0.10")], calibration_buckets=[]
        )

        assert signal.strength == pytest.approx(0.85 * 10.0)
