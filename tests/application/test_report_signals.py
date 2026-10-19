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


class TestColdStartSignalRanking:
    """WIA-13 (G03): bez wag XGBoost target == bieżąca cena, więc
    pewność × |expected_change| zeruje każdy sygnał i ranking staje się
    kolejnością przetwarzania. Przy cold-starcie siła = (skalibrowana) pewność."""

    @staticmethod
    def _cold(symbol: str, confidence: float) -> SymbolResult:
        return SymbolResult(
            symbol=symbol,
            status="saved",
            trend="BULLISH",
            confidence_score=confidence,
            current_price=Decimal("100"),
            target_price=Decimal("100"),
            ml_cold_start=True,
        )

    def test_ranking_orders_by_calibrated_confidence_when_ml_target_equals_price(
        self,
    ) -> None:
        signals = build_trade_signals([self._cold("AAPL", 0.6), self._cold("MSFT", 0.9)])

        assert [s.symbol for s in signals] == ["MSFT", "AAPL"]
        assert [s.strength for s in signals] == [pytest.approx(0.9), pytest.approx(0.6)]

    def test_cold_start_strength_uses_calibrated_confidence(self) -> None:
        buckets = [
            CalibrationBucket(0.8, 0.9, count=100, mean_confidence=0.85, hit_rate=0.30),
            CalibrationBucket(0.6, 0.7, count=100, mean_confidence=0.65, hit_rate=0.70),
        ]

        signals = build_trade_signals(
            [self._cold("AAPL", 0.85), self._cold("MSFT", 0.65)],
            calibration_buckets=buckets,
        )

        assert [s.symbol for s in signals] == ["MSFT", "AAPL"]
        assert signals[0].strength == pytest.approx(signals[0].calibrated_confidence)

    def test_cold_start_signal_is_flagged(self) -> None:
        (signal,) = build_trade_signals([self._cold("AAPL", 0.7)])

        assert signal.ml_cold_start is True

    def test_model_backed_signals_rank_before_cold_start_ones(self) -> None:
        # Siły w dwóch skalach (ruch × pewność vs sama pewność) są nieporównywalne.
        backed = SymbolResult(
            symbol="NVDA",
            status="saved",
            trend="BULLISH",
            confidence_score=0.5,
            current_price=Decimal("100"),
            target_price=Decimal("100.5"),
        )

        signals = build_trade_signals([self._cold("AAPL", 0.95), backed])

        assert [s.symbol for s in signals] == ["NVDA", "AAPL"]
        assert signals[0].ml_cold_start is False


class TestCalibrationBucketCountOnSignal:
    """WIA-12: kubełki z małą próbą skaczą, więc mail pokazuje n kubełka obok
    pewności skalibrowanej. Sygnał niesie je z tego samego kubełka, którym
    `calibrated_confidence` skorygowało pewność."""

    @staticmethod
    def _saved(confidence: float) -> SymbolResult:
        return SymbolResult(
            symbol="AAPL",
            status="saved",
            trend="BULLISH",
            confidence_score=confidence,
            current_price=Decimal("100"),
            target_price=Decimal("110"),
        )

    def test_signal_carries_the_count_of_its_calibration_bucket(self) -> None:
        buckets = [CalibrationBucket(0.8, 0.9, count=12, mean_confidence=0.85, hit_rate=0.5)]

        (signal,) = build_trade_signals([self._saved(0.85)], calibration_buckets=buckets)

        assert signal.calibration_bucket_count == 12

    def test_no_matching_bucket_means_an_empty_bucket(self) -> None:
        buckets = [CalibrationBucket(0.6, 0.7, count=40, mean_confidence=0.65, hit_rate=0.6)]

        (signal,) = build_trade_signals([self._saved(0.85)], calibration_buckets=buckets)

        assert signal.calibrated_confidence == pytest.approx(0.85)
        assert signal.calibration_bucket_count == 0

    def test_without_calibration_there_is_no_bucket_count(self) -> None:
        (signal,) = build_trade_signals([self._saved(0.85)])

        assert signal.calibration_bucket_count is None


def test_divergence_signal_writes_the_numbers_the_polish_way() -> None:
    # MAIL-58: opis sygnału ryzyka trafia do maila — „3,00%” i „-0,45”, nie z kropką.
    from src.application.report_signals import detect_risk_signals

    result = SymbolResult(
        symbol="NVDA", status="saved", delta=Decimal("0.03"), sentiment_score=-0.45
    )

    descriptions = [s.description for s in detect_risk_signals([result])]

    assert "cena rośnie 3,00% ale sentyment -0,45 -> możliwa korekta lub pump" in descriptions


class TestBuildTradeSignalsPortfolioRisk:
    """PORT-04: ryzyko realnego portfela (VaR, obsunięcie, hedge) trafia do
    `suggest_band` i może bandę wyłącznie zwęzić."""

    @staticmethod
    def _tier(**risk: object) -> str:
        council = _verdict(consensus=0.9, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals(
            [_saved_result(council=council)], hit_rate=0.7, **risk  # type: ignore[arg-type]
        )
        band = signals[0].size_band
        assert band is not None
        return band.tier

    def test_without_risk_the_band_is_unchanged(self) -> None:
        assert self._tier() == "full"

    def test_high_var_lowers_the_band(self) -> None:
        assert self._tier(portfolio_var=0.05) == "standard"

    def test_deep_drawdown_forces_the_starter_band(self) -> None:
        assert self._tier(drawdown=0.2) == "starter"

    def test_ineffective_hedge_lowers_the_band(self) -> None:
        assert self._tier(hedge_effective=False) == "standard"
