import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application.report_models import SymbolResult, TradeSignal
from src.application.report_signals import (
    build_trade_signals,
    forecast_horizon_label,
    market_status,
)
from src.domain.calibration_curve import CalibrationBucket
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.isotonic_calibration import (
    CalibrationClass,
    ClassIsotonicSelection,
    IsotonicMap,
    IsotonicSelection,
)


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

    def test_signal_has_no_band_without_council(self) -> None:
        # Brak werdyktu rady → brak danych do sizingu → size_band None.
        signals = build_trade_signals([_saved_result(council=None)])
        assert len(signals) == 1
        assert signals[0].size_band is None

    def test_strong_consensus_good_hitrate_gives_full_band(self) -> None:
        council = _verdict(consensus=0.9, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals([_saved_result(council=council)], hit_rate=0.7)
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "full"

    def test_weak_consensus_gives_starter_band(self) -> None:
        council = _verdict(consensus=0.4, opinions=(_opinion("BUY"), _opinion("SELL")))
        signals = build_trade_signals([_saved_result(council=council)], hit_rate=0.7)
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "starter"

    def test_unknown_hitrate_is_conservative(self) -> None:
        # Cold-start: rada przekonana, ale brak track recordu → nie pełna banda.
        council = _verdict(consensus=0.95, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals([_saved_result(council=council)])
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier != "full"

    def test_high_dissent_pulls_band_down(self) -> None:
        # Werdykt BUY, ale połowa rady głosuje SELL → wysoki dissent → starter.
        council = _verdict(
            final="BUY",
            consensus=0.9,
            opinions=(_opinion("BUY"), _opinion("SELL"), _opinion("SELL")),
        )
        signals = build_trade_signals([_saved_result(council=council)], hit_rate=0.7)
        assert signals[0].size_band is not None
        assert signals[0].size_band.tier == "starter"

    def test_default_size_band_is_none_on_dto(self) -> None:
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
            CalibrationBucket(lower=0.8, upper=0.9, count=100, mean_confidence=0.85, hit_rate=0.55)
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
        assert signal.calibrated_confidence is not None
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
        (signal,) = build_trade_signals([self._saved("AAPL", 0.85, "0.10")], calibration_buckets=[])

        assert signal.strength == pytest.approx(0.85 * 10.0)


class TestIsotonicSignalCalibration:
    """ML-28: mapa izotoniczna, która wygrała poza próbą, zastępuje w rankingu
    shrinkage kubełkowy — monotonicznie, więc wyższa pewność nie spada niżej."""

    _saved = staticmethod(TestCalibratedConfidenceInSignalRanking._saved)

    def test_isotonic_map_replaces_the_bucket_shrinkage(self) -> None:
        from src.domain.isotonic_calibration import IsotonicMap

        isotonic = IsotonicMap(knots=(0.6, 0.9), values=(0.55, 0.65))

        (signal,) = build_trade_signals(
            [self._saved("AAPL", 0.85, "0.10")],
            calibration_buckets=TestCalibratedConfidenceInSignalRanking._overconfident_bucket(),
            isotonic_calibration=isotonic,
        )

        assert signal.calibrated_confidence == pytest.approx(isotonic.apply(0.85))
        assert signal.strength == pytest.approx(isotonic.apply(0.85) * 10.0)

    def test_isotonic_map_keeps_the_confidence_order(self) -> None:
        from src.domain.isotonic_calibration import IsotonicMap

        # Kubełki odwracają porządek 0.85 vs 0.65; mapa izotoniczna nie może.
        buckets = [
            CalibrationBucket(0.8, 0.9, count=100, mean_confidence=0.85, hit_rate=0.30),
            CalibrationBucket(0.6, 0.7, count=100, mean_confidence=0.65, hit_rate=0.70),
        ]
        results = [self._saved("MSFT", 0.65, "0.10"), self._saved("AAPL", 0.85, "0.10")]

        ranked = build_trade_signals(
            results,
            calibration_buckets=buckets,
            isotonic_calibration=IsotonicMap(knots=(0.65, 0.85), values=(0.5, 0.5)),
        )

        assert [s.calibrated_confidence for s in ranked] == [0.5, 0.5]

    def test_without_buckets_the_isotonic_map_stays_unused(self) -> None:
        # WIA-38: bez kubełków (pewność nie odróżnia trafień) ranking idzie na surowej.
        from src.domain.isotonic_calibration import IsotonicMap

        (signal,) = build_trade_signals(
            [self._saved("AAPL", 0.85, "0.10")],
            isotonic_calibration=IsotonicMap(knots=(0.6, 0.9), values=(0.1, 0.2)),
        )

        assert signal.calibrated_confidence is None


class TestClassIsotonicSignalCalibration:
    """ML-54: sygnał krypto kalibruje mapa krypto, reszta mapa reszty; symbol bez
    klasy aktywa dostaje mapę globalną."""

    _CRYPTO_MAP = IsotonicMap(knots=(0.6, 0.9), values=(0.3, 0.4))
    _REST_MAP = IsotonicMap(knots=(0.6, 0.9), values=(0.6, 0.7))
    _GLOBAL_MAP = IsotonicMap(knots=(0.6, 0.9), values=(0.5, 0.5))

    @staticmethod
    def _decision(calibrator: IsotonicMap | None) -> IsotonicSelection:
        return IsotonicSelection(
            use_isotonic=calibrator is not None,
            calibrator=calibrator,
            brier_isotonic=None,
            brier_raw=None,
            brier_buckets=None,
            n_fit=40,
            n_eval=40,
            reason="test",
        )

    def _selection(self) -> ClassIsotonicSelection:
        return ClassIsotonicSelection(
            global_selection=self._decision(self._GLOBAL_MAP),
            class_selections={
                CalibrationClass.CRYPTO: self._decision(self._CRYPTO_MAP),
                CalibrationClass.NON_CRYPTO: self._decision(self._REST_MAP),
            },
        )

    @staticmethod
    def _saved_in_class(symbol: str, asset_class: str | None) -> SymbolResult:
        return dataclasses.replace(
            TestCalibratedConfidenceInSignalRanking._saved(symbol, 0.85, "0.10"),
            asset_class=asset_class,
        )

    def _calibrated_by_symbol(self, results: list[SymbolResult]) -> dict[str, float | None]:
        signals = build_trade_signals(
            results,
            calibration_buckets=TestCalibratedConfidenceInSignalRanking._overconfident_bucket(),
            isotonic_calibration=self._GLOBAL_MAP,
            isotonic_class_calibration=self._selection(),
        )
        return {signal.symbol: signal.calibrated_confidence for signal in signals}

    def test_each_signal_is_calibrated_with_the_map_of_its_asset_class(self) -> None:
        calibrated = self._calibrated_by_symbol(
            [self._saved_in_class("BTC", "CRYPTO"), self._saved_in_class("SPY", "ETF")]
        )

        assert calibrated == {
            "BTC": pytest.approx(self._CRYPTO_MAP.apply(0.85)),
            "SPY": pytest.approx(self._REST_MAP.apply(0.85)),
        }

    def test_signal_without_asset_class_takes_the_global_map(self) -> None:
        calibrated = self._calibrated_by_symbol([self._saved_in_class("AAPL", None)])

        assert calibrated == {"AAPL": pytest.approx(self._GLOBAL_MAP.apply(0.85))}


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

    assert "cena rośnie 3,00% ale sentyment -0,45 -> rozbieżność cena–sentyment" in descriptions


@pytest.mark.parametrize(
    ("delta", "sentiment"), [(Decimal("0.03"), -0.45), (Decimal("-0.03"), 0.45)]
)
def test_divergence_copy_is_descriptive_not_diagnostic(delta: Decimal, sentiment: float) -> None:
    # WIA-47: DIVERGENCE to niezweryfikowana heurystyka — mail opisuje rozbieżność
    # zamiast wróżyć „korektę lub pump” albo „okazję lub odbicie”.
    from src.application.report_signals import detect_risk_signals

    result = SymbolResult(symbol="NVDA", status="saved", delta=delta, sentiment_score=sentiment)

    (signal,) = detect_risk_signals([result])

    assert signal.type == "DIVERGENCE"
    assert signal.description.endswith("-> rozbieżność cena–sentyment")
    for diagnosis in ("pump", "korekta", "okazja", "odbicie"):
        assert diagnosis not in signal.description


def test_conflict_copy_is_descriptive_not_diagnostic() -> None:
    # WIA-47: niezgodność AV↔LLM nie jest zweryfikowana na wynikach (evaluate
    # --by risk_signal), więc mail opisuje rozbieżność zamiast diagnozować
    # „fake news / manipulację”.
    from src.application.report_signals import detect_risk_signals

    result = SymbolResult(symbol="AMD", status="saved", av_llm_agreement=0.25)

    (signal,) = detect_risk_signals([result])

    assert signal.type == "AV_LLM_CONFLICT"
    assert "zgoda=0,25" in signal.description
    assert "rozbieżne" in signal.description
    assert "fake news" not in signal.description
    assert "manipulacja" not in signal.description


class TestBuildTradeSignalsPortfolioRisk:
    """PORT-04: ryzyko realnego portfela (VaR, obsunięcie, hedge) trafia do
    `suggest_band` i może bandę wyłącznie zwęzić."""

    @staticmethod
    def _tier(**risk: object) -> str:
        council = _verdict(consensus=0.9, opinions=(_opinion("BUY"), _opinion("BUY")))
        signals = build_trade_signals(
            [_saved_result(council=council)],
            hit_rate=0.7,
            **risk,  # type: ignore[arg-type]
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


class TestMarketStatusCalendar:
    """ALR-15: status sesji w nagłówku maila zna święta i skrócone sesje NYSE.
    Wcześniej był „uproszczony — bez świąt”, więc w Boże Narodzenie mail pisał „Otwarta”."""

    def test_christmas_is_a_named_holiday_not_an_open_session(self) -> None:
        # 2026-12-25 15:00 UTC = 10:00 ET, piątek — dawniej „🟢 Otwarta”.
        status = market_status(datetime(2026, 12, 25, 15, 0, tzinfo=UTC))

        assert "święto" in status["label"]
        assert "Boże Narodzenie" in status["label"]
        assert "Otwarta" not in status["label"]
        assert status["detail"] == "Najbliższe otwarcie: pon 28.12 09:30 ET"

    def test_thanksgiving_is_closed(self) -> None:
        status = market_status(datetime(2026, 11, 26, 15, 0, tzinfo=UTC))

        assert "święto" in status["label"]
        assert status["detail"] == "Najbliższe otwarcie: pt 27.11 09:30 ET"

    def test_early_close_session_is_named_with_its_close(self) -> None:
        # Piątek po Thanksgiving: 10:00 ET, zamknięcie 13:00 ET zamiast 16:00.
        status = market_status(datetime(2026, 11, 27, 15, 0, tzinfo=UTC))

        assert "Otwarta" in status["label"]
        assert "skrócona sesja do 13:00 ET" in status["label"]
        assert status["detail"] == "Zamknięcie za 3h 0min"

    def test_after_an_early_close_the_market_is_closed(self) -> None:
        # 14:00 ET po skróconej sesji — dawniej „Otwarta” do 16:00.
        status = market_status(datetime(2026, 11, 27, 19, 0, tzinfo=UTC))

        assert "After-hours" in status["label"]
        assert "skrócona sesja do 13:00 ET" in status["label"]
        assert status["detail"] == "Najbliższe otwarcie: pon 30.11 09:30 ET"

    def test_premarket_on_an_early_close_day_says_so(self) -> None:
        status = market_status(datetime(2026, 12, 24, 13, 0, tzinfo=UTC))

        assert "Premarket" in status["label"]
        assert "skrócona sesja do 13:00 ET" in status["label"]
        assert status["detail"] == "Otwarcie za 1h 30min"

    def test_friday_after_hours_points_to_monday_not_tomorrow(self) -> None:
        status = market_status(datetime(2026, 10, 16, 21, 0, tzinfo=UTC))

        assert "After" in status["label"]
        assert status["detail"] == "Najbliższe otwarcie: pon 19.10 09:30 ET"

    def test_weekday_after_hours_keeps_the_tomorrow_wording(self) -> None:
        status = market_status(datetime(2026, 10, 15, 22, 0, tzinfo=UTC))

        assert status["detail"] == "Otwarcie jutro 09:30 ET"

    def test_weekend_before_a_monday_holiday_points_to_tuesday(self) -> None:
        # Sobota przed Dniem Martina Luthera Kinga (pon 19.01.2026).
        status = market_status(datetime(2026, 1, 17, 18, 0, tzinfo=UTC))

        assert "weekend" in status["label"].lower()
        assert status["detail"] == "Najbliższe otwarcie: wt 20.01 09:30 ET"


class TestMarketStatusLocalTime:
    """MAIL-29: polski odbiorca myśli w godzinach 15:30–22:00, więc obok godzin ET
    status sesji podaje godzinę w strefie wyświetlania maila (`display_timezone`)."""

    def test_weekend_status_uses_polish_weekday_and_local_open(self) -> None:
        # Sobota 2026-10-17: oba kraje na czasie letnim, 09:30 EDT = 15:30 CEST.
        status = market_status(
            datetime(2026, 10, 17, 12, 0, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Najbliższe otwarcie: pon 19.10 09:30 ET (15:30 czasu PL)"

    def test_premarket_detail_names_the_local_open(self) -> None:
        status = market_status(
            datetime(2026, 10, 15, 10, 30, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Otwarcie za 3h 0min (15:30 czasu PL)"

    def test_open_session_detail_names_the_local_close(self) -> None:
        status = market_status(
            datetime(2026, 10, 15, 15, 0, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Zamknięcie za 5h 0min (22:00 czasu PL)"

    def test_early_close_is_converted_too(self) -> None:
        # Piątek po Thanksgiving: zamknięcie 13:00 EST = 19:00 CET.
        status = market_status(
            datetime(2026, 11, 27, 15, 0, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Zamknięcie za 3h 0min (19:00 czasu PL)"

    def test_tomorrow_wording_keeps_the_local_open(self) -> None:
        status = market_status(
            datetime(2026, 10, 15, 22, 0, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Otwarcie jutro 09:30 ET (15:30 czasu PL)"

    def test_dst_gap_between_the_us_and_poland_shifts_the_local_hour(self) -> None:
        # 28.10.2026: Polska już na CET (od 25.10), USA jeszcze na EDT (do 1.11),
        # więc 09:30 ET to 14:30 w Warszawie, nie 15:30.
        status = market_status(
            datetime(2026, 10, 28, 22, 0, tzinfo=UTC), display_timezone="Europe/Warsaw"
        )

        assert status["detail"] == "Otwarcie jutro 09:30 ET (14:30 czasu PL)"

    def test_other_display_timezone_is_called_local_not_polish(self) -> None:
        status = market_status(
            datetime(2026, 10, 17, 12, 0, tzinfo=UTC), display_timezone="Europe/London"
        )

        assert status["detail"] == (
            "Najbliższe otwarcie: pon 19.10 09:30 ET (14:30 czasu lokalnego)"
        )

    def test_without_a_display_timezone_only_et_is_shown(self) -> None:
        status = market_status(datetime(2026, 10, 17, 12, 0, tzinfo=UTC))

        assert status["detail"] == "Najbliższe otwarcie: pon 19.10 09:30 ET"


class TestForecastHorizonLabel:
    """ALR-15: etykieta horyzontu predykcji nazywa najbliższe zamknięcie sesji NYSE —
    cykl rozlicza prognozę ceną z następnego przebiegu, czyli po tej sesji."""

    def test_premarket_cycle_names_the_same_day_close(self) -> None:
        label = forecast_horizon_label(datetime(2026, 10, 15, 10, 30, tzinfo=UTC))

        assert label == "najbliższe zamknięcie NYSE: czw 15.10 16:00 ET"

    def test_cycle_before_a_holiday_skips_to_the_early_close(self) -> None:
        label = forecast_horizon_label(datetime(2026, 11, 26, 10, 30, tzinfo=UTC))

        assert label == "najbliższe zamknięcie NYSE: pt 27.11 13:00 ET (skrócona sesja)"

    def test_weekend_cycle_names_monday(self) -> None:
        label = forecast_horizon_label(datetime(2026, 10, 17, 10, 30, tzinfo=UTC))

        assert label == "najbliższe zamknięcie NYSE: pon 19.10 16:00 ET"

    def test_display_timezone_adds_the_local_close_hour(self) -> None:
        # Decyzja 2026-10-25: godzina w strefie odbiorcy tym samym helperem co MAIL-29.
        label = forecast_horizon_label(datetime(2026, 10, 15, 10, 30, tzinfo=UTC), "Europe/Warsaw")

        assert label == "najbliższe zamknięcie NYSE: czw 15.10 16:00 ET (22:00 czasu PL)"

    def test_local_close_hour_follows_the_dst_gap(self) -> None:
        # 26–30.10: Europa już na CET, USA jeszcze na EDT — zamknięcie o 21:00 PL.
        label = forecast_horizon_label(datetime(2026, 10, 27, 5, 30, tzinfo=UTC), "Europe/Warsaw")

        assert label == "najbliższe zamknięcie NYSE: wt 27.10 16:00 ET (21:00 czasu PL)"

    def test_local_hour_follows_the_early_close_note(self) -> None:
        label = forecast_horizon_label(datetime(2026, 11, 26, 10, 30, tzinfo=UTC), "Europe/Warsaw")

        assert label == (
            "najbliższe zamknięcie NYSE: pt 27.11 13:00 ET (skrócona sesja) (19:00 czasu PL)"
        )
