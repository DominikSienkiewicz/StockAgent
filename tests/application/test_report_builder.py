import dataclasses
import re
from datetime import UTC, date, datetime, timedelta, tzinfo
from decimal import Decimal
from typing import Any, Literal

import pytest
from pytest_mock import MockerFixture

from src.application import report_builder, report_risk_signals, report_trade_signals
from src.application.report_builder import (
    PortfolioMood,
    ReportContext,
    ResolvedPrediction,
    RiskSignal,
    SymbolResult,
    TopNewsItem,
    TradeSignal,
    build_correlation_chart_html,
    build_delta_chart_html,
    build_forecast_chart_html,
    build_html_report,
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    market_status,
    parse_resolved_predictions,
    to_lead_signals,
    to_symbol_result,
)
from src.application.report_formatting import company_label
from src.application.report_model_scorecard import ModelScorecardView
from src.application.run_context import RunContext
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.asset import Asset
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.cycle_maturity import SkipReason
from src.domain.earnings import EarningsEvent, EarningsTiming
from src.domain.equity_curve import EquityCurve, EquityPoint
from src.domain.model_scorecard import summarize_evaluations, summarize_run
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.portfolio import PlnRates, Portfolio, Position
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import AssetType


def _saved_result() -> SymbolResult:
    return SymbolResult(
        symbol="AAPL",
        status="saved",
        delta=Decimal("0.025"),
        current_price=Decimal("298.87"),
        trend="BULLISH",
        target_price=Decimal("305.00"),
        confidence_score=0.85,
        reasoning="Strong macro tailwinds + positive earnings expectations.",
        sentiment_score=0.42,
        sentiment_label="Somewhat-Bullish",
        news_volume=12,
    )


def _ignored_result(symbol: str = "MSFT") -> SymbolResult:
    return SymbolResult(symbol=symbol, status="ignored", delta=Decimal("0.005"))


def _error_result() -> SymbolResult:
    return SymbolResult(
        symbol="ASMIY", status="error", error_message="HTTPError: 429 Too Many Requests"
    )


def _crypto_saved() -> SymbolResult:
    return SymbolResult(
        symbol="BTC", status="saved", delta=Decimal("0.07"),
        current_price=Decimal("65000"), trend="BULLISH",
        target_price=Decimal("68000"), confidence_score=0.7,
        sentiment_score=0.5, sentiment_label="Bullish", news_volume=4,
        asset_class="CRYPTO",
    )


def _crypto_ignored() -> SymbolResult:
    return SymbolResult(
        symbol="ETH", status="ignored", delta=Decimal("0.01"),
        current_price=Decimal("3200"), asset_class="CRYPTO",
    )


class TestCryptoSection:
    """Krypto musi być WIDOCZNE w raporcie jako osobna sekcja — także gdy
    cykl został zignorowany (ruch poniżej progu 5%). Wcześniej krypto ginęło
    wśród ~40 akcji albo pojawiało się tylko jako mały chip w 'Pominięte'."""

    def test_to_symbol_result_sets_asset_class_from_asset(self) -> None:
        from src.domain.asset import Asset
        from src.domain.value_objects import AssetType

        raw = {
            "status": "ignored",
            "delta": 0.01,
            "current_price": 3200,
            "asset": Asset(symbol="ETH", asset_type=AssetType.CRYPTO),
        }
        result = to_symbol_result("ETH", raw)
        assert result.asset_class == "CRYPTO"

    def test_html_has_dedicated_crypto_section(self) -> None:
        html, _ = build_html_report(
            [_saved_result(), _crypto_saved(), _crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        # Wyróżnik sekcji (emoji), niezależny od taga sektora "Krypto" w treści.
        assert "🪙" in html
        assert "BTC" in html

    def test_ignored_crypto_visible_with_price_in_section(self) -> None:
        # Ignored krypto (poniżej progu) MUSI pokazać cenę w sekcji krypto —
        # lista 'Pominięte' pokazuje tylko deltę, nie cenę.
        html, _ = build_html_report(
            [_crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        assert "🪙" in html
        assert "$3\u00a0200,00" in html

    def test_crypto_section_in_plain_text(self) -> None:
        _, text = build_html_report(
            [_crypto_saved(), _crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        assert "KRYPTO" in text.upper()
        assert "BTC" in text
        assert "ETH" in text

    def test_no_crypto_section_when_no_crypto(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        assert "🪙" not in html

    def test_crypto_section_shows_prediction_reasoning(self) -> None:
        # "dlaczego wzrośnie/spadnie" dla krypto — widoczne w sekcji krypto.
        btc = SymbolResult(
            symbol="BTC", status="saved", delta=Decimal("0.07"),
            current_price=Decimal("65000"), trend="BULLISH",
            target_price=Decimal("68000"), confidence_score=0.7,
            reasoning="Napływy do ETF-ów spot napędzają popyt.",
            asset_class="CRYPTO",
        )
        html, text = build_html_report(
            [btc], datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        assert "Napływy do ETF-ów spot napędzają popyt" in html
        assert "Napływy do ETF-ów spot napędzają popyt" in text


class TestBuildHtmlReport:
    def test_returns_html_and_plain_text(self) -> None:
        results = [_saved_result(), _ignored_result(), _error_result()]
        html, text = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 12.5
        )
        assert isinstance(html, str)
        assert isinstance(text, str)
        assert len(html) > 100
        assert len(text) > 50

    def test_html_contains_summary_counts(self) -> None:
        results = [_saved_result(), _ignored_result(), _ignored_result("GOOGL")]
        html, _ = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        # 1 saved, 2 ignored, 0 errors
        assert ">1<" in html  # saved count
        assert ">2<" in html  # ignored count
        assert "AAPL" in html

    def test_html_contains_reasoning_for_saved(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "macro tailwinds" in html

    def test_html_shows_sector_in_predictions(self) -> None:
        # AAPL → Big Tech, doklejony przy nazwie spółki w sekcji PREDYKCJE.
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(sector_map={"AAPL": "Big Tech"}),
        )
        assert "Big Tech" in html

    def test_plain_text_shows_sector_in_predictions(self) -> None:
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(sector_map={"AAPL": "Big Tech"}),
        )
        assert "Big Tech" in text

    def test_suggestions_section_appears_for_hot_sector(self) -> None:
        # CRWD +6% → gorący sektor cyber → sekcja WARTE UWAGI z peerami.
        hot = SymbolResult(
            symbol="CRWD", status="saved", delta=Decimal("0.06"),
            current_price=Decimal("769"), trend="BULLISH",
            target_price=Decimal("780"), confidence_score=0.75,
            sentiment_score=0.3, sentiment_label="Bullish",
        )
        html, text = build_html_report(
            [hot], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(
                sector_map={"CRWD": "Cyberbezpieczeństwo"},
                sector_peers={"Cyberbezpieczeństwo": ["ZS", "FTNT", "CYBR", "TENB", "QLYS"]},
            ),
        )
        assert "Warte uwagi" in html
        assert "WARTE UWAGI" in text
        # Marker slotu nie może wyciec do treści.
        assert "SUGGESTIONS_SLOT" not in html
        assert "SUGGESTIONS_SLOT" not in text

    def test_no_suggestions_section_when_no_hot_sector(self) -> None:
        # Mały ruch, neutralny sentyment → brak sekcji i brak markera.
        cold = SymbolResult(
            symbol="AAPL", status="saved", delta=Decimal("0.004"),
            current_price=Decimal("100"), trend="SIDEWAYS",
            target_price=Decimal("100"), confidence_score=0.6,
            sentiment_score=0.05, sentiment_label="Neutral",
        )
        html, text = build_html_report(
            [cold], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(
                sector_map={"AAPL": "Big Tech"},
                sector_peers={"Big Tech": ["NFLX", "DIS"]},
            ),
        )
        assert "Warte uwagi" not in html
        assert "WARTE UWAGI" not in text
        assert "SUGGESTIONS_SLOT" not in html
        assert "SUGGESTIONS_SLOT" not in text

    def test_html_shows_errors_section_when_present(self) -> None:
        html, _ = build_html_report(
            [_error_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Błędy" in html
        assert "429" in html

    def test_plain_text_includes_all_sections(self) -> None:
        results = [_saved_result(), _ignored_result(), _error_result()]
        _, text = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        assert "PREDYKCJE" in text
        assert "POMINIĘTE" in text
        assert "BŁĘDY" in text
        assert "AAPL" in text
        assert "MSFT" in text
        assert "ASMIY" in text

    def test_plain_text_formatted_price_and_delta(self) -> None:
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "$298,87" in text
        assert "2,50%" in text

    def test_handles_empty_results(self) -> None:
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 0.0
        )
        # Brak crasha, nadal generuje strukturę
        assert ">0<" in html  # zero predykcji
        assert "Symboli:" in text
        assert "0" in text


class TestPolishLabels:
    def test_html_uses_polish_trend_label(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # BULLISH → "Wzrostowy"
        assert "Wzrostowy" in html
        assert "BULLISH" not in html

    def test_html_uses_polish_sentiment_label(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # "Somewhat-Bullish" → "Lekko pozytywny"
        assert "Lekko pozytywny" in html
        assert "Somewhat-Bullish" not in html

    def test_plain_text_uses_polish_trend(self) -> None:
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Wzrostowy" in text
        assert "BULLISH" not in text

    def test_polish_field_labels_in_table(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Label horyzontu jest uczciwy względem realnej kadencji (dziennej,
        # weekend ~72h) — "(cykl)" zamiast mylącego sztywnego "12h".
        assert "Zmiana (cykl)" in html
        assert "Prognoza" in html
        assert "Sentyment" in html
        assert "Target" not in html  # stara wersja
        assert "Zmiana 12h" not in html  # mylące sztywne "12h" usunięte


class TestDeltaChart:
    """#2 — wykres Δ jako inline HTML. Kontrakt zmieniony świadomie: dawniej
    `<img>` z quickchart.io (dziura w Gmailu z blokadą obrazków + wyciek danych
    portfela w URL-u), teraz div-bary renderowane w treści maila."""

    def test_renders_inline_bars_not_a_remote_image(self) -> None:
        results = [
            SymbolResult(symbol="AAPL", status="saved", delta=Decimal("0.025")),
            SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("-0.005")),
        ]
        chart = build_delta_chart_html(results)
        assert chart
        assert "quickchart" not in chart
        assert "AAPL" in chart

    def test_returns_empty_for_empty_results(self) -> None:
        assert build_delta_chart_html([]) == ""

    def test_excludes_error_results(self) -> None:
        results = [
            SymbolResult(symbol="X", status="error", error_message="boom"),
        ]
        assert build_delta_chart_html(results) == ""

    def test_excludes_results_without_delta(self) -> None:
        results = [SymbolResult(symbol="X", status="ignored", delta=None)]
        assert build_delta_chart_html(results) == ""

    def test_report_embeds_the_chart_without_any_remote_host(self) -> None:
        html, _ = build_html_report(
            [_saved_result(), _ignored_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0,
        )
        # Sedno #2: żadne dane nie opuszczają maila, żaden obrazek się nie ładuje.
        assert "quickchart.io" not in html
        assert "<img" not in html


class TestForecastDetails:
    """Prognoza: expected_change, confidence, drugi wykres."""

    def test_expected_change_computed_from_current_and_target(self) -> None:
        r = _saved_result()
        # (305 - 298.87) / 298.87 ≈ 0.02051
        assert r.expected_change is not None
        assert abs(r.expected_change - Decimal("0.02051")) < Decimal("0.001")

    def test_expected_change_is_none_when_target_missing(self) -> None:
        r = SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))
        assert r.expected_change is None

    def test_html_table_shows_signed_forecast_change(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Prognoza pokazana jako "$305.00 (+2.05%)"
        assert "$305,00" in html
        assert "+2,05%" in html or "+2,04%" in html

    def test_html_shows_confidence_percent(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "85%" in html
        assert "Pewność" in html

    def test_plain_text_shows_expected_move_and_confidence(self) -> None:
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "prognoza" in text.lower()
        assert "pewność 85%" in text.lower()

    def test_reasoning_section_includes_expected_move(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Oczekiwany ruch" in html
        assert "$298,87 → $305,00" in html


class TestForecastChart:
    def test_renders_inline_bars_for_saved(self) -> None:
        results = [
            SymbolResult(
                symbol="AAPL", status="saved",
                current_price=Decimal("100"), target_price=Decimal("105"),
            )
        ]
        chart = build_forecast_chart_html(results)
        assert chart
        assert "quickchart" not in chart

    def test_skips_ignored_in_forecast_chart(self) -> None:
        results = [
            SymbolResult(symbol="X", status="ignored", delta=Decimal("0.005")),
        ]
        assert build_forecast_chart_html(results) == ""

    def test_skips_saved_without_target(self) -> None:
        results = [SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))]
        assert build_forecast_chart_html(results) == ""

    def test_report_contains_both_charts_when_saved_present(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Dwa wykresy: Δ cyklu + prognoza. Oba inline, zero <img>.
        assert "Zmiana ceny (cykl)" in html
        assert "<img" not in html


class TestPortfolioMood:
    def test_avg_sentiment_and_extremes(self) -> None:
        results = [
            SymbolResult(symbol="A", status="saved", sentiment_score=0.4, confidence_score=0.8),
            SymbolResult(symbol="B", status="saved", sentiment_score=-0.3),
            SymbolResult(symbol="C", status="ignored", sentiment_score=0.1),
        ]
        mood = build_portfolio_mood(results)
        # avg = (0.4 + -0.3 + 0.1) / 3 ≈ 0.0667
        assert mood.avg_sentiment is not None
        assert abs(mood.avg_sentiment - 0.0667) < 0.001
        assert mood.most_bullish is not None
        assert mood.most_bullish.symbol == "A"
        assert mood.most_bearish is not None
        assert mood.most_bearish.symbol == "B"
        assert mood.high_confidence_count == 1
        assert mood.saved_count == 2

    def test_no_sentiment_gives_no_average(self) -> None:
        # MAIL-10: bez ani jednego sentymentu średniej nie ma (None), a nie 0.0
        # udające neutralny nastrój; wynik to zamrożony dataclass, nie dict.
        mood = build_portfolio_mood([SymbolResult(symbol="A", status="ignored")])

        assert isinstance(mood, PortfolioMood)
        assert mood.avg_sentiment is None
        assert mood.avg_sentiment_label is None
        assert mood.sentiment_count == 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            mood.__setattr__("saved_count", 1)

    def test_handles_empty_results(self) -> None:
        mood = build_portfolio_mood([])
        assert mood.avg_sentiment is None
        assert mood.most_bullish is None
        assert mood.most_bearish is None
        assert mood.saved_count == 0


class TestMarketStatus:
    def test_open_during_regular_session(self) -> None:
        # Czwartek 15:00 UTC = 10:00 ET (zima EST UTC-5)
        # 2026-02-05 to czwartek
        status = market_status(datetime(2026, 2, 5, 15, 0, tzinfo=UTC))
        assert "Otwarta" in status["label"]
        assert "Zamknięcie za" in status["detail"]

    def test_premarket_before_open(self) -> None:
        # Czwartek 13:00 UTC = 8:00 ET (przed otwarciem)
        status = market_status(datetime(2026, 2, 5, 13, 0, tzinfo=UTC))
        assert "Premarket" in status["label"]

    def test_after_hours_after_close(self) -> None:
        # Czwartek 22:00 UTC = 17:00 ET (po zamknięciu)
        status = market_status(datetime(2026, 2, 5, 22, 0, tzinfo=UTC))
        assert "After" in status["label"]

    def test_weekend_closed(self) -> None:
        # Sobota
        status = market_status(datetime(2026, 5, 16, 18, 0, tzinfo=UTC))
        assert "weekend" in status["label"].lower()


class TestReflectionAndTopNews:
    def test_reflection_insight_rendered_in_html(self) -> None:
        r = SymbolResult(
            symbol="AAPL",
            status="saved",
            sentiment_score=0.3,
            reflection_insight="Ostatni błąd: zignorowałem makro Fed.",
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Self-Reflection" in html
        assert "zignorowałem makro Fed" in html

    def test_cold_start_reflection_not_shown(self) -> None:
        r = SymbolResult(
            symbol="AAPL",
            status="saved",
            sentiment_score=0.3,
            reflection_insight=None,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Brak header'a gdy nikt nie ma reflection
        assert "Self-Reflection" not in html

    def test_top_news_rendered_in_reasoning(self) -> None:
        r = SymbolResult(
            symbol="NVDA",
            status="saved",
            sentiment_score=0.5,
            current_price=Decimal("100"),
            target_price=Decimal("105"),
            reasoning="strong AI",
            top_news=[
                TopNewsItem(
                    title="Nvidia raises guidance",
                    source="Bloomberg",
                    url="https://example.com/1",
                    relevance=0.98,
                    sentiment=0.7,
                ),
            ],
        )
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Top newsy" in html
        assert "Bloomberg" in html
        assert "Nvidia raises guidance" in html
        # Plain text też
        assert "Nvidia raises guidance" in text


class TestAccuracyStats:
    def test_html_renders_accuracy_when_provided(self) -> None:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(
                         accuracy_stats={
                    "mean_accuracy": 0.72,
                    "sample_count": 50,
                    "correct_count": 38,
                    "days_window": 30,
                },
                     ))
        assert "72,0%" in html or "72%" in html
        assert "50" in html
        assert "30 dni" in html

    def test_html_renders_no_data_placeholder(self) -> None:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(
                         accuracy_stats={
                    "mean_accuracy": None,
                    "sample_count": 0,
                    "correct_count": 0,
                    "days_window": 30,
                },
                     ))
        assert "brak ocenionych predykcji" in html.lower()

    def test_no_accuracy_section_when_none(self) -> None:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=None))
        assert "Historia trafności" not in html


class TestCorrelationChart:
    def test_renders_quadrant_table_for_3_plus_points(self) -> None:
        results = [
            SymbolResult(symbol=s, status="saved",
                         sentiment_score=v, delta=Decimal(str(v / 2)))
            for s, v in [("A", 0.3), ("B", -0.2), ("C", 0.1)]
        ]
        chart = build_correlation_chart_html(results)
        assert chart
        assert "quickchart" not in chart
        assert "<table" in chart

    def test_returns_empty_for_less_than_3(self) -> None:
        results = [
            SymbolResult(symbol="A", status="saved",
                         sentiment_score=0.3, delta=Decimal("0.05")),
            SymbolResult(symbol="B", status="saved",
                         sentiment_score=-0.2, delta=Decimal("-0.03")),
        ]
        assert build_correlation_chart_html(results) == ""


class TestTradeSignals:
    def test_sorts_by_strength_desc(self) -> None:
        weak = SymbolResult(
            symbol="W", status="saved", trend="BULLISH",
            confidence_score=0.4, current_price=Decimal("100"),
            target_price=Decimal("101"),  # +1% × 0.4 = 0.4
        )
        strong = SymbolResult(
            symbol="S", status="saved", trend="BULLISH",
            confidence_score=0.9, current_price=Decimal("100"),
            target_price=Decimal("106"),  # +6% × 0.9 = 5.4
        )
        signals = build_trade_signals([weak, strong])
        assert signals[0].symbol == "S"
        assert signals[1].symbol == "W"
        assert signals[0].strength > signals[1].strength

    def test_maps_trend_to_polish_direction(self) -> None:
        results = [
            SymbolResult(
                symbol="A", status="saved", trend="BULLISH",
                confidence_score=0.8, current_price=Decimal("100"),
                target_price=Decimal("105"),
            ),
            SymbolResult(
                symbol="B", status="saved", trend="BEARISH",
                confidence_score=0.7, current_price=Decimal("100"),
                target_price=Decimal("95"),
            ),
            SymbolResult(
                symbol="C", status="saved", trend="SIDEWAYS",
                confidence_score=0.5, current_price=Decimal("100"),
                target_price=Decimal("100.5"),
            ),
        ]
        signals = build_trade_signals(results)
        by_sym = {s.symbol: s for s in signals}
        assert by_sym["A"].direction == "KUP"
        assert by_sym["B"].direction == "SPRZEDAJ"
        assert by_sym["C"].direction == "OBSERWUJ"

    def test_skips_results_without_confidence_or_target(self) -> None:
        results = [
            SymbolResult(symbol="A", status="saved", trend="BULLISH"),  # no conf/target
            SymbolResult(symbol="B", status="ignored", delta=Decimal("0.01")),
        ]
        assert build_trade_signals(results) == []

    def test_html_renders_strongest_signals_section(self) -> None:
        r = SymbolResult(
            symbol="NVDA", status="saved", trend="BULLISH",
            confidence_score=0.85, current_price=Decimal("225"),
            target_price=Decimal("232.50"),
        )
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Najsilniejsze sygnały" in html
        assert "KUP" in html
        assert "NAJSILNIEJSZE SYGNAŁY" in text


class TestRiskSignals:
    def test_detects_divergence_up(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.04"), sentiment_score=-0.4,
        )
        signals = detect_risk_signals([r])
        types = [s.type for s in signals]
        assert "DIVERGENCE" in types

    def test_detects_divergence_down(self) -> None:
        r = SymbolResult(
            symbol="Y", status="saved",
            delta=Decimal("-0.04"), sentiment_score=0.4,
        )
        types = [s.type for s in detect_risk_signals([r])]
        assert "DIVERGENCE" in types

    def test_detects_av_llm_conflict(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved",
            av_llm_agreement=0.15,
        )
        types = [s.type for s in detect_risk_signals([r])]
        assert "AV_LLM_CONFLICT" in types

    def test_detects_low_signal(self) -> None:
        r = SymbolResult(symbol="X", status="saved", news_volume=2)
        types = [s.type for s in detect_risk_signals([r])]
        assert "LOW_SIGNAL" in types

    def test_no_signals_for_healthy_prediction(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.02"), sentiment_score=0.3,
            av_llm_agreement=0.85, news_volume=20,
        )
        assert detect_risk_signals([r]) == []

    def test_html_renders_risk_section(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.05"), sentiment_score=-0.4,
        )
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Sygnały ostrzegawcze" in html or "SYGNAŁY OSTRZEGAWCZE" in text


class TestDayOverDay:
    def test_parses_resolved_predictions(self) -> None:
        rows: list[dict[str, Any]] = [
            {"symbol": "NVDA", "predicted_trend": "BULLISH", "is_trend_correct": True},
            {"symbol": "SAP",  "predicted_trend": "BEARISH", "is_trend_correct": False},
            {"symbol": "Z",    "predicted_trend": "BULLISH", "is_trend_correct": None},  # filtered
        ]
        resolved = parse_resolved_predictions(rows)
        assert len(resolved) == 2
        by_sym = {r.symbol: r for r in resolved}
        assert by_sym["NVDA"].is_correct is True
        assert by_sym["SAP"].is_correct is False

    def test_correctness_reflects_trend_direction_not_price_proximity(self) -> None:
        """Regresja: predykcja BULLISH na spółce, która SPADŁA, musi być
        oznaczona jako błędna — nawet gdy liczbowa prognoza wypadła blisko.

        Bug: 'trafność' liczono z accuracy_score (odległość od celu +0.00%
        z cold-startu), więc kierunkowo błędne prognozy dostawały ~100%."""
        rows = [
            {"symbol": "AMD", "predicted_trend": "BULLISH",
             "accuracy_score": 0.995, "is_trend_correct": False},
        ]
        resolved = parse_resolved_predictions(rows)
        assert len(resolved) == 1
        assert resolved[0].is_correct is False

    def test_html_renders_resolved_section(self) -> None:
        resolved = [
            ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH",
                               is_correct=True),
            ResolvedPrediction(symbol="SAP", predicted_trend="BEARISH",
                               is_correct=False),
        ]
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        assert "Zamknięte predykcje" in html
        assert "NVDA" in html
        assert "SAP" in html
        assert "Trafiona" in html
        assert "Błędna" in html
        assert "ZAMKNIĘTE PREDYKCJE" in text
        assert "Trafiona" in text
        assert "Błędna" in text

    def test_empty_settlements_say_so(self) -> None:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(resolved_predictions=[]))
        assert "Brak rozliczeń predykcji z ostatnich 24h." in html


class TestResolvedPostMortem:
    """Sekcja 'Zamknięte predykcje' to pełen post-mortem: dlaczego prognoza
    mówiła wzrost/spadek (oryginalne uzasadnienie), co się faktycznie stało
    (ruch ceny) oraz czy/​dlaczego się potwierdziła (diagnoza dla chybionych).
    Dotyczy akcji I krypto (wszystko leci przez prediction_logs)."""

    def test_parse_enriches_with_reasoning_move_and_insight(self) -> None:
        rows = [
            {
                "symbol": "NVDA", "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "reasoning_text": "Silny popyt na GPU AI.",
                "correction_insights": "Trafiona predykcja.",
                "price_at_prediction": 100.0,
                "actual_price_after_12h": 106.0,
            }
        ]
        r = parse_resolved_predictions(rows)[0]
        assert r.reasoning == "Silny popyt na GPU AI."
        assert r.insight == "Trafiona predykcja."
        assert r.price_at_prediction == Decimal("100.0")
        assert r.actual_price == Decimal("106.0")
        assert r.actual_change_ratio == Decimal("0.06")

    def test_parse_reads_the_mistake_category(self) -> None:
        # WIA-57: kolumna `correction_category` (migracja 046); brak albo śmieć → None.
        from src.domain.mistake_taxonomy import MistakeCategory

        base = {"symbol": "NVDA", "predicted_trend": "BULLISH", "is_trend_correct": False}
        rows = [
            {**base, "correction_category": "MACRO"},
            {**base, "correction_category": "???"},
            base,
        ]

        parsed = parse_resolved_predictions(rows)

        assert [r.correction_category for r in parsed] == [MistakeCategory.MACRO, None, None]

    def test_miss_shows_reasoning_actual_move_and_diagnosis(self) -> None:
        resolved = [
            ResolvedPrediction(
                symbol="SAP", predicted_trend="BEARISH", is_correct=False,
                reasoning="Słabe wyniki kwartalne.",
                insight="Zignorowałem odbicie całego sektora chmury.",
                price_at_prediction=Decimal("100"),
                actual_price=Decimal("104"),
            )
        ]
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        # (1) dlaczego prognoza mówiła spadek
        assert "Słabe wyniki kwartalne" in html
        # (2) co się faktycznie stało — ruch ceny
        assert "+4,00%" in html
        # (3) dlaczego się NIE potwierdziła — diagnoza
        assert "Zignorowałem odbicie całego sektora chmury" in html
        # plain text też niesie pełen post-mortem
        assert "Słabe wyniki kwartalne" in text
        assert "Zignorowałem odbicie całego sektora chmury" in text

    def test_miss_with_placeholder_insight_shows_neutral_why(self) -> None:
        # WIA-10: historyczny placeholder „diagnoza odłożona” to nie diagnoza —
        # „dlaczego” mówi uczciwie, że kierunek się nie potwierdził.
        resolved = [
            ResolvedPrediction(
                symbol="SAP", predicted_trend="BEARISH", is_correct=False,
                reasoning="Słabe wyniki kwartalne.",
                insight="Niska zmienność — diagnoza LLM odłożona.",
                price_at_prediction=Decimal("100"),
                actual_price=Decimal("104"),
            )
        ]
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        assert "diagnoza LLM odłożona" not in html
        assert "diagnoza LLM odłożona" not in text
        assert "Kierunek się nie potwierdził." in html
        assert "Kierunek się nie potwierdził." in text

    def test_miss_diagnosis_mentioning_trafiona_is_still_shown(self) -> None:
        # WIA-10: placeholder rozpoznaje predykat, a nie podciąg „trafiona” —
        # prawdziwa diagnoza o „nietrafionej” prognozie nie może zniknąć.
        resolved = [
            ResolvedPrediction(
                symbol="SAP", predicted_trend="BEARISH", is_correct=False,
                insight="Prognoza nietrafiona, bo zignorowałem odbicie sektora.",
                price_at_prediction=Decimal("100"),
                actual_price=Decimal("104"),
            )
        ]
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        assert "zignorowałem odbicie sektora" in html

    def test_hit_shows_reasoning_and_confirmation(self) -> None:
        resolved = [
            ResolvedPrediction(
                symbol="NVDA", predicted_trend="BULLISH", is_correct=True,
                reasoning="Popyt na GPU do AI rośnie.",
                insight="Trafiona predykcja.",
                price_at_prediction=Decimal("100"),
                actual_price=Decimal("106"),
            )
        ]
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        assert "Popyt na GPU do AI rośnie" in html
        assert "+6,00%" in html
        assert "Trafiona" in html

    def test_crypto_closed_prediction_is_covered_and_tagged(self) -> None:
        resolved = [
            ResolvedPrediction(
                symbol="BTC", predicted_trend="BULLISH", is_correct=True,
                reasoning="Napływy do ETF-ów spot.",
                insight="Trafiona predykcja.",
                price_at_prediction=Decimal("60000"),
                actual_price=Decimal("63000"),
            )
        ]
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved, sector_map={"BTC": "Krypto"}),
        )
        assert "BTC" in html
        assert "Krypto" in html  # tag klasy aktywa (company_label_with_sector)
        assert "Napływy do ETF-ów spot" in html

    def test_backward_compatible_without_new_fields(self) -> None:
        # Stary kształt (bez reasoning/insight/cen) wciąż się renderuje.
        resolved = [
            ResolvedPrediction(
                symbol="AMD", predicted_trend="BULLISH", is_correct=True
            )
        ]
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        assert "AMD" in html
        assert "Trafiona" in html


class TestClickableNewsLinks:
    def test_top_news_with_url_renders_as_anchor(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r",
            sentiment_score=0.3,
            top_news=[
                TopNewsItem(
                    title="Big news",
                    source="Reuters",
                    url="https://reuters.com/abc",
                    relevance=0.95, sentiment=0.5,
                )
            ],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "<a href='https://reuters.com/abc'" in html
        assert "Big news</a>" in html

    def test_news_without_url_falls_back_to_plain_text(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r",
            sentiment_score=0.3,
            top_news=[
                TopNewsItem(title="No URL news", source="X",
                            url=None, relevance=0.9, sentiment=0.1)
            ],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "No URL news" in html
        assert "<a href" not in html.split("No URL news")[0][-200:]


class TestHtmlEscaping:
    def test_escapes_llm_news_and_error_text(self) -> None:
        r = SymbolResult(
            symbol="<NVDA>",
            status="saved",
            reasoning="<script>alert(1)</script>",
            sentiment_score=0.3,
            top_news=[
                TopNewsItem(
                    title="<b>bad</b>",
                    source="<src>",
                    url="https://example.com/?a=1&b=2",
                    relevance=0.9,
                    sentiment=0.2,
                )
            ],
        )
        err = SymbolResult(
            symbol="<ERR>",
            status="error",
            error_message="<img src=x onerror=alert(1)>",
        )

        html, _ = build_html_report(
            [r, err], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )

        assert "<script>" not in html
        assert "<img src=x" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;b&gt;bad&lt;/b&gt;" in html
        assert "&lt;src&gt;" in html
        assert "https://example.com/?a=1&amp;b=2" in html

    def test_rejects_unsafe_news_url(self) -> None:
        r = SymbolResult(
            symbol="X",
            status="saved",
            reasoning="r",
            sentiment_score=0.3,
            top_news=[
                TopNewsItem(
                    title="unsafe link",
                    source="Reuters",
                    url="javascript:alert(1)",
                    relevance=0.9,
                    sentiment=0.2,
                )
            ],
        )

        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        assert "javascript:alert" not in html
        assert "<a href" not in html.split("unsafe link")[0][-200:]

    def test_escapes_unknown_risk_signal_type_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding #36: gdy `rs.type` nie ma etykiety w `type_label`, surowa
        wartość trafia do HTML jako fallback. Dziś typy są hardcodowane, ale
        jeśli kiedyś staną się danymi pochodzącymi z zewnątrz, fallback bez
        escapowania jest sinkiem dla wstrzyknięcia HTML. Sąsiedni
        `rs.description` JEST escapowany — fallback musi być symetryczny."""
        injected = RiskSignal(
            symbol="X",
            type="<b>x</b>",  # nieznany typ z HTML — uderza w fallback
            severity="high",
            description="<i>desc</i>",
        )
        monkeypatch.setattr(
            report_risk_signals, "detect_risk_signals", lambda results: [injected]
        )
        r = SymbolResult(symbol="X", status="saved", delta=Decimal("0.02"))

        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        # Surowy HTML z `type` nie może wyciec — musi być zescapowany.
        assert "<b>x</b>" not in html
        assert "&lt;b&gt;x&lt;/b&gt;" in html
        # Sąsiednie pole pozostaje (już wcześniej) zescapowane.
        assert "&lt;i&gt;desc&lt;/i&gt;" in html


# Payloady używane przez regresje escapowania (finding #35). Każdy niesie
# surowy znacznik, który MUSI zniknąć z outputu, oraz "marker" — fragment
# zescapowany identycznie przez stdlib `html.escape` i autoescape Jinja
# (MarkupSafe). Różnią się one tylko escapowaniem `"` (`&quot;` vs `&#34;`),
# więc marker celowo zawiera tylko kątowe nawiasy, na które obie się zgadzają.
_XSS_SCRIPT = "<script>alert(1)</script>"
_XSS_SCRIPT_MARKER = "&lt;script&gt;alert(1)&lt;/script&gt;"
_XSS_IMG = '"><img src=x onerror=alert(1)>'
_XSS_IMG_MARKER = "&lt;img src=x onerror=alert(1)&gt;"


def _assert_escaped(html: str, payload: str, marker: str) -> None:
    """Twierdzi, że surowy payload nie wyciekł do HTML, a pole faktycznie
    zostało wyrenderowane w zescapowanej formie (a nie po cichu pominięte).

    Niezależne od biblioteki escapującej: f-string używa stdlib
    `html.escape`, autoescape Jinja — MarkupSafe; różnią się tylko na `"`.
    `marker` zawiera wyłącznie kątowe nawiasy, więc pasuje do obu."""
    assert payload not in html, f"Surowy payload wyciekł do HTML: {payload!r}"
    assert marker in html, (
        f"Brak zescapowanego markera (czy pole zostało wyrenderowane?): {marker!r}"
    )


class TestHtmlEscapingRegression:
    """Finding #35 — twardy regression guard na escapowanie niezaufanych pól.

    Większość maila składana jest f-stringami, gdzie KAŻDA niezaufana wartość
    musi być ręcznie owinięta w `_html(...)`. Jedno zapomniane `_html()` =
    stored-HTML-injection. Te testy zamrażają obecny (poprawny) stan: dla każdego
    niezaufanego pola renderowanego do maila wstrzykujemy payload XSS i żądamy,
    by output był zescapowany. Jeśli ktoś w przyszłości doda nowe pole bez
    `_html()` (lub zdejmie istniejące), odpowiedni test zczerwienieje."""

    def test_news_title_escaped(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r", sentiment_score=0.3,
            current_price=Decimal("100"), target_price=Decimal("105"),
            top_news=[TopNewsItem(title=_XSS_SCRIPT, source="Reuters",
                                  url="https://example.com/1",
                                  relevance=0.9, sentiment=0.2)],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_news_source_escaped(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r", sentiment_score=0.3,
            current_price=Decimal("100"), target_price=Decimal("105"),
            top_news=[TopNewsItem(title="ok", source=_XSS_IMG,
                                  url="https://example.com/1",
                                  relevance=0.9, sentiment=0.2)],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_llm_reasoning_escaped(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", reasoning=_XSS_SCRIPT,
            sentiment_score=0.3, current_price=Decimal("100"),
            target_price=Decimal("105"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_reflection_insight_escaped(self) -> None:
        r = SymbolResult(
            symbol="X", status="saved", sentiment_score=0.3,
            reflection_insight=_XSS_IMG,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_error_message_escaped(self) -> None:
        r = SymbolResult(
            symbol="X", status="error", error_message=_XSS_SCRIPT,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_resolved_prediction_reasoning_and_insight_escaped(self) -> None:
        resolved = [
            ResolvedPrediction(
                symbol="X", predicted_trend="BEARISH", is_correct=False,
                reasoning=_XSS_SCRIPT, insight=_XSS_IMG,
                price_at_prediction=Decimal("100"), actual_price=Decimal("104"),
            )
        ]
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_unknown_symbol_in_ignored_chip_escaped(self) -> None:
        # Symbol w chipie 'Pominięte' renderowany przez _html(r.symbol).
        r = SymbolResult(
            symbol=_XSS_IMG, status="ignored", delta=Decimal("0.001"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_rag_precedent_summary_escaped(self) -> None:
        # Q5: news_summary analogu pochodzi z LLM-a (dane zewnętrzne) — musi
        # być escapowany w bloku "Na podstawie analogów".
        raw = {
            "status": "saved", "delta": Decimal("0.03"),
            "current_price": Decimal("100"), "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH",
                             "confidence_score": 0.8, "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.2},
            "similar_precedents": [
                {"summary": _XSS_SCRIPT, "predicted_trend": "BEARISH",
                 "is_trend_correct": True},
            ],
        }
        result = to_symbol_result("X", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_provenance_badge_detail_escaped(self) -> None:
        # Q6: degraded_reason / nazwy flag trafiają do detalu chipa (title) —
        # muszą być escapowane.
        raw = {
            "status": "saved", "delta": Decimal("0.03"),
            "current_price": Decimal("100"), "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH",
                             "confidence_score": 0.8, "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.0,
                          "degraded_reason": _XSS_IMG},
            "data_quality_flags": [_XSS_IMG],
        }
        result = to_symbol_result("X", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)


class TestToSymbolResult:
    def test_maps_saved_status(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("-0.03"),
            "current_price": Decimal("100.50"),
            "ml_target_price": Decimal("97.0"),
            "llm_analysis": {
                "trend_direction": "BEARISH",
                "confidence_score": 0.72,
                "reasoning": "macro headwinds",
            },
            "sentiment": {
                "av_sentiment_score": -0.3,
                "av_sentiment_label": "Bearish",
                "news_volume_24h": 8,
            },
            "reflection_context": "Ostatni błąd: nie doceniłem hawkish Fed.",
            "news": [
                {
                    "title": "Fed signals hawkish stance",
                    "source": "Reuters",
                    "url": "https://example.com/1",
                    "relevance_score": "0.95",
                    "ticker_sentiment_score": "-0.4",
                },
            ],
        }
        result = to_symbol_result("AAPL", raw)
        assert result.symbol == "AAPL"
        assert result.status == "saved"
        assert result.trend == "BEARISH"
        assert result.target_price == Decimal("97.0")
        assert result.confidence_score == pytest.approx(0.72)
        assert result.reflection_insight == "Ostatni błąd: nie doceniłem hawkish Fed."
        assert len(result.top_news) == 1
        assert result.top_news[0].source == "Reuters"
        assert result.top_news[0].relevance == pytest.approx(0.95)

    def test_cold_start_reflection_filtered(self) -> None:
        raw = {
            "status": "saved",
            "reflection_context": "Brak danych historycznych do oceny.",
            "llm_analysis": {},
            "sentiment": {},
        }
        result = to_symbol_result("AAPL", raw)
        assert result.reflection_insight is None

    def test_maps_ignored_status(self) -> None:
        raw = {"status": "ignored", "delta": Decimal("0.005")}
        result = to_symbol_result("MSFT", raw)
        assert result.status == "ignored"
        assert result.delta == Decimal("0.005")

    def test_maps_error_from_exception(self) -> None:
        result = to_symbol_result("AAPL", None, error="HTTPError 500")
        assert result.status == "error"
        assert result.error_message == "HTTPError 500"

    def test_handles_missing_optional_fields(self) -> None:
        # Defensive: niekompletny raw dict (np. agent crashował w połowie)
        raw = {"status": "saved"}
        result = to_symbol_result("X", raw)
        assert result.status == "saved"
        assert result.delta is None
        assert result.trend is None


class TestRagPrecedentReceipts:
    """Q5 — analogi RAG z cyklu wystawione jako audytowalny blok."""

    def _raw_with_precedents(self, precedents: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.2},
            "similar_precedents": precedents,
        }

    def test_to_symbol_result_maps_precedents(self) -> None:
        raw = self._raw_with_precedents([
            {"summary": "Fed hawkish surprise", "predicted_trend": "BEARISH",
             "is_trend_correct": True},
        ])
        result = to_symbol_result("AAPL", raw)
        assert len(result.similar_precedents) == 1
        assert result.similar_precedents[0].summary == "Fed hawkish surprise"
        assert result.similar_precedents[0].predicted_trend == "BEARISH"
        assert result.similar_precedents[0].is_trend_correct is True

    def test_missing_precedents_yields_empty_list(self) -> None:
        result = to_symbol_result("AAPL", {"status": "saved", "llm_analysis": {},
                                           "sentiment": {}})
        assert result.similar_precedents == []

    def test_html_renders_analog_block_with_outcomes(self) -> None:
        raw = self._raw_with_precedents([
            {"summary": "Fed hawkish surprise spooks tech",
             "predicted_trend": "BEARISH", "is_trend_correct": True},
            {"summary": "Strong earnings beat", "predicted_trend": "BULLISH",
             "is_trend_correct": False},
        ])
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Na podstawie analogów" in html
        assert "Fed hawkish surprise spooks tech" in html
        assert "Strong earnings beat" in html

    def test_no_analog_block_when_empty(self) -> None:
        raw = self._raw_with_precedents([])
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Na podstawie analogów" not in html

    def test_plain_text_shows_analog_summary(self) -> None:
        raw = self._raw_with_precedents([
            {"summary": "Fed hawkish surprise spooks tech",
             "predicted_trend": "BEARISH", "is_trend_correct": True},
        ])
        result = to_symbol_result("AAPL", raw)
        _, text = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Fed hawkish surprise spooks tech" in text
        assert "analog" in text

    def test_analog_summary_escaped(self) -> None:
        raw = self._raw_with_precedents([
            {"summary": "<script>alert(1)</script>", "predicted_trend": "BULLISH",
             "is_trend_correct": True},
        ])
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


class TestProvenanceBadgesInReport:
    """Q6 — odznaki proweniencji renderowane jako chipy przy wierszu symbolu."""

    def test_to_symbol_result_builds_degraded_badge(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.0,
                          "degraded_reason": "av_keys_exhausted"},
            "data_quality_flags": ["av_keys_exhausted"],
        }
        result = to_symbol_result("AAPL", raw)
        levels = [b.level.value for b in result.provenance_badges]
        assert "DEGRADED" in levels

    def test_clean_inputs_yield_fresh_only(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.2},
            "data_quality_flags": [],
        }
        result = to_symbol_result("AAPL", raw)
        levels = [b.level.value for b in result.provenance_badges]
        assert levels == ["FRESH"]

    def test_html_renders_degraded_chip(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.0,
                          "degraded_reason": "av_keys_exhausted"},
            "data_quality_flags": ["av_keys_exhausted"],
        }
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Dane zdegradowane" in html

    def test_fresh_chip_not_rendered_no_noise(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.2},
            "data_quality_flags": [],
        }
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Świeże" not in html

    def test_plain_text_shows_provenance_label(self) -> None:
        raw = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.0,
                          "degraded_reason": "av_keys_exhausted"},
            "data_quality_flags": ["av_keys_exhausted"],
        }
        result = to_symbol_result("AAPL", raw)
        _, text = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "proweniencja" in text
        assert "Dane zdegradowane" in text


def _wall_clock_at(moment: datetime) -> type[datetime]:
    """Podmiana `datetime` w module raportu: `now()` zwraca `moment`."""

    class _WallClock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Any:
            return moment

    return _WallClock


class TestInjectedClock:
    """ARCH-26 — wiek fundamentów w odznakach liczy się od zegara cyklu
    (`now`), nie od chwili, w której mapper akurat się wykonał."""

    _NOW = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)

    def _raw(self, fundamentals_age: timedelta) -> dict[str, Any]:
        from src.domain.value_objects import Fundamentals

        return {
            "status": "saved",
            "delta": Decimal("0.03"),
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("103"),
            "llm_analysis": {"trend_direction": "BULLISH", "confidence_score": 0.8,
                             "reasoning": "r"},
            "sentiment": {"av_sentiment_score": 0.2},
            "data_quality_flags": [],
            "fundamentals": Fundamentals(
                trailing_pe=None, forward_pe=None, peg_ratio=None, eps_growth_yoy=None,
                fetched_at=self._NOW - fundamentals_age,
            ),
        }

    def test_to_symbol_result_uses_injected_now(self) -> None:
        # 24 h względem wstrzykniętego zegara = świeże; wobec zegara rzeczywistego
        # (dowolna data po 2026-01-16) te same fundamenty przekraczają TTL 144 h.
        raw = self._raw(timedelta(hours=24))
        first = to_symbol_result("AAPL", raw, now=self._NOW)
        second = to_symbol_result("AAPL", raw, now=self._NOW)
        assert first.provenance_badges == second.provenance_badges
        assert [b.level.value for b in first.provenance_badges] == ["FRESH"]

    def test_stale_age_is_measured_from_injected_now(self) -> None:
        result = to_symbol_result("AAPL", self._raw(timedelta(hours=200)), now=self._NOW)
        details = [b.detail for b in result.provenance_badges if b.level.value == "STALE"]
        assert details == ["fundamenty: 200h (TTL 144h)"]

    def test_same_now_gives_identical_report_whatever_the_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw = self._raw(timedelta(hours=100))
        reports = []
        for wall_clock in (self._NOW, self._NOW + timedelta(days=30)):
            monkeypatch.setattr(report_builder, "datetime", _wall_clock_at(wall_clock))
            result = to_symbol_result("AAPL", raw, now=self._NOW)
            reports.append(build_html_report([result], self._NOW, 1.0))
        assert reports[0] == reports[1]

    def test_without_now_the_mapper_falls_back_to_the_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            report_builder, "datetime", _wall_clock_at(self._NOW + timedelta(days=30))
        )
        result = to_symbol_result("AAPL", self._raw(timedelta(hours=100)))
        assert "STALE" in [b.level.value for b in result.provenance_badges]


class TestCompanyLabel:
    def test_known_symbol_returns_name(self) -> None:
        assert company_label("AAPL") == "AAPL (Apple Inc.)"

    def test_known_symbol_msft(self) -> None:
        assert company_label("MSFT") == "MSFT (Microsoft Corporation)"

    def test_known_symbol_nvda(self) -> None:
        assert company_label("NVDA") == "NVDA (NVIDIA Corporation)"

    def test_unknown_symbol_returns_symbol_only(self) -> None:
        assert company_label("ZZZZ") == "ZZZZ"

    def test_empty_symbol_returns_empty(self) -> None:
        assert company_label("") == ""


class TestCompanyNamesInReport:
    def test_html_shows_company_name_in_predictions_table(self) -> None:
        r = SymbolResult(
            symbol="AAPL", status="saved",
            delta=Decimal("0.025"), current_price=Decimal("298.87"),
            trend="BULLISH", target_price=Decimal("305.00"),
            confidence_score=0.85, reasoning="strong macro",
            sentiment_score=0.42,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in html

    def test_html_shows_company_name_in_reasoning_section(self) -> None:
        r = SymbolResult(
            symbol="MSFT", status="saved",
            trend="BULLISH", confidence_score=0.7,
            current_price=Decimal("400"), target_price=Decimal("420"),
            reasoning="cloud growth", sentiment_score=0.3,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Microsoft Corporation" in html

    def test_html_shows_company_name_in_trade_signals(self) -> None:
        r = SymbolResult(
            symbol="NVDA", status="saved", trend="BULLISH",
            confidence_score=0.9, current_price=Decimal("900"),
            target_price=Decimal("950"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "NVIDIA Corporation" in html

    def test_html_does_not_show_company_name_in_ignored_section(self) -> None:
        r = SymbolResult(symbol="AAPL", status="ignored", delta=Decimal("0.001"))
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Sekcja "Pominięte" NIE powinna zawierać nazwy spółki
        ignored_section = html.split("Pominięte")[1] if "Pominięte" in html else ""
        assert "Apple Inc." not in ignored_section

    def test_html_shows_company_name_in_errors_section(self) -> None:
        r = SymbolResult(symbol="AAPL", status="error", error_message="timeout")
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in html

    def test_plain_text_shows_company_name_in_predictions(self) -> None:
        r = SymbolResult(
            symbol="AAPL", status="saved",
            delta=Decimal("0.025"), current_price=Decimal("298.87"),
            trend="BULLISH", target_price=Decimal("305.00"),
            confidence_score=0.85, reasoning="strong macro",
            sentiment_score=0.42,
        )
        _, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in text

    def test_unknown_symbol_shows_only_ticker(self) -> None:
        r = SymbolResult(
            symbol="ZZZZ", status="saved",
            trend="SIDEWAYS", confidence_score=0.5,
            current_price=Decimal("10"), reasoning="no data",
            sentiment_score=0.0,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "ZZZZ" in html
        assert "ZZZZ (" not in html  # Brak nawiasów gdy nazwa nieznana


class TestRecommendationInReport:
    def test_html_shows_kup_for_bullish(self) -> None:
        r = SymbolResult(
            symbol="AAPL", status="saved", trend="BULLISH",
            confidence_score=0.8, current_price=Decimal("100"),
            target_price=Decimal("106"), reasoning="positive outlook",
            sentiment_score=0.4,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Rekomendacja KUP musi pojawić się w sekcji predykcji/uzasadnień
        after_header = (
            html.split("Wygenerowane predykcje")[1]
            if "Wygenerowane predykcje" in html
            else html
        )
        assert "KUP" in after_header

    def test_html_shows_sprzedaj_for_bearish(self) -> None:
        r = SymbolResult(
            symbol="TSLA", status="saved", trend="BEARISH",
            confidence_score=0.75, current_price=Decimal("200"),
            target_price=Decimal("190"), reasoning="negative outlook",
            sentiment_score=-0.3,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        after_header = (
            html.split("Wygenerowane predykcje")[1]
            if "Wygenerowane predykcje" in html
            else html
        )
        assert "SPRZEDAJ" in after_header

    def test_html_shows_trzymaj_for_sideways(self) -> None:
        r = SymbolResult(
            symbol="IBM", status="saved", trend="SIDEWAYS",
            confidence_score=0.6, current_price=Decimal("150"),
            target_price=Decimal("151"), reasoning="mixed signals",
            sentiment_score=0.0,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        after_header = (
            html.split("Wygenerowane predykcje")[1]
            if "Wygenerowane predykcje" in html
            else html
        )
        assert "TRZYMAJ" in after_header

    def test_sideways_signal_uses_the_council_hold_label(self) -> None:
        # Decyzja 2026-10-19: ruch boczny i HOLD rady to jedna etykieta — TRZYMAJ.
        r = SymbolResult(
            symbol="IBM", status="saved", trend="SIDEWAYS",
            confidence_score=0.6, current_price=Decimal("150"),
            target_price=Decimal("151"), reasoning="mixed signals",
            sentiment_score=0.0,
        )
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        assert "Sygnał modelu: TRZYMAJ" in html
        assert "Sygnał modelu: TRZYMAJ" in text
        assert "WSTRZYMAJ" not in html
        assert "WSTRZYMAJ" not in text

    def test_html_recommendation_includes_why_reason(self) -> None:
        r = SymbolResult(
            symbol="AAPL", status="saved", trend="BULLISH",
            confidence_score=0.8, current_price=Decimal("100"),
            target_price=Decimal("106"), reasoning="positive outlook",
            sentiment_score=0.4,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Powinien być powód (pewność / prognoza zmiana)
        assert "80%" in html  # pewność
        assert "+6,00%" in html  # prognozowana zmiana

    def test_plain_text_includes_recommendation(self) -> None:
        r = SymbolResult(
            symbol="AAPL", status="saved", trend="BULLISH",
            confidence_score=0.8, current_price=Decimal("100"),
            target_price=Decimal("106"), reasoning="positive outlook",
            sentiment_score=0.4,
        )
        _, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "KUP" in text or "REKOMENDACJA" in text.upper()


class TestBuildReportWithRiskWatch:
    """Integracja MacroRiskReport → wstawiana sekcja Risk Watch w mailu."""

    def _make_macro_report(self) -> MacroRiskReport:
        from src.application.use_cases.monitor_macro_risk import MacroRiskReport
        from src.domain.macro_risk import (
            MacroAlertLevel,
            MacroRiskInstrumentType,
            MacroRiskSignal,
        )
        from src.domain.value_objects import Money

        return MacroRiskReport(
            signals=[
                MacroRiskSignal(
                    symbol="SH",
                    instrument_type=MacroRiskInstrumentType.INVERSE_EQUITY,
                    current_price=Money(Decimal("104")),
                    previous_price=Money(Decimal("100")),
                )
            ],
            overall_alert=MacroAlertLevel.CRITICAL,
        )

    def test_html_contains_risk_watch_section_when_report_provided(self) -> None:
        r = _saved_result()
        html, _ = build_html_report(
            [r], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(macro_risk_report=self._make_macro_report()),
        )
        assert "Risk Watch" in html
        assert "SH" in html
        # Slot placeholder musi być wymieniony, nie zostać surowo w outpucie.
        assert "RISK_WATCH_SLOT" not in html

    def test_html_skips_section_when_no_macro_report(self) -> None:
        r = _saved_result()
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Surowy komentarz HTML jest dopuszczalny — placeholder pozostaje
        # w drzewie i jest ignorowany przez klientów mailowych.
        assert "Risk Watch" not in html

    def test_plain_text_contains_risk_watch_block(self) -> None:
        r = _saved_result()
        _, text = build_html_report(
            [r], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(macro_risk_report=self._make_macro_report()),
        )
        assert "Risk Watch" in text
        assert "SH" in text
        assert "RISK_WATCH_SLOT" not in text


class TestPersonaLeaderboardSection:
    """#3 — sekcja "Rada — ranking wiarygodności" wpięta w slot raportu."""

    @staticmethod
    def _records() -> list[PersonaTrackRecord]:
        return [
            PersonaTrackRecord(investor_name="Buffett", hit_rate=0.68, votes=22),
            PersonaTrackRecord(investor_name="Wood", hit_rate=0.41, votes=18),
        ]

    def test_html_contains_leaderboard_and_fills_slot(self) -> None:
        r = _saved_result()
        html, _ = build_html_report(
            [r], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(persona_track_record=self._records()),
        )
        assert "ranking wiarygodności" in html
        assert "Buffett" in html
        assert "68%" in html
        assert "22 głosy" in html
        # Slot musi zostać wymieniony, nie zostać surowo w treści maila.
        assert "PERSONA_LEADERBOARD_SLOT" not in html

    def test_html_suppresses_section_when_no_track_record(self) -> None:
        r = _saved_result()
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "ranking wiarygodności" not in html
        # Slot zawsze wymieniany (na ""), żeby marker nie wyciekł do maila.
        assert "PERSONA_LEADERBOARD_SLOT" not in html

    def test_html_suppresses_section_for_empty_track_record(self) -> None:
        r = _saved_result()
        html, _ = build_html_report(
            [r], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(persona_track_record=[]))
        assert "ranking wiarygodności" not in html
        assert "PERSONA_LEADERBOARD_SLOT" not in html


class TestLeadSection:
    """#1 — lead na pierwszym ekranie, PO quota bannerze. Ranking robi domena
    (`build_lead`); report_builder tylko mapuje DTO i renderuje."""

    _HEADING = "Najważniejsze w tym cyklu"

    @staticmethod
    def _critical_alert() -> QuotaAlert:
        return QuotaAlert(
            source="Alpha Vantage",
            severity=QuotaSeverity.CRITICAL,
            message="Wyczerpano wszystkie klucze",
            action="Dodaj klucz",
            occurred_at=datetime(2026, 5, 14, tzinfo=UTC),
        )

    def test_lead_suppressed_when_cycle_has_no_signal(self) -> None:
        r = _ignored_result("MSFT")
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        assert self._HEADING not in html
        assert "LEAD_SLOT" not in html
        assert "LEAD_SLOT" not in text

    def test_critical_quota_alert_produces_a_lead(self) -> None:
        html, _ = build_html_report(
            [_ignored_result("MSFT")],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        assert self._HEADING in html
        assert "Alpha Vantage" in html
        assert "LEAD_SLOT" not in html

    def test_plain_text_carries_the_lead(self) -> None:
        _, text = build_html_report(
            [_ignored_result("MSFT")],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        # Wariant plain-text niesie nagłówek sekcji wersalikami.
        assert self._HEADING.upper() in text
        assert "Alpha Vantage" in text
        assert "LEAD_SLOT" not in text

    def test_lead_appears_before_the_body_of_the_report(self) -> None:
        # Lead to PIERWSZY ekran — musi wyprzedzać sekcje szczegółowe.
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        assert html.index(self._HEADING) < html.index("AAPL")

    def test_split_council_lead_renders_fractional_delta_as_percent(self) -> None:
        # Styk mapper ↔ domena: `SymbolResult.delta` to UŁAMEK (-0.042 = -4,2%),
        # a `LeadSignal.price_delta_pct` to PROCENT. Bez przeliczenia lead i temat
        # maila pokazywały „NVDA -0.0%" zamiast „NVDA -4.2%".
        verdict = CouncilVerdict(
            final_recommendation="HOLD",
            consensus_strength=0.5,
            summary="1 BUY / 1 SELL",
            dissenting_views=(),
            investor_opinions=(
                InvestorOpinion(
                    investor_name="A", recommendation="BUY",
                    confidence=0.8, reasoning="x", key_factors=(),
                ),
                InvestorOpinion(
                    investor_name="B", recommendation="SELL",
                    confidence=0.8, reasoning="x", key_factors=(),
                ),
            ),
        )
        r = SymbolResult(
            symbol="NVDA", status="saved", delta=Decimal("-0.042"),
            current_price=Decimal("180.00"), trend="BEARISH",
            council_verdict=verdict,
        )

        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        assert "NVDA -4,2%, rada PODZIELONA" in html
        assert "NVDA -4,2%, rada PODZIELONA" in text

    def test_to_lead_signals_converts_fractional_delta_to_percent(self) -> None:
        # ARCH-04: `SymbolResult.delta` to ułamek, a `LeadSignal.price_delta_pct`
        # to `Percent` — przeliczenie należy do mappera, domena tylko formatuje.
        results = [SymbolResult(symbol="NVDA", status="saved", delta=Decimal("-0.042"))]

        (signal,) = to_lead_signals(results, [])

        assert signal.price_delta_pct == Decimal("-4.2")

    def test_council_flip_from_the_context_leads_the_report_and_its_subject(self) -> None:
        # MAIL-19: flip rady był tylko w sekcji #8 w połowie maila, a temat mówił
        # o czymś innym — teraz to lead, więc i temat (lead_headline).
        from src.domain.consensus_shift import ConsensusShift, ShiftKind
        from src.domain.digest_lead import lead_headline

        verdict = CouncilVerdict(
            final_recommendation="SELL",
            consensus_strength=0.6,
            summary="Rada zmieniła zdanie po wynikach.",
            dissenting_views=(),
            investor_opinions=(
                InvestorOpinion(
                    investor_name="A", recommendation="SELL",
                    confidence=0.8, reasoning="x", key_factors=(),
                ),
            ),
        )
        flipped = SymbolResult(
            symbol="NVDA", status="saved", delta=Decimal("-0.042"),
            current_price=Decimal("180.00"), trend="BEARISH", council_verdict=verdict,
        )
        shift = ConsensusShift(
            kind=ShiftKind.FLIP, previous_recommendation="BUY",
            current_recommendation="SELL", votes_changed=1,
            sentiment_delta=-0.3, consensus_delta=0.0, gap_hours=24.0,
        )
        context = ReportContext(consensus_shifts=[("NVDA", shift)])

        html, text = build_html_report(
            [flipped], datetime(2026, 5, 14, tzinfo=UTC), 1.0, context
        )

        headline = "NVDA -4,2%: rada odwróciła się KUP→SPRZEDAJ"
        assert headline in html
        assert headline in text
        assert lead_headline(report_builder.build_cycle_lead([flipped], context)) == headline


class TestSkipReasonMapping:
    """#4 — `skip_reason` rozdziela przyczyny pominięcia. Bez tego powitalny
    ton „Dzień 1" przykryłby masową awarię źródeł."""

    def test_cold_start_is_detected_from_zero_previous_price(self) -> None:
        raw = {"status": "ignored", "delta": Decimal("0"), "previous_price": Decimal("0")}

        result = to_symbol_result("AAPL", raw)

        assert result.skip_reason is SkipReason.COLD_START

    def test_below_threshold_when_previous_price_exists(self) -> None:
        raw = {
            "status": "ignored",
            "delta": Decimal("0.001"),
            "previous_price": Decimal("298.87"),
        }

        result = to_symbol_result("AAPL", raw)

        assert result.skip_reason is SkipReason.BELOW_THRESHOLD

    def test_saved_symbol_has_no_skip_reason(self) -> None:
        raw = {"status": "saved", "delta": Decimal("0.05"), "previous_price": Decimal("100")}

        assert to_symbol_result("AAPL", raw).skip_reason is None

    def test_error_symbol_has_no_skip_reason(self) -> None:
        # Błąd to NIE pominięcie — inaczej awaria wyglądałaby jak cold-start.
        result = to_symbol_result("AAPL", None, error="HTTP 429")

        assert result.status == "error"
        assert result.skip_reason is None


class TestUserPortfolioSection:
    """#15 — sekcja 💼 z realnymi wagami i P/L. Kurs PLN dostaje jako wartość
    z `MacroRiskReport.polish_macro`, NIE przez port — renderery są czyste."""

    @staticmethod
    def _portfolio(days_old: int) -> Portfolio:
        as_of = datetime(2026, 5, 14, tzinfo=UTC) - timedelta(days=days_old)
        return Portfolio(
            positions=(
                Position(
                    symbol="AAPL",
                    quantity=Decimal("10"),
                    purchase_price=Decimal("250"),
                    purchase_date=as_of.date(),
                ),
            ),
            as_of=as_of,
        )

    def test_section_suppressed_without_a_portfolio(self) -> None:
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )

        assert "USER_PORTFOLIO_SLOT" not in html

    def test_fresh_portfolio_renders_real_pnl(self) -> None:
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(user_portfolio=self._portfolio(days_old=0)))

        assert "AAPL" in html
        assert "USER_PORTFOLIO_SLOT" not in html

    def test_stale_portfolio_is_badged(self) -> None:
        # Stęchły portfel daje FAŁSZYWY P/L, gorszy niż brak sekcji.
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(user_portfolio=self._portfolio(days_old=30)))

        assert "STALE" in html.upper()


def _cold_start_result() -> SymbolResult:
    # Cold-start: graf zostawia target == bieżąca cena (baseline "bez zmian").
    return SymbolResult(
        symbol="AAPL",
        status="saved",
        delta=Decimal("0.025"),
        current_price=Decimal("298.87"),
        trend="BULLISH",
        target_price=Decimal("298.87"),
        confidence_score=0.85,
        reasoning="Strong macro tailwinds + positive earnings expectations.",
        sentiment_score=0.42,
        sentiment_label="Somewhat-Bullish",
        news_volume=12,
        ml_cold_start=True,
    )


class TestColdStartForecast:
    """MAIL-01 (G03): bez wag XGBoost "prognoza" to bieżąca cena. Raport ma to
    nazwać, a nie pokazywać +0.00% jak wynik modelu."""

    def test_mapper_carries_cold_start_flag(self) -> None:
        raw = {
            "status": "saved",
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("100"),
            "ml_cold_start": True,
        }
        assert to_symbol_result("AAPL", raw).ml_cold_start is True

    def test_mapper_without_flag_is_not_cold_start(self) -> None:
        raw = {"status": "saved", "current_price": Decimal("100")}
        assert to_symbol_result("AAPL", raw).ml_cold_start is False

    def test_cold_start_forecast_is_labelled_instead_of_zero_percent(self) -> None:
        html, text = build_html_report(
            [_cold_start_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "— (ML: cold-start)" in html
        assert "Oczekiwany ruch: — (ML: cold-start)" in html
        assert "Prognozowana zmiana" not in html
        # "(+0.00%)" to dotychczasowy format kolumny prognozy i linii ruchu.
        assert "(+0,00%)" not in html
        assert "prognoza — (ML: cold-start)" in text
        assert "(+0,00%)" not in text

    def test_trained_forecast_has_no_cold_start_label(self) -> None:
        html, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "ML: cold-start" not in html
        assert "ML: cold-start" not in text

    def test_scorecard_admits_cold_start_when_no_model_is_loaded(self) -> None:
        # ReportContext niesie fakt z portu ML — bez niego karta twierdziłaby,
        # że "jedziemy na sprawdzonym modelu", którego nie ma.
        view = ModelScorecardView(
            summary=summarize_run(["skipped_validation_failed"] * 2),
            trained_at=datetime(2026, 5, 13, tzinfo=UTC),
        )
        html, _ = build_html_report(
            [_cold_start_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0,
            ReportContext(model_scorecard=view, ml_model_loaded=False),
        )
        assert "Brak wytrenowanego modelu" in html
        assert "sprawdzonym" not in html.lower()

    def test_scorecard_names_the_candidate_behind_the_metrics(self) -> None:
        # ML-03: metryki w karcie mówią, którego kandydata dotyczą i na ilu próbkach.
        view = ModelScorecardView(
            summary=summarize_run(["trained_successfully"]),
            trained_at=datetime(2026, 5, 13, tzinfo=UTC),
            candidate_rmse=0.021,
            baseline_rmse=0.024,
            metrics_symbol="ALL",
            metrics_n_samples=243,
        )
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0,
            ReportContext(model_scorecard=view, ml_model_loaded=True),
        )
        assert "n=243" in html

    def test_scorecard_shows_per_symbol_evaluations_next_to_the_pooled_verdict(self) -> None:
        # Decyzja 3 (2026-10-18): werdykt z wiersza ALL, ewaluacje osobną linią.
        view = ModelScorecardView(
            summary=summarize_run(["trained_successfully"]),
            trained_at=datetime(2026, 5, 13, tzinfo=UTC),
            evaluations=summarize_evaluations(["evaluated_passed_gate"] * 3),
        )
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0,
            ReportContext(model_scorecard=view, ml_model_loaded=True),
        )
        assert "bramkę przeszło 3, nie przeszło 0, pominięto 0" in html


class TestColdStartSignalsAndChart:
    """WIA-13 (G03): sekcja „Najsilniejsze sygnały” i wykres prognoz nie mogą
    pokazywać baseline'u „cena bez zmian” jako prognozy +0.00%."""

    def test_signal_section_names_its_ranking_source_on_cold_start(self) -> None:
        html, text = build_html_report(
            [_cold_start_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Najsilniejsze sygnały" in html
        assert "ranking wg pewności" in html
        assert "+0,00%" not in html
        assert "ranking wg pewności" in text
        assert "+0,00%" not in text

    def test_forecast_chart_skips_cold_start_results(self) -> None:
        assert build_forecast_chart_html([_cold_start_result()]) == ""


class TestAccuracyHistoryInterval:
    """WIA-32: historia trafności w mailu pokazuje przedział Wilsona obok procentu."""

    _STATS = {
        "mean_accuracy": 0.72,
        "sample_count": 50,
        "correct_count": 36,
        "days_window": 30,
    }

    def test_html_shows_wilson_interval(self) -> None:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=self._STATS)
        )

        # 36/50 → Wilson 95% ≈ 58–83%.
        assert "95% CI 58–83%" in html

    def test_plain_text_shows_wilson_interval(self) -> None:
        _, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=self._STATS)
        )

        assert "95% CI 58–83%" in text


class TestResolvedPredictionTimestamp:
    """WIA-17/23: klucz cyklu (dzień UTC) wymaga znacznika czasu predykcji."""

    def test_parse_keeps_the_prediction_timestamp(self) -> None:
        rows = [
            {
                "symbol": "NVDA",
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "timestamp": "2026-10-17T22:30:00+00:00",
            }
        ]

        r = parse_resolved_predictions(rows)[0]

        assert r.predicted_at == datetime(2026, 10, 17, 22, 30, tzinfo=UTC)

    def test_unparseable_timestamp_is_none(self) -> None:
        rows = [
            {"symbol": "NVDA", "predicted_trend": "BULLISH", "is_trend_correct": True,
             "timestamp": "not-a-date"},
        ]

        assert parse_resolved_predictions(rows)[0].predicted_at is None


def _signal(**overrides: object) -> TradeSignal:
    fields: dict[str, object] = {
        "symbol": "AAPL",
        "direction": "KUP",
        "confidence": 0.85,
        "expected_change": Decimal("0.031"),
        "strength": 0.58 * 3.1,
        "current_price": Decimal("100"),
        "target_price": Decimal("103.1"),
        "calibrated_confidence": 0.58,
    }
    fields.update(overrides)
    return TradeSignal(**fields)  # type: ignore[arg-type]


def test_trade_signal_shows_calibrated_confidence() -> None:
    """WIA-12: ranking liczy się na pewności skalibrowanej, a mail pokazywał
    tylko surową — `TradeSignal` obiecywał format „85% → 58%”."""
    html = report_trade_signals._render_trade_signals_html([
        TradeSignal(
            symbol="AAPL",
            direction="KUP",
            confidence=0.85,
            expected_change=Decimal("0.031"),
            strength=0.58 * 3.1,
            current_price=Decimal("100"),
            target_price=Decimal("103.1"),
            calibrated_confidence=0.58,
        )
    ])

    assert "58%" in html


class TestTradeSignalCalibrationAndStrength:
    """WIA-12: pewność surowa → skalibrowana z n kubełka, a zamiast gołej
    „siły sygnału” pozycja w rankingu i składowe, z których wynika."""

    def test_html_shows_raw_to_calibrated_with_bucket_n(self) -> None:
        html = report_trade_signals._render_trade_signals_html(
            [_signal(calibration_bucket_count=12)]
        )

        assert "85%" in html
        assert "→" in html
        assert "58%" in html
        assert "po kalibracji" in html
        assert "n kubełka: 12" in html

    def test_html_replaces_bare_strength_with_rank_and_components(self) -> None:
        html = report_trade_signals._render_trade_signals_html([
            _signal(symbol="AAPL"),
            _signal(symbol="MSFT", calibrated_confidence=None, strength=0.85 * 3.1),
        ])

        assert "siła sygnału" not in html
        assert "#1" in html
        assert "#2" in html
        assert "ruch 3,1% × pewność skalibrowana 58%" in html
        assert "ruch 3,1% × pewność 85%" in html

    def test_html_uncalibrated_signal_shows_raw_confidence_only(self) -> None:
        html = report_trade_signals._render_trade_signals_html(
            [_signal(calibrated_confidence=None)]
        )

        assert "pewność <strong>85%</strong>" in html
        assert "po kalibracji" not in html

    def test_html_cold_start_strength_is_the_confidence_alone(self) -> None:
        html = report_trade_signals._render_trade_signals_html([
            _signal(
                expected_change=Decimal("0"),
                target_price=Decimal("100"),
                strength=0.58,
                ml_cold_start=True,
            )
        ])

        assert "wg pewności skalibrowanej 58%" in html
        assert "ruch 0,0%" not in html

    def test_text_shows_calibration_and_rank_components(self) -> None:
        lines = report_trade_signals._render_trade_signals_text(
            [_signal(calibration_bucket_count=12)]
        )
        text = "\n".join(lines)

        assert "pewność 85% → 58% po kalibracji (n kubełka: 12)" in text
        assert "#1 · ruch 3,1% × pewność skalibrowana 58%" in text
        assert "siła " not in text


class TestAccuracyHistoryBothDefinitions:
    """WIA-29 (c): „Historia trafności” liczyła SIDEWAYS do trafności, a Track
    Record pokazuje trafność kierunkową — dwie różne liczby pod jedną nazwą.
    Mail pokazuje obie obok siebie, z etykietami mówiącymi, co liczą."""

    _STATS: dict[str, object] = {
        "mean_accuracy": 0.72,
        "sample_count": 50,
        "correct_count": 36,
        "days_window": 30,
        "directional_count": 40,
        "directional_correct_count": 26,
    }

    def _html(self, stats: dict[str, object]) -> str:
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=stats)
        )
        return html

    def _text(self, stats: dict[str, object]) -> str:
        _, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=stats)
        )
        return text

    def test_html_shows_directional_next_to_total(self) -> None:
        html = self._html(self._STATS)

        assert "Trafność kierunkowa (bez prognoz „bez ruchu”)" in html
        assert "65,0%" in html
        assert "n=40" in html
        assert "Trafność łączna (z prognozami „bez ruchu”)" in html
        assert "72,0%" in html
        assert "95% CI 58–83%, n=50" in html
        assert "Średnia trafność" not in html

    def test_text_shows_directional_next_to_total(self) -> None:
        text = self._text(self._STATS)

        assert "Trafność kierunkowa (bez prognoz „bez ruchu”): 65,0%" in text
        assert "Trafność łączna (z prognozami „bez ruchu”): 72,0%" in text
        assert "Średnia trafność" not in text

    def test_no_directional_calls_is_said_plainly(self) -> None:
        stats = {**self._STATS, "directional_count": 0, "directional_correct_count": 0}

        assert "brak prognoz kierunkowych" in self._html(stats)
        assert "brak prognoz kierunkowych" in self._text(stats)

    def test_stats_without_directional_counts_show_the_labelled_total_only(self) -> None:
        stats = {
            key: value
            for key, value in self._STATS.items()
            if not key.startswith("directional")
        }

        html = self._html(stats)

        assert "Trafność łączna (z prognozami „bez ruchu”)" in html
        assert "Trafność kierunkowa" not in html


class TestAccuracyHistorySessionClose:
    """WIA-61 (okres przejściowy, opcja „oba”): obok trafności na cenie /quote
    Historia trafności pokazuje trafność na oficjalnym zamknięciu sesji i, dla
    porównania, trafność /quote na tych samych predykcjach."""

    _STATS: dict[str, object] = {
        **TestAccuracyHistoryBothDefinitions._STATS,
        "session_close_count": 40,
        "session_close_correct_count": 30,
        "session_close_quote_correct_count": 28,
    }

    @staticmethod
    def _report(stats: dict[str, object]) -> tuple[str, str]:
        return build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=stats)
        )

    def test_both_hit_rates_are_shown_in_html_and_text(self) -> None:
        html, text = self._report(self._STATS)

        for body in (html, text):
            assert "Na oficjalnym zamknięciu sesji: 75,0%" in body
            assert "na cenie /quote tych samych predykcji: 70,0%" in body
            assert "n=40" in body

    def test_without_session_close_counts_the_line_is_absent(self) -> None:
        html, text = self._report(TestAccuracyHistoryBothDefinitions._STATS)

        assert "oficjalnym zamknięciu sesji" not in html
        assert "oficjalnym zamknięciu sesji" not in text


class TestResolvedSummaryBothDefinitions:
    """Decyzja 2026-10-19 (a): podsumowanie „Zamknięte predykcje (24h)” mówiło
    „(X% accuracy)” z SIDEWAYS w mianowniku. Teraz ta sama para etykiet co
    Historia trafności: kierunkowa obok łącznej."""

    @staticmethod
    def _resolved(symbol: str, trend: str, correct: bool) -> ResolvedPrediction:
        return ResolvedPrediction(symbol=symbol, predicted_trend=trend, is_correct=correct)

    def _report(self, resolved: list[ResolvedPrediction]) -> tuple[str, str]:
        return build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=resolved),
        )

    def test_summary_shows_directional_next_to_total(self) -> None:
        html, text = self._report([
            self._resolved("NVDA", "BULLISH", True),
            self._resolved("AMD", "BEARISH", False),
            self._resolved("SPY", "SIDEWAYS", True),
        ])

        for body in (html, text):
            assert "Trafność kierunkowa (bez prognoz „bez ruchu”): 50% (n=2)" in body
            assert "Trafność łączna (z prognozami „bez ruchu”): 67% (n=3)" in body
            assert "% accuracy" not in body

    def test_summary_without_directional_calls_says_so(self) -> None:
        html, text = self._report([self._resolved("SPY", "SIDEWAYS", True)])

        assert "brak prognoz kierunkowych" in html
        assert "brak prognoz kierunkowych" in text


def _strong_buy_result() -> SymbolResult:
    """Saved BUY z jednogłośną radą — jedyną brakującą przesłanką pełnej bandy
    jest track record."""
    verdict = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.9,
        summary="Rada zgodna.",
        dissenting_views=(),
        investor_opinions=(
            InvestorOpinion(
                investor_name="A", recommendation="BUY",
                confidence=0.9, reasoning="x", key_factors=(),
            ),
            InvestorOpinion(
                investor_name="B", recommendation="BUY",
                confidence=0.9, reasoning="x", key_factors=(),
            ),
        ),
    )
    return SymbolResult(
        symbol="NVDA", status="saved", delta=Decimal("0.04"),
        current_price=Decimal("100"), trend="BULLISH",
        target_price=Decimal("106"), confidence_score=0.85,
        council_verdict=verdict,
    )


def _sizing_report(accuracy_stats: dict[str, object]) -> tuple[str, str]:
    return build_html_report(
        [_strong_buy_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(accuracy_stats=accuracy_stats),
    )


def test_full_band_not_suggested_on_tiny_accuracy_sample() -> None:
    # PORT-01 (G31): jedno trafienie (100%, n_eff = 1) nie jest track recordem.
    html, text = _sizing_report({
        "mean_accuracy": 1.0, "sample_count": 1, "correct_count": 1,
        "days_window": 30, "n_eff": 1.0,
    })

    assert "Pozycja pełna" not in html
    assert "Pozycja pełna" not in text
    assert "Pozycja standardowa" in html


def test_full_band_not_suggested_when_accuracy_stats_have_no_effective_sample() -> None:
    # Bez n_eff nie wiadomo, ile niezależnych zakładów stoi za hit-rate —
    # 40 trafień jednego dnia spadku NASDAQ to jeden zakład.
    html, _ = _sizing_report({
        "mean_accuracy": 0.9, "sample_count": 40, "correct_count": 36,
        "days_window": 30,
    })

    assert "Pozycja pełna" not in html


def test_full_band_suggested_on_a_sufficient_effective_sample() -> None:
    html, _ = _sizing_report({
        "mean_accuracy": 0.7, "sample_count": 200, "correct_count": 140,
        "days_window": 30, "n_eff": 25.0,
    })

    assert "Pozycja pełna 4-5% portfela" in html


def test_plain_text_report_says_var_lacks_history() -> None:
    # PORT-22: wariant tekstowy też mówi, ile historii brakuje.
    from src.application.use_cases.portfolio_risk import PortfolioRiskReport

    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(portfolio_risk_report=PortfolioRiskReport(
            watchlist_size=3, insufficient=(("var", 9, 60),),
        )),
    )

    assert "za mało historii (9/60)" in html
    assert "za mało historii (9/60)" in text


def test_user_portfolio_lists_a_position_the_cycle_could_not_price() -> None:
    # PORT-03: XDWD.DE ma status UNSUPPORTED_PRICE, więc cykl nie daje mu ceny —
    # sekcja portfela mówi to wprost zamiast po cichu zawyżać wagi reszty.
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    portfolio = Portfolio(
        positions=(
            Position(symbol="AAPL", quantity=Decimal("10"),
                     purchase_price=Decimal("250"), purchase_date=as_of.date()),
            Position(symbol="XDWD.DE", quantity=Decimal("20"),
                     purchase_price=Decimal("100"), purchase_date=as_of.date()),
        ),
        as_of=as_of,
    )
    unsupported = SymbolResult(
        symbol="XDWD.DE", status="ignored", skip_reason=SkipReason.UNSUPPORTED_PRICE,
    )

    html, _ = build_html_report(
        [_saved_result(), unsupported], as_of, 1.0,
        ReportContext(user_portfolio=portfolio),
    )

    assert "Bez wyceny w tym cyklu: XDWD.DE (≈ 44,4% kosztu)" in html


def test_held_price_outside_the_watchlist_prices_the_position() -> None:
    # PORT-07: cena z samego /quote (ReportContext.held_prices) wycenia pozycję
    # spoza watchlisty — trafia do tabeli zamiast do „Bez wyceny”.
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    portfolio = Portfolio(
        positions=(
            Position(symbol="AAPL", quantity=Decimal("10"),
                     purchase_price=Decimal("250"), purchase_date=as_of.date()),
            Position(symbol="MSFT", quantity=Decimal("2"),
                     purchase_price=Decimal("400"), purchase_date=as_of.date()),
        ),
        as_of=as_of,
    )

    html, _ = build_html_report(
        [_saved_result()], as_of, 1.0,
        ReportContext(user_portfolio=portfolio, held_prices={"MSFT": Decimal("410")}),
    )

    assert "Bez wyceny" not in html
    assert "$820,00" in html


def test_lead_warns_about_a_council_signal_against_a_large_position() -> None:
    # PORT-10: rada SPRZEDAJ na pozycji ważącej ≥ 10% wyprzedza silny konsensus
    # na spółce, której użytkownik nie ma.
    def verdict(rec: Literal["BUY", "SELL", "HOLD"]) -> CouncilVerdict:
        return CouncilVerdict(
            final_recommendation=rec, consensus_strength=0.9, summary=f"Rada: {rec}.",
            dissenting_views=(),
            investor_opinions=tuple(
                InvestorOpinion(investor_name=n, recommendation=rec,
                                confidence=0.9, reasoning="x", key_factors=())
                for n in ("A", "B")
            ),
        )

    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    results = [
        SymbolResult(symbol="NVDA", status="saved", delta=Decimal("-0.042"),
                     current_price=Decimal("100"), council_verdict=verdict("SELL")),
        SymbolResult(symbol="AMD", status="saved", delta=Decimal("0.08"),
                     current_price=Decimal("50"), council_verdict=verdict("BUY")),
    ]
    portfolio = Portfolio(
        positions=(
            Position(symbol="NVDA", quantity=Decimal("10"),
                     purchase_price=Decimal("90"), purchase_date=as_of.date()),
            Position(symbol="AAPL", quantity=Decimal("1"),
                     purchase_price=Decimal("3000"), purchase_date=as_of.date()),
        ),
        as_of=as_of,
    )

    html, text = build_html_report(
        results, as_of, 1.0,
        ReportContext(user_portfolio=portfolio, held_prices={"AAPL": Decimal("3000")}),
    )

    headline = "Sygnał przeciw Twojej pozycji: rada SPRZEDAJ na NVDA (25% portfela) -4,2%"
    assert headline in html
    assert html.index(headline) < html.index("AMD +8,0%, rada zgodna")
    assert headline in text


def _held_aapl_portfolio() -> Portfolio:
    as_of = datetime(2026, 5, 14, tzinfo=UTC)
    return Portfolio(
        positions=(
            Position(symbol="AAPL", quantity=Decimal("10"),
                     purchase_price=Decimal("250"), purchase_date=as_of.date()),
        ),
        as_of=as_of,
    )


def test_plain_text_email_carries_the_user_portfolio_section() -> None:
    # Decyzja 2026-10-19: wersja tekstowa maila ma sekcję portfela jak HTML.
    _, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(user_portfolio=_held_aapl_portfolio()),
    )

    assert "=== 💼 Twój portfel ===" in text
    assert "$2\u00a0988,70" in text
    assert "USER_PORTFOLIO_SLOT" not in text


def test_plain_text_without_user_portfolio_keeps_the_html_section() -> None:
    # Stary push (pełny tekst do Telegram/Slack) dostaje tekst BEZ kwot portfela.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(user_portfolio=_held_aapl_portfolio(), user_portfolio_in_text=False),
    )

    assert "Twój portfel" not in text
    assert "$2\u00a0988,70" not in text
    assert "USER_PORTFOLIO_SLOT" not in text
    assert "Twój portfel" in html


@pytest.mark.parametrize(
    "context",
    [None, ReportContext(macro_risk_report=MacroRiskReport())],
    ids=["no-risk-watch-report", "empty-risk-watch-report"],
)
def test_plain_text_without_risk_watch_data_has_no_slot_marker(
    context: ReportContext | None,
) -> None:
    # Pusty slot = brak sekcji. Komentarz HTML jest w mailu HTML niewidoczny,
    # ale w wersji tekstowej był dosłowną linią „<!-- RISK_WATCH_SLOT -->”.
    _, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0, context
    )

    assert "RISK_WATCH_SLOT" not in text
    assert "<!--" not in text


def test_plain_text_without_quota_alerts_has_no_slot_marker() -> None:
    # Ten sam mechanizm co Risk Watch: bez alertów kwot tekst zaczynał się od
    # dosłownego „<!-- QUOTA_BANNER_SLOT -->”.
    _, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    assert "QUOTA_BANNER_SLOT" not in text
    assert text.startswith("=" * 64)


def test_plain_text_with_quota_alert_still_starts_with_the_banner() -> None:
    alert = QuotaAlert(
        source="OpenAI", severity=QuotaSeverity.CRITICAL,
        message="OpenAI 429: insufficient_quota", action="Doładuj konto OpenAI.",
        occurred_at=datetime(2026, 5, 14, tzinfo=UTC),
    )

    _, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(quota_alerts=[alert]),
    )

    assert "QUOTA_BANNER_SLOT" not in text
    assert "OpenAI 429" in text.split("STOCKAGENT — RAPORT CYKLU")[0]


def test_quota_banner_shows_the_alert_in_the_display_timezone() -> None:
    # MAIL-59: baner kwot mówi czasem lokalnym, jak nagłówek, a nie „05:30 UTC”.
    # Londyn zamiast domyślnej Warszawy, żeby test łapał brak przekazania strefy.
    alert = QuotaAlert(
        source="OpenAI", severity=QuotaSeverity.CRITICAL,
        message="OpenAI 429: insufficient_quota", action="Doładuj konto OpenAI.",
        occurred_at=datetime(2026, 10, 16, 5, 30, tzinfo=UTC),
    )

    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 8, 0, tzinfo=UTC), 1.0,
        ReportContext(quota_alerts=[alert], display_timezone="Europe/London"),
    )

    assert "pt 16.10 06:30" in html
    assert "(pt 16.10 06:30)" in text
    assert "05:30 UTC" not in html
    assert "05:30 UTC" not in text


def test_header_shows_the_cycle_start_in_the_display_timezone() -> None:
    # MAIL-14: nagłówek HTML i tekstu w czasie lokalnym, nie „… UTC”.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(display_timezone="Europe/Warsaw"),
    )

    assert "Czas rozpoczęcia: pt 16.10 07:30" in html
    assert "Czas rozpoczęcia:  pt 16.10 07:30" in text
    assert "05:30" not in html
    assert "05:30" not in text


def test_header_session_line_gives_the_nyse_open_in_the_display_timezone() -> None:
    # MAIL-29: sobota → „pon 19.10 09:30 ET” plus godzina polska, w HTML i w tekście.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 17, 12, 0, tzinfo=UTC), 1.0,
        ReportContext(display_timezone="Europe/Warsaw"),
    )

    expected = "Najbliższe otwarcie: pon 19.10 09:30 ET (15:30 czasu PL)"
    assert expected in html
    assert expected in text


def test_header_shows_regime_and_multiplier() -> None:
    # ALR-05: reżim zacieśnia bramkę każdego cyklu — czytelnik widzi mniej predykcji,
    # więc nagłówek mówi, jaki reżim, jaki mnożnik progu i dlaczego. Liczby po
    # polsku (MAIL-58), stąd „×1,5”, a nie „×1.5” z opisu wpisu.
    from src.application.report_regime import RegimeBanner
    from src.domain.regime import MarketRegime

    banner = RegimeBanner(
        regime=MarketRegime.RISK_OFF,
        multiplier=1.5,
        base_threshold=Decimal("0.02"),
        reason="2 głosy ELEVATED",
    )

    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(regime=banner),
    )

    expected = "Reżim: RISK_OFF · próg ×1,5 (2,0% → 3,0%) · powód: 2 głosy ELEVATED"
    assert expected in html
    assert expected in text
    assert text.index(expected) < text.index("Sygnały ponad próg")


def test_header_without_multiplier_shows_one_threshold() -> None:
    from src.application.report_regime import RegimeBanner
    from src.domain.regime import MarketRegime

    banner = RegimeBanner(
        regime=MarketRegime.RISK_ON,
        multiplier=1.0,
        base_threshold=Decimal("0.02"),
        reason="wszystkie sygnały NORMAL",
    )

    _, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(regime=banner),
    )

    assert "Reżim: RISK_ON · próg ×1,0 (2,0%) · powód: wszystkie sygnały NORMAL" in text


def test_header_without_regime_has_no_regime_line() -> None:
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0, ReportContext()
    )

    assert "Reżim:" not in html
    assert "Reżim:" not in text


def test_regime_reason_is_escaped_in_html() -> None:
    from src.application.report_regime import RegimeBanner
    from src.domain.regime import MarketRegime

    banner = RegimeBanner(
        regime=MarketRegime.NEUTRAL,
        multiplier=1.0,
        base_threshold=Decimal("0.02"),
        reason="<b>x</b>",
    )

    html, _ = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(regime=banner),
    )

    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def _consensus(summary: str) -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation="BUY", consensus_strength=0.9, summary=summary,
        dissenting_views=(),
        investor_opinions=tuple(
            InvestorOpinion(investor_name=n, recommendation="BUY",
                            confidence=0.9, reasoning="x", key_factors=())
            for n in ("A", "B")
        ),
    )


def _first_element(html: str) -> str:
    # MAIL-57: mail jest kompletnym dokumentem — preheader to pierwszy element <body>.
    body = html[html.index("<body>") + len("<body>"):]
    return body[: body.index("</div>") + len("</div>")]


def test_html_starts_with_hidden_preheader_from_lead() -> None:
    # MAIL-13: podgląd w Gmailu/Apple Mail był „📊 StockAgent — raport cyklu Czas
    # rozpoczęcia: …”. Teraz pierwszy element to ukryty preheader z detail leadu.
    result = SymbolResult(
        symbol="AMD", status="saved", delta=Decimal("0.08"),
        current_price=Decimal("50"), council_verdict=_consensus("Popyt na GPU rośnie."),
    )

    html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    first = _first_element(html)
    assert first.startswith("<div ")
    assert "display:none" in first
    assert "mso-hide:all" in first
    assert "Popyt na GPU rośnie." in first
    assert "&zwnj;&nbsp;" in first


def test_preheader_escapes_the_lead_detail() -> None:
    # Detail pochodzi z rady i newsów — w preheaderze przechodzi przez html.escape.
    result = SymbolResult(
        symbol="AMD", status="saved", delta=Decimal("0.08"), current_price=Decimal("50"),
        council_verdict=_consensus("<img src=x onerror=alert(1)> & marże"),
    )

    html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    first = _first_element(html)
    assert "&lt;img src=x onerror=alert(1)&gt; &amp; marże" in first
    assert "<img" not in html


def test_preheader_without_lead_counts_closed_predictions() -> None:
    # Zamknięte predykcje symboli spoza wyników cyklu nie tworzą leadu.
    resolved = [
        ResolvedPrediction(symbol=s, predicted_trend="BULLISH", is_correct=True)
        for s in ("AAPL", "TSLA")
    ]

    html, _ = build_html_report(
        [_ignored_result("MSFT")], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(resolved_predictions=resolved),
    )

    first = _first_element(html)
    assert "display:none" in first
    assert "Brak sygnałów ponad próg · 2 zamkniętych predykcji" in first


def test_plain_text_has_no_preheader() -> None:
    _, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    assert "&zwnj;" not in text
    assert "Brak sygnałów ponad próg" not in text


class TestResolvedSectionWindowHeading:
    """WIA-19: za flagą `settled_at_windows_enabled` sekcja pokazuje predykcje
    rozliczone w TYM cyklu (po `settled_at`), więc nagłówek nie może dalej
    obiecywać „ostatnich 24h”, których okno już nie liczy."""

    _RESOLVED = [ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH", is_correct=True)]

    def test_cycle_window_heading(self) -> None:
        html, text = build_html_report(
            [], datetime(2026, 10, 19, 5, 30, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=self._RESOLVED, resolved_this_cycle=True),
        )

        assert "Zamknięte predykcje (rozliczone w tym cyklu)" in html
        assert "ZAMKNIĘTE PREDYKCJE (rozliczone w tym cyklu)" in text
        assert "ostatnie 24h" not in html
        assert "ostatnie 24h" not in text

    def test_default_heading_keeps_the_24h_window(self) -> None:
        html, text = build_html_report(
            [], datetime(2026, 10, 19, 5, 30, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=self._RESOLVED),
        )

        assert "Zamknięte predykcje (ostatnie 24h)" in html
        assert "ZAMKNIĘTE PREDYKCJE (ostatnie 24h)" in text
        assert "rozliczone w tym cyklu" not in html


class TestFooter:
    """MAIL-42: stopka mówi, który przebieg i który commit wygenerował mail,
    a w GitHub Actions linkuje logi przebiegu. Logi repo są publiczne, więc
    stopka niesie WYŁĄCZNIE numer przebiegu, krótki SHA i link."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
    _GHA = RunContext(
        run_id="77",
        git_sha="abc1234def",
        run_url="https://github.com/x/y/actions/runs/77",
    )

    def test_footer_links_run_and_commit(self) -> None:
        html, _ = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=self._GHA)
        )

        assert 'href="https://github.com/x/y/actions/runs/77"' in html
        assert "przebieg #77" in html
        assert "commit abc1234" in html
        # Krótki SHA, nie pełny.
        assert "abc1234def" not in html

    def test_plain_text_footer_carries_run_commit_and_logs_url(self) -> None:
        _, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=self._GHA)
        )

        assert text.rstrip().endswith(
            "przebieg #77 · commit abc1234 · logi: https://github.com/x/y/actions/runs/77"
        )

    def test_no_run_context_means_no_link_and_unchanged_text(self) -> None:
        html, text = build_html_report([], self._STARTED, 1.0, ReportContext())

        assert "actions/runs" not in html
        assert "przebieg #" not in html
        assert "przebieg #" not in text
        assert "raport automatyczny" in html

    def test_local_run_has_no_logs_link(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=RunContext())
        )

        assert "commit local" in html
        assert ">logi</a>" not in html
        assert "commit local" in text
        assert "logi:" not in text

    def test_footer_escapes_values_and_drops_unsafe_url(self) -> None:
        hostile = RunContext(
            run_id='<script>1</script>', git_sha="abc1234", run_url="javascript:alert(1)"
        )

        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=hostile)
        )

        assert "<script>1" not in html
        assert "&lt;script&gt;" in html
        assert "javascript:" not in html
        assert "javascript:" not in text


class TestMoveIntervalInReport:
    """ML-34 — przedział ruchu w mailu od razu (liczony w cyklu, bez migracji):
    „80% przedział: −2.1% … +2.4% (n=57)", a przy krótkim oknie jawne „za mało
    historii (n/20)" zamiast liczby."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    @staticmethod
    def _with_interval(**interval_fields: Any) -> SymbolResult:
        import dataclasses

        from src.domain.conformal import MoveInterval

        return dataclasses.replace(
            _saved_result(), move_interval=MoveInterval(**interval_fields)
        )

    def test_interval_reaches_html_and_text(self) -> None:
        result = self._with_interval(low=-0.021, high=0.024, calibration_size=57)

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "80% przedział: -2,1% … +2,4% (n=57)" in html
        assert "80% przedział: -2,1% … +2,4% (n=57)" in text

    def test_short_window_says_so_instead_of_a_number(self) -> None:
        result = self._with_interval(low=None, high=None, calibration_size=12)

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "80% przedział: za mało historii (12/20)" in html
        assert "80% przedział: za mało historii (12/20)" in text

    def test_no_interval_no_line(self) -> None:
        html, text = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert "przedział:" not in html
        assert "przedział:" not in text

    def test_interval_shows_even_without_reasoning(self) -> None:
        import dataclasses

        result = dataclasses.replace(
            self._with_interval(low=-0.01, high=0.012, calibration_size=30), reasoning=None
        )

        html, _ = build_html_report([result], self._STARTED, 1.0)

        assert "80% przedział: -1,0% … +1,2% (n=30)" in html

    def test_to_symbol_result_carries_the_interval_from_the_graph(self) -> None:
        from src.domain.conformal import MoveInterval

        interval = MoveInterval(low=-0.02, high=0.03, calibration_size=40)

        result = to_symbol_result("AAPL", {"status": "saved", "move_interval": interval})

        assert result.move_interval == interval

    def test_to_symbol_result_ignores_a_malformed_interval(self) -> None:
        result = to_symbol_result("AAPL", {"status": "saved", "move_interval": "x"})

        assert result.move_interval is None


class TestInvalidationInReport:
    """ML-27 — poziom unieważnienia tezy pod uzasadnieniem: „Teza upada poniżej
    $290,00 — …” w HTML i w tekście. Bez poziomu (flaga off) mail bez zmian."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    @staticmethod
    def _with_level(price: str, reason: str = "Zejście pod 290 obala wzrost.") -> SymbolResult:
        from src.domain.invalidation import Invalidation

        return dataclasses.replace(
            _saved_result(), invalidation=Invalidation(price=Decimal(price), reason=reason)
        )

    def test_level_below_the_price_reaches_html_and_text(self) -> None:
        html, text = build_html_report([self._with_level("290")], self._STARTED, 1.0)

        assert "Teza upada poniżej $290,00 — Zejście pod 290 obala wzrost." in html
        assert "Teza upada poniżej $290,00 — Zejście pod 290 obala wzrost." in text

    def test_level_above_the_price_says_above(self) -> None:
        html, _ = build_html_report([self._with_level("310", reason="")], self._STARTED, 1.0)

        assert "Teza upada powyżej $310,00" in html
        assert "Teza upada powyżej $310,00 —" not in html

    def test_reason_is_escaped(self) -> None:
        html, _ = build_html_report(
            [self._with_level("290", reason="<b>wsparcie</b>")], self._STARTED, 1.0
        )

        assert "&lt;b&gt;wsparcie&lt;/b&gt;" in html
        assert "<b>wsparcie</b>" not in html

    def test_no_level_no_line(self) -> None:
        html, text = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert "Teza upada" not in html
        assert "Teza upada" not in text

    def test_to_symbol_result_carries_the_level_from_the_graph(self) -> None:
        from src.domain.invalidation import Invalidation

        level = Invalidation(price=Decimal("290"), reason="x")

        carried = to_symbol_result("AAPL", {"status": "saved", "invalidation": level})
        malformed = to_symbol_result("AAPL", {"status": "saved", "invalidation": "x"})

        assert carried.invalidation == level
        assert malformed.invalidation is None


class TestInvestmentDisclaimer:
    """MAIL-08 (G21): „Sygnał modelu: KUP” i bandy % portfela nie mogą iść bez
    zastrzeżenia. Stopka HTML i koniec tekstu niosą je ZAWSZE — także w raporcie
    bez predykcji, w raporcie subskrybenta (fan-out) i w HTML-u web digestu."""

    _DISCLAIMER = "nie stanowi rekomendacji inwestycyjnej"
    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
    _RUN = RunContext(run_id="77", git_sha="abc1234def")

    @pytest.mark.parametrize(
        ("results", "context"),
        [
            ([], ReportContext()),
            ([_saved_result(), _ignored_result(), _error_result()], ReportContext()),
            ([_saved_result()], ReportContext(symbols_filter=frozenset({"MSFT"}))),
            ([_saved_result()], ReportContext(run_context=_RUN)),
        ],
        ids=["empty-cycle", "full-cycle", "subscriber-fan-out", "with-run-footer"],
    )
    def test_html_and_plain_always_carry_disclaimer(
        self, results: list[SymbolResult], context: ReportContext
    ) -> None:
        html, text = build_html_report(results, self._STARTED, 1.0, context)

        assert self._DISCLAIMER in html
        assert self._DISCLAIMER in text

    def test_html_disclaimer_sits_in_the_footer_after_the_predictions(self) -> None:
        html, _ = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert html.index("Wygenerowane predykcje") < html.index(self._DISCLAIMER)
        assert html.index(self._DISCLAIMER) < html.index("raport automatyczny")

    def test_plain_disclaimer_closes_the_body_before_the_run_footer(self) -> None:
        _, text = build_html_report(
            [_saved_result()], self._STARTED, 1.0, ReportContext(run_context=self._RUN)
        )

        assert text.index("Strong macro tailwinds") < text.index(self._DISCLAIMER)
        assert text.index(self._DISCLAIMER) < text.index("przebieg #77")


def test_predictions_table_says_model_signal_not_recommendation() -> None:
    # MAIL-27: zastrzeżenie „nie stanowi rekomendacji” obok nagłówka
    # „Rekomendacja” było sprzeczne — tabela i blok uzasadnień mówią o sygnale.
    html, _ = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    predictions = html.split("Wygenerowane predykcje")[1]
    assert "Sygnał modelu: KUP" in predictions
    assert ">Rekomendacja<" not in html
    assert "Rekomendacja: KUP" not in html


def test_plain_text_says_model_signal_not_recommendation() -> None:
    _, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    assert "Sygnał modelu: KUP (pewność 85%" in text
    assert "Rekomendacja" not in text



class TestLeadInPolish:
    """MAIL-15: lead i temat maila mówią po polsku — angielskie BULLISH/BUY
    zostają w logach i w bazie, nie w treści."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def test_resolved_label_names_the_forecast_in_polish(self) -> None:
        results = [
            SymbolResult(symbol="NVDA", status="saved", delta=Decimal("0.03")),
            SymbolResult(symbol="AMD", status="saved", delta=Decimal("-0.02")),
            SymbolResult(symbol="IBM", status="saved", delta=Decimal("0.001")),
        ]
        resolved = [
            ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH", is_correct=True),
            ResolvedPrediction(symbol="AMD", predicted_trend="BEARISH", is_correct=False),
            ResolvedPrediction(symbol="IBM", predicted_trend="SIDEWAYS", is_correct=True),
        ]

        labels = {s.symbol: s.resolved_label for s in to_lead_signals(results, resolved)}

        assert labels == {
            "NVDA": "trafiona prognoza wzrostu",
            "AMD": "chybiona prognoza spadku",
            "IBM": "trafiona prognoza ruchu bocznego",
        }

    def test_closed_prediction_lead_has_no_english_trend(self) -> None:
        results = [SymbolResult(symbol="NVDA", status="saved", delta=Decimal("0.03"))]
        resolved = [
            ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH", is_correct=True)
        ]

        html, text = build_html_report(
            results, self._STARTED, 1.0, ReportContext(resolved_predictions=resolved)
        )

        assert "NVDA +3,0%: trafiona prognoza wzrostu" in html
        assert "NVDA +3,0%: trafiona prognoza wzrostu" in text
        assert "BULLISH" not in html
        assert "BULLISH" not in text

    def test_quota_banner_severity_is_polish(self) -> None:
        alert = QuotaAlert(
            source="Resend", severity=QuotaSeverity.CRITICAL,
            message="Resend zwrócił 429.", action="Poczekaj na reset.",
            occurred_at=self._STARTED,
        )

        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(quota_alerts=[alert])
        )

        assert "KRYTYCZNY" in html
        assert "KRYTYCZNY" in text
        assert "CRITICAL" not in html
        assert "CRITICAL" not in text



class TestResolvedSectionWhenNothingSettled:
    """MAIL-25: pusta sekcja nie odróżniała „nic się nie zamknęło” od awarii
    odczytu. Pusta lista rozliczeń → jawna linia zgodna z oknem nagłówka
    (WIA-19); awaria odczytu → ostrzeżenie, nigdy „brak rozliczeń”."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def test_cycle_window_says_nothing_settled_in_this_cycle(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0,
            ReportContext(resolved_predictions=[], resolved_this_cycle=True),
        )

        assert "Zamknięte predykcje (rozliczone w tym cyklu)" in html
        assert "Brak rozliczeń w tym cyklu." in html
        assert "ZAMKNIĘTE PREDYKCJE (rozliczone w tym cyklu)" in text
        assert "Brak rozliczeń w tym cyklu." in text
        assert "ostatnich 24h" not in html

    def test_24h_window_keeps_its_own_wording(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(resolved_predictions=[])
        )

        assert "Zamknięte predykcje (ostatnie 24h)" in html
        assert "ZAMKNIĘTE PREDYKCJE (ostatnie 24h)" in text
        assert "Brak rozliczeń predykcji z ostatnich 24h." in text
        assert "w tym cyklu" not in text

    def test_read_failure_is_not_dressed_up_as_no_settlements(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0,
            ReportContext(resolved_predictions=[], resolved_unavailable=True),
        )

        assert "Nie udało się odczytać rozliczeń" in html
        assert "Nie udało się odczytać rozliczeń" in text
        assert "Brak rozliczeń" not in html
        assert "Brak rozliczeń" not in text

    def test_preheader_does_not_count_zero_settlements_on_read_failure(self) -> None:
        html, _ = build_html_report(
            [], self._STARTED, 1.0, ReportContext(resolved_unavailable=True)
        )

        first = _first_element(html)
        assert "Brak sygnałów ponad próg · rozliczenia niedostępne" in first
        assert "0 zamkniętych predykcji" not in first

    def test_report_without_settlement_data_keeps_no_section(self) -> None:
        # Wołający, który nie dociąga rozliczeń (None), nie dostaje fałszywego
        # „brak rozliczeń” — tylko pusta lista z udanego odczytu jest informacją.
        html, text = build_html_report([], self._STARTED, 1.0, ReportContext())

        assert "Zamknięte predykcje" not in html
        assert "ZAMKNIĘTE PREDYKCJE" not in text


class TestStaleReferenceReport:
    """FIN-14 — symbol z referencją sprzed pauzy nie jest „poniżej progu”: cykl
    zapisał nowy punkt odniesienia bez analizy, a raport mówi to wprost."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
    _REFERENCE_AT = datetime(2026, 9, 19, 17, 30, tzinfo=UTC)

    def _raw(self, *, stale: bool) -> dict[str, object]:
        return {
            "status": "ignored",
            "delta": Decimal("-0.10"),
            "previous_price": Decimal("100"),
            "current_price": Decimal("90"),
            "reference_at": self._REFERENCE_AT,
            "stale_reference": stale,
        }

    def test_mapper_marks_stale_reference_and_keeps_its_date(self) -> None:
        result = to_symbol_result("AAPL", self._raw(stale=True), now=self._STARTED)

        assert result.skip_reason is SkipReason.STALE_REFERENCE
        assert result.reference_at == self._REFERENCE_AT

    def test_mapper_keeps_below_threshold_for_a_fresh_reference(self) -> None:
        result = to_symbol_result("AAPL", self._raw(stale=False), now=self._STARTED)

        assert result.skip_reason is SkipReason.BELOW_THRESHOLD
        assert result.reference_at == self._REFERENCE_AT

    def test_report_says_new_reference_point_with_the_reference_date(self) -> None:
        result = to_symbol_result("AAPL", self._raw(stale=True), now=self._STARTED)

        html, text = build_html_report([result], self._STARTED, 1.0)

        expected = "referencja sprzed 19.09.2026 — nowy punkt odniesienia"
        assert expected in html
        assert expected in text

    def test_stale_symbol_is_not_listed_below_the_volatility_threshold(self) -> None:
        result = to_symbol_result("AAPL", self._raw(stale=True), now=self._STARTED)

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "poniżej progu zmienności" not in html
        assert "poniżej progu zmienności" not in text

    def test_crypto_section_calls_a_stale_symbol_a_new_reference_point(self) -> None:
        result = SymbolResult(
            symbol="BTC", status="ignored", delta=Decimal("0.12"),
            current_price=Decimal("65000"), asset_class="CRYPTO",
            skip_reason=SkipReason.STALE_REFERENCE, reference_at=self._REFERENCE_AT,
        )

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "poniżej progu" not in html
        assert "poniżej progu" not in text
        assert "nowy punkt odniesienia" in html.split("🪙 Krypto")[1]
        assert "nowy punkt odniesienia" in text.split("KRYPTO")[1]


class TestPortfolioChangeInReport:
    """PORT-06: zmiana portfela od ostatniego maila w sekcji 💼 i na pierwszym
    ekranie. Cena poprzednia odtwarzana z delty cyklu (prev = current / (1 + Δ)),
    także dla symboli pominiętych bramką; cold-start nie ma punktu odniesienia."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    @staticmethod
    def _portfolio(*positions: tuple[str, str]) -> Portfolio:
        return Portfolio(
            positions=tuple(
                Position(symbol=symbol, quantity=Decimal(qty), purchase_price=Decimal("5"),
                         purchase_date=datetime(2026, 3, 2).date())
                for symbol, qty in positions
            ),
            as_of=datetime(2026, 10, 18, tzinfo=UTC),
        )

    def _results(self, **overrides: Any) -> list[SymbolResult]:
        nvda = SymbolResult(
            symbol="NVDA", status="saved", delta=Decimal("0.10"),
            current_price=Decimal("11"), trend="BULLISH", confidence_score=0.7,
            target_price=Decimal("12"),
        )
        aapl = SymbolResult(
            symbol="AAPL", status="ignored", skip_reason=SkipReason.BELOW_THRESHOLD,
            delta=Decimal("-0.02"), current_price=Decimal("9.80"),
        )
        return [dataclasses.replace(nvda, **overrides), aapl]

    def test_first_screen_box_and_section_show_the_change(self) -> None:
        context = ReportContext(user_portfolio=self._portfolio(("NVDA", "75"), ("AAPL", "25")))

        html, text = build_html_report(self._results(), self._STARTED, 1.0, context)

        assert "Twój portfel od ostatniego maila" in html
        assert "+7,0%" in html
        assert "Zmiana od ostatniego maila: <strong" in html
        assert "NVDA +$75,00, AAPL -$5,00" in html
        assert "Twój portfel od ostatniego maila: +7,0% (+$70,00)" in text
        assert "Zmiana od ostatniego maila: +$70,00 (+7,0%)" in text

    def test_first_screen_box_comes_before_the_portfolio_section(self) -> None:
        context = ReportContext(user_portfolio=self._portfolio(("NVDA", "75"), ("AAPL", "25")))

        html, text = build_html_report(self._results(), self._STARTED, 1.0, context)

        assert html.index("Twój portfel od ostatniego maila") < html.index("💼 Twój portfel")
        assert text.index("Twój portfel od ostatniego maila") < text.index("=== 💼 Twój portfel")

    def test_cold_start_position_is_left_out_of_the_change(self) -> None:
        context = ReportContext(
            user_portfolio=self._portfolio(("NVDA", "75"), ("AAPL", "25"), ("NEW", "1"))
        )
        new = SymbolResult(
            symbol="NEW", status="ignored", skip_reason=SkipReason.COLD_START,
            delta=Decimal("0"), current_price=Decimal("50"),
        )

        html, _ = build_html_report([*self._results(), new], self._STARTED, 1.0, context)

        assert "+7,0%" in html
        assert "bez punktu odniesienia: NEW" in html

    def test_after_a_pause_the_change_is_dated(self) -> None:
        reference_at = datetime(2026, 9, 19, 17, 30, tzinfo=UTC)
        context = ReportContext(user_portfolio=self._portfolio(("NVDA", "75"), ("AAPL", "25")))
        results = [
            dataclasses.replace(
                r, status="ignored", skip_reason=SkipReason.STALE_REFERENCE,
                reference_at=reference_at,
            )
            for r in self._results()
        ]

        html, text = build_html_report(results, self._STARTED, 1.0, context)

        assert "Twój portfel od 19.09.2026" in html
        assert "Zmiana od 19.09.2026" in html
        assert "Twój portfel od 19.09.2026: +7,0%" in text
        assert "od ostatniego maila" not in html

    def test_no_portfolio_no_change(self) -> None:
        html, text = build_html_report(self._results(), self._STARTED, 1.0, ReportContext())

        assert "od ostatniego maila" not in html
        assert "od ostatniego maila" not in text

    def test_legacy_push_text_has_no_change_line(self) -> None:
        context = ReportContext(
            user_portfolio=self._portfolio(("NVDA", "75"), ("AAPL", "25")),
            user_portfolio_in_text=False,
        )

        html, text = build_html_report(self._results(), self._STARTED, 1.0, context)

        assert "Twój portfel od ostatniego maila" in html
        assert "od ostatniego maila" not in text
        assert "$70,00" not in text


class TestFirstScreenReaderNumbers:
    """MAIL-16: pierwszy ekran odpowiada na pytania czytelnika — co z moim
    portfelem, ile sygnałów ponad próg, czy model ostatnio trafiał. Liczniki
    pracy pipeline'u (pominięte, błędy, czas trwania, liczba symboli) są
    w stopce technicznej, w HTML i w tekście."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
    _DISCLAIMER = "nie stanowi rekomendacji inwestycyjnej"

    @staticmethod
    def _resolved(correct: int, wrong: int) -> list[ResolvedPrediction]:
        return [
            ResolvedPrediction(symbol=f"C{i}", predicted_trend="BULLISH", is_correct=True)
            for i in range(correct)
        ] + [
            ResolvedPrediction(symbol=f"W{i}", predicted_trend="BEARISH", is_correct=False)
            for i in range(wrong)
        ]

    @staticmethod
    def _saved(symbol: str, trend: str) -> SymbolResult:
        return SymbolResult(
            symbol=symbol, status="saved", delta=Decimal("0.03"),
            current_price=Decimal("100"), trend=trend, target_price=Decimal("102"),
            confidence_score=0.7,
        )

    def test_first_screen_shows_resolved_hit_count_before_counters(self) -> None:
        results = [_saved_result(), _ignored_result(), _error_result()]
        context = ReportContext(resolved_predictions=self._resolved(3, 2))

        html, text = build_html_report(results, self._STARTED, 1.0, context)

        assert "3/5" in html
        assert html.index("3/5") < html.index("Pominięte")
        assert text.index("Rozliczone: 3/5 trafionych") < text.index("pominiętych")

    def test_signals_box_counts_saved_predictions_with_a_direction(self) -> None:
        results = [
            self._saved("AAPL", "BULLISH"),
            self._saved("MSFT", "BEARISH"),
            self._saved("GOOGL", "SIDEWAYS"),
            _ignored_result("NVDA"),
        ]

        html, text = build_html_report(results, self._STARTED, 1.0)

        signals_box = html.split("Sygnały ponad próg")[1][:200]
        assert ">2</div>" in signals_box
        assert "Sygnały ponad próg: 2" in text

    def test_reader_boxes_come_in_order_portfolio_signals_settlements(self) -> None:
        portfolio = Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("2"), purchase_price=Decimal("90"),
                         purchase_date=datetime(2026, 3, 2).date()),
            ),
            as_of=datetime(2026, 10, 18, tzinfo=UTC),
        )
        context = ReportContext(
            user_portfolio=portfolio, resolved_predictions=self._resolved(1, 1)
        )

        html, text = build_html_report(
            [self._saved("AAPL", "BULLISH")], self._STARTED, 1.0, context
        )

        assert (
            html.index("Twój portfel od ostatniego maila")
            < html.index("Sygnały ponad próg")
            < html.index("Rozliczone")
        )
        assert (
            text.index("Twój portfel od ostatniego maila")
            < text.index("Sygnały ponad próg")
            < text.index("Rozliczone: 1/2 trafionych")
        )

    def test_without_a_portfolio_the_portfolio_box_is_hidden(self) -> None:
        html, text = build_html_report(
            [self._saved("AAPL", "BULLISH")], self._STARTED, 1.0,
            ReportContext(resolved_predictions=self._resolved(1, 0)),
        )

        assert "Twój portfel" not in html
        assert "Twój portfel" not in text
        assert "Sygnały ponad próg" in html
        assert "1/1" in html

    def test_operational_counters_move_to_the_technical_footer(self) -> None:
        results = [_saved_result(), _ignored_result(), _ignored_result("GOOGL"), _error_result()]

        html, text = build_html_report(results, self._STARTED, 12.5)

        header = html[: html.index("Sygnały ponad próg")]
        assert "Czas trwania" not in header
        assert "Liczba symboli" not in header
        footer = html[html.index(self._DISCLAIMER):]
        assert "Symboli: <strong>4</strong>" in footer
        assert "Predykcje: <strong>1</strong>" in footer
        assert "Pominięte: <strong>2</strong>" in footer
        assert "Błędy: <strong>1</strong>" in footer
        assert "Czas trwania: 12,5s" in footer
        text_footer = text[text.index(self._DISCLAIMER):]
        assert (
            "Symboli: 4 · predykcji: 1 · pominiętych: 2 · błędów: 1 · czas trwania: 12,5s"
            in text_footer
        )
        assert "Czas trwania" not in text[: text.index(self._DISCLAIMER)]

    def test_no_settlements_box_says_none_instead_of_zero_of_zero(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(resolved_predictions=[])
        )

        assert "0/0" not in html
        assert "Rozliczone: brak" in text

    def test_unreadable_settlements_box_says_unavailable(self) -> None:
        html, text = build_html_report(
            [], self._STARTED, 1.0,
            ReportContext(resolved_predictions=None, resolved_unavailable=True),
        )

        settlements_box = html.split("Rozliczone")[1][:300]
        assert "niedostępne" in settlements_box
        assert "Rozliczone: niedostępne" in text


def test_trade_signals_precede_track_record() -> None:
    # MAIL-17: Track Record stoi w „Zapleczu” — przy przycięciu przez Gmaila
    # dolna część maila znika pierwsza, więc akcyjne sygnały muszą być wyżej.
    curve = EquityCurve(
        points=(EquityPoint(index=0, period_return=0.01, equity=1.01),),
        total_return=0.01, win_rate=1.0, max_drawdown=0.0, n_trades=1,
    )

    html, _ = build_html_report(
        [_saved_result()], datetime(2026, 10, 19, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(equity_curve=curve),
    )

    assert html.index("Najsilniejsze sygnały") < html.index("Track Record")


# PORT-04: VaR / obsunięcie / hedge zwężają bandę WYŁĄCZNIE przy realnych
# pozycjach. Orkiestrator podaje `position_risk` tylko za `portfolio_var_enabled`.

_SUFFICIENT_ACCURACY = {
    "mean_accuracy": 0.7, "sample_count": 200, "correct_count": 140,
    "days_window": 30, "n_eff": 25.0,
}


def _owner_portfolio() -> Portfolio:
    # Pozycja w innym symbolu niż sygnał — ekspozycja NVDA nie zwęża bandy.
    return Portfolio(
        positions=(Position("AAPL", Decimal("1"), Decimal("100"), datetime(2026, 1, 2).date()),),
        as_of=datetime(2026, 5, 14, tzinfo=UTC),
    )


def _risk_sizing_report(**context: object) -> str:
    html, _ = build_html_report(
        [_strong_buy_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(accuracy_stats=_SUFFICIENT_ACCURACY, **context),  # type: ignore[arg-type]
    )
    return html


def test_portfolio_var_of_real_positions_narrows_the_band() -> None:
    from src.application.use_cases.portfolio_risk import PositionRisk

    html = _risk_sizing_report(
        user_portfolio=_owner_portfolio(), position_risk=PositionRisk(var=0.05)
    )

    assert "Pozycja pełna" not in html
    assert "Pozycja standardowa 2-3% portfela" in html


def test_deep_portfolio_drawdown_forces_the_starter_band() -> None:
    from src.application.use_cases.portfolio_risk import PositionRisk

    html = _risk_sizing_report(
        user_portfolio=_owner_portfolio(), position_risk=PositionRisk(drawdown=0.2)
    )

    assert "Pozycja startowa 1-2% portfela" in html


def test_position_risk_without_real_positions_leaves_the_band() -> None:
    from src.application.use_cases.portfolio_risk import PositionRisk

    html = _risk_sizing_report(position_risk=PositionRisk(var=0.05, drawdown=0.2))

    assert "Pozycja pełna 4-5% portfela" in html


def test_equal_weight_radar_var_never_narrows_the_band() -> None:
    from src.application.use_cases.portfolio_risk import PortfolioRiskReport

    html = _risk_sizing_report(
        user_portfolio=_owner_portfolio(),
        portfolio_risk_report=PortfolioRiskReport(watchlist_size=3, var=0.2),
    )

    assert "Pozycja pełna 4-5% portfela" in html



# PORT-05: etykiety sektorów i peerzy Warte uwagi idą wyłącznie z mapy configu
# w `ReportContext` — w kodzie nie ma zapasowej kopii.


def test_sector_label_comes_only_from_the_context_map() -> None:
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
    )

    assert "Big Tech" not in html
    assert "Big Tech" not in text


def test_renamed_sector_keeps_its_suggestions() -> None:
    hot = SymbolResult(
        symbol="CRWD", status="saved", delta=Decimal("0.06"),
        current_price=Decimal("769"), trend="BULLISH",
        target_price=Decimal("780"), confidence_score=0.75,
    )

    html, _ = build_html_report(
        [hot], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(sector_map={"CRWD": "Cyber"}, sector_peers={"Cyber": ["ZS"]}),
    )

    assert "Warte uwagi" in html
    assert "Cyber" in html
    assert "ZS" in html


class TestPortfolioInOneCurrency:
    """Wagi portfela (lead, sizing, sekcja 💼) i zmiana od ostatniego maila liczone
    w złotych kursami NBP — nie suma dolarów z euro. Pozycja bez kursu wypada
    z wag i sum z jawną notą, nigdy nie jest liczona jak USD."""

    _STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)
    _RATES = PlnRates(current={"USD": Decimal("4.00"), "EUR": Decimal("4.30")})

    @staticmethod
    def _portfolio() -> Portfolio:
        # AAPL 43 × 100 USD × 4,00 = 17 200 zł = SAP.DE 40 × 100 EUR × 4,30.
        # Natywnie 4300 „+” 4000 dawało AAPL 52% portfela.
        return Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("43"), purchase_price=Decimal("90")),
                Position(
                    symbol="SAP.DE",
                    quantity=Decimal("40"),
                    purchase_price=Decimal("90"),
                    currency="EUR",
                ),
            ),
            as_of=datetime(2026, 10, 18, tzinfo=UTC),
        )

    @staticmethod
    def _results(aapl_delta: str = "0.05") -> list[SymbolResult]:
        return [
            SymbolResult(symbol="AAPL", status="saved", delta=Decimal(aapl_delta),
                         current_price=Decimal("100")),
            SymbolResult(symbol="SAP.DE", status="ignored",
                         skip_reason=SkipReason.BELOW_THRESHOLD, delta=Decimal("0"),
                         current_price=Decimal("100")),
        ]

    def _context(self, rates: PlnRates | None = None) -> ReportContext:
        return ReportContext(user_portfolio=self._portfolio(), portfolio_fx=rates or self._RATES)

    def test_lead_weight_of_a_position_is_its_share_in_pln(self) -> None:
        html, text = build_html_report(self._results(), self._STARTED, 1.0, self._context())

        assert "AAPL (50% portfela)" in html
        assert "AAPL (50% portfela)" in text

    def test_sizing_sees_weights_and_sector_exposure_in_pln(self, mocker: MockerFixture) -> None:
        signals = mocker.patch(
            "src.application.report_builder.build_trade_signals", return_value=[]
        )
        context = dataclasses.replace(
            self._context(), portfolio_clusters={"AAPL": "Mix", "SAP.DE": "Mix"}
        )

        build_html_report(self._results(), self._STARTED, 1.0, context)

        kwargs = signals.call_args.kwargs
        assert kwargs["portfolio_weights"] == {"AAPL": Decimal("0.5"), "SAP.DE": Decimal("0.5")}
        assert kwargs["cluster_exposures"] == {"AAPL": Decimal("1"), "SAP.DE": Decimal("1")}

    def test_section_weights_are_shares_in_pln(self) -> None:
        _, text = build_html_report(self._results(), self._STARTED, 1.0, self._context())

        rows = [line for line in text.splitlines() if " waga " in line]
        assert len(rows) == 2
        assert all("waga  50,0%" in row for row in rows), rows

    def test_position_without_its_rate_is_out_of_the_weights_with_a_note(self) -> None:
        context = self._context(PlnRates(current={"USD": Decimal("4.00")}))

        html, text = build_html_report(self._results(), self._STARTED, 1.0, context)

        assert "AAPL (100% portfela)" in html
        assert "Bez kursu NBP: SAP.DE — poza wagami i sumami w zł" in html
        assert "Bez kursu NBP: SAP.DE — poza wagami i sumami w zł" in text
        sap_row = next(line for line in text.splitlines() if line.startswith("  SAP.DE   waga"))
        assert "waga      —" in sap_row

    def test_change_since_the_last_email_is_measured_in_pln(self) -> None:
        # AAPL 95 → 100 USD (Δ 5,26%) po 4,00, SAP.DE bez zmian po 4,30:
        # 43 × 5 × 4,00 = +860 zł na 16 340 + 17 200 zł → +2,6% (natywnie +2,7%).
        results = self._results(aapl_delta=str(Decimal("100") / Decimal("95") - 1))

        html, text = build_html_report(results, self._STARTED, 1.0, self._context())

        assert "Zmiana od ostatniego maila: +860,00 zł (+2,6%)" in text
        assert "najwięcej: AAPL +860,00 zł" in text
        assert "Twój portfel od ostatniego maila: +2,6% (+860,00 zł)" in text
        assert "+860,00 zł" in html
        assert "$215" not in text

    def test_change_leaves_out_a_position_without_its_rate(self) -> None:
        results = self._results(aapl_delta=str(Decimal("100") / Decimal("95") - 1))
        context = self._context(PlnRates(current={"USD": Decimal("4.00")}))

        _, text = build_html_report(results, self._STARTED, 1.0, context)

        # Samo AAPL: +860 zł na 16 340 zł → +5,3%; SAP.DE nie udaje „bez zmian”.
        assert "Zmiana od ostatniego maila: +860,00 zł (+5,3%)" in text
        assert "bez punktu odniesienia: SAP.DE" not in text

    def test_without_any_rate_the_report_keeps_native_amounts(self) -> None:
        context = ReportContext(user_portfolio=self._portfolio())

        _, text = build_html_report(self._results(), self._STARTED, 1.0, context)

        # Bez kursów zostaje stara ścieżka: kwoty w walucie notowania.
        assert "Twój portfel od ostatniego maila: +2,5% (+$204,76)" in text


class TestForecastHorizonInPredictions:
    """ALR-15: sekcja predykcji mówi, po zamknięciu której sesji NYSE następny cykl
    rozliczy prognozę — kalendarz zna święta i skrócone sesje."""

    # Thanksgiving (czw 26.11.2026), 05:30 ET — najbliższa sesja to skrócony piątek.
    _STARTED = datetime(2026, 11, 26, 10, 30, tzinfo=UTC)
    _HORIZON = (
        "Horyzont prognozy: następny cykl · najbliższe zamknięcie NYSE: "
        "pt 27.11 13:00 ET (skrócona sesja)"
    )

    def test_html_names_the_next_session_close(self) -> None:
        html, _ = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert self._HORIZON in html

    def test_plain_text_names_the_next_session_close(self) -> None:
        _, text = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert self._HORIZON in text

    def test_cycle_without_predictions_has_no_horizon_line(self) -> None:
        html, text = build_html_report([_ignored_result()], self._STARTED, 1.0)

        assert "Horyzont prognozy" not in html
        assert "Horyzont prognozy" not in text


class TestBudgetExhaustedReport:
    """FIN-12 — symbol nad progiem, którego cykl nie przeanalizował, bo skończył
    się budżet płatnych wywołań LLM. To nie „poniżej progu”: ruch był, analizy nie."""

    _STARTED = datetime(2026, 10, 20, 5, 30, tzinfo=UTC)

    def _raw(self, **overrides: object) -> dict[str, object]:
        return {
            "status": "ignored",
            "delta": Decimal("-0.10"),
            "previous_price": Decimal("100"),
            "current_price": Decimal("90"),
            "budget_exhausted": True,
            **overrides,
        }

    def test_mapper_marks_the_budget_cutoff(self) -> None:
        result = to_symbol_result("NVDA", self._raw(), now=self._STARTED)

        assert result.status == "ignored"
        assert result.skip_reason is SkipReason.BUDGET_EXHAUSTED

    def test_report_names_the_cut_symbols_with_their_move(self) -> None:
        result = to_symbol_result("NVDA", self._raw(), now=self._STARTED)

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "Pominięte — wyczerpany budżet cyklu" in html
        assert "NVDA" in html.split("Pominięte — wyczerpany budżet cyklu")[1]
        assert "POMINIĘTE — WYCZERPANY BUDŻET CYKLU" in text
        assert "NVDA(-10,00%)" in text.split("POMINIĘTE — WYCZERPANY BUDŻET CYKLU")[1]

    def test_cut_symbol_is_not_listed_below_the_volatility_threshold(self) -> None:
        result = to_symbol_result("NVDA", self._raw(), now=self._STARTED)

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "poniżej progu zmienności" not in html
        assert "poniżej progu zmienności" not in text

    def test_below_threshold_symbols_stay_in_their_own_section(self) -> None:
        cut = to_symbol_result("NVDA", self._raw(), now=self._STARTED)
        flat = to_symbol_result(
            "MSFT",
            self._raw(delta=Decimal("0.005"), budget_exhausted=False),
            now=self._STARTED,
        )

        html, text = build_html_report([cut, flat], self._STARTED, 1.0)

        below = html.split("poniżej progu zmienności")[1]
        assert "MSFT" in below
        assert "NVDA" not in below

    def test_no_cutoff_no_section(self) -> None:
        html, text = build_html_report([_ignored_result()], self._STARTED, 1.0)

        assert "wyczerpany budżet cyklu" not in html
        assert "WYCZERPANY BUDŻET CYKLU" not in text

    def test_crypto_section_names_the_budget_cutoff(self) -> None:
        result = SymbolResult(
            symbol="BTC", status="ignored", delta=Decimal("0.12"),
            current_price=Decimal("65000"), asset_class="CRYPTO",
            skip_reason=SkipReason.BUDGET_EXHAUSTED,
        )

        html, text = build_html_report([result], self._STARTED, 1.0)

        assert "poniżej progu" not in html.split("🪙 Krypto")[1]
        assert "budżet cyklu" in html.split("🪙 Krypto")[1]
        assert "budżet cyklu" in text.split("KRYPTO")[1]


class TestColdStartReport:
    """MAIL-60 — pierwszy cykl symbolu to „brak punktu odniesienia”, nie „poniżej
    progu”: blok powitalny mówi o braku historii, więc lista pominiętych nie może
    twierdzić, że ruch był za mały. Δ = 0 przy istniejącej referencji to co innego."""

    _STARTED = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)

    @staticmethod
    def _cold(symbol: str, *, crypto: bool = False) -> SymbolResult:
        return to_symbol_result(
            symbol,
            {
                "status": "ignored",
                "delta": Decimal("0"),
                "previous_price": Decimal("0"),
                "current_price": Decimal("65000" if crypto else "140"),
                "asset": Asset(
                    symbol=symbol, asset_type=AssetType.CRYPTO if crypto else AssetType.STOCK
                ),
            },
            now=TestColdStartReport._STARTED,
        )

    @staticmethod
    def _flat(symbol: str) -> SymbolResult:
        return SymbolResult(
            symbol=symbol, status="ignored", skip_reason=SkipReason.BELOW_THRESHOLD,
            delta=Decimal("0"), current_price=Decimal("100"),
        )

    def test_cold_start_symbol_has_no_reference_point_not_a_small_move(self) -> None:
        html, text = build_html_report([self._cold("NVDA")], self._STARTED, 1.0)

        assert "brak punktu odniesienia" in html
        assert "brak punktu odniesienia" in text
        assert "poniżej progu zmienności" not in html
        assert "poniżej progu zmienności" not in text

    def test_cold_start_chip_does_not_pretend_the_move_was_zero(self) -> None:
        html, text = build_html_report([self._cold("NVDA")], self._STARTED, 1.0)

        assert "⏸ Pominięte (brak punktu odniesienia" in html
        assert "POMINIĘTE (brak punktu odniesienia" in text
        group_html = html.split("⏸ Pominięte (brak punktu odniesienia")[1]
        group_text = text.split("POMINIĘTE (brak punktu odniesienia")[1]
        assert "NVDA" in group_html
        assert "0,00%" not in group_html.split("</div>")[0]
        assert "NVDA" in group_text
        assert "0,00%" not in group_text.split("\n\n")[0]

    def test_crypto_section_says_there_is_no_reference_point(self) -> None:
        html, text = build_html_report([self._cold("BTC", crypto=True)], self._STARTED, 1.0)

        crypto_html = html.split("🪙 Krypto")[1].split("Zaplecze")[0]
        crypto_text = text.split("KRYPTO")[1].split("ZAPLECZE")[0]
        assert "brak punktu odniesienia (pierwszy cykl)" in crypto_html
        assert "brak punktu odniesienia" in crypto_text
        assert "poniżej progu" not in crypto_html
        assert "poniżej progu" not in crypto_text
        assert "0,00%" not in crypto_html
        assert "0,00%" not in crypto_text

    def test_flat_move_with_a_reference_is_still_below_the_threshold(self) -> None:
        html, text = build_html_report([self._flat("MSFT")], self._STARTED, 1.0)

        assert "poniżej progu zmienności" in html
        assert "poniżej progu zmienności" in text
        assert "brak punktu odniesienia" not in html
        assert "brak punktu odniesienia" not in text

    def test_mixed_cycle_keeps_the_two_reasons_in_separate_groups(self) -> None:
        results = [self._cold("NVDA"), self._flat("MSFT")]

        html, text = build_html_report(results, self._STARTED, 1.0)

        assert "⏸ Pominięte (brak punktu odniesienia" in html
        below_html, cold_html = html.split("⏸ Pominięte (brak punktu odniesienia")
        assert "MSFT" in below_html.split("⏸ Pominięte (poniżej progu zmienności)")[1]
        assert "NVDA" not in below_html.split("⏸ Pominięte (poniżej progu zmienności)")[1]
        assert "NVDA" in cold_html
        assert "MSFT" not in cold_html.split("</div>")[0]
        assert "POMINIĘTE (brak punktu odniesienia" in text
        below_text, cold_text = text.split("POMINIĘTE (brak punktu odniesienia")
        assert "MSFT" in below_text.split("POMINIĘTE (poniżej progu zmienności)")[1]
        assert "NVDA" not in below_text.split("POMINIĘTE (poniżej progu zmienności)")[1]
        assert "NVDA" in cold_text.split("\n\n")[0]

    def test_cold_start_symbol_is_not_a_near_threshold_move(self) -> None:
        cold = dataclasses.replace(self._cold("NVDA"), effective_threshold=Decimal("0.02"))

        html, text = build_html_report([cold], self._STARTED, 1.0)

        assert "Blisko progu" not in html
        assert "Blisko progu" not in text


class TestPositionAwareSignals:
    """PORT-08: przy znanym portfelu 🎯 sygnały mówią „ZREDUKUJ” o trzymanej
    spółce i „UNIKAJ” o nietrzymanej zamiast gołego „SPRZEDAJ” (sugestia shorta
    dla inwestora long-only)."""

    _AS_OF = datetime(2026, 5, 14, tzinfo=UTC)

    @staticmethod
    def _bearish(symbol: str) -> SymbolResult:
        return SymbolResult(
            symbol=symbol, status="saved", trend="BEARISH", delta=Decimal("-0.03"),
            confidence_score=0.8, current_price=Decimal("100"),
            target_price=Decimal("95"),
        )

    @staticmethod
    def _flat_msft() -> SymbolResult:
        return SymbolResult(
            symbol="MSFT", status="ignored", delta=Decimal("0"), current_price=Decimal("100")
        )

    def _portfolio(self) -> Portfolio:
        return Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("12"),
                         purchase_price=Decimal("100"), purchase_date=self._AS_OF.date()),
                Position(symbol="MSFT", quantity=Decimal("88"),
                         purchase_price=Decimal("100"), purchase_date=self._AS_OF.date()),
            ),
            as_of=self._AS_OF,
        )

    def _signals_html(self, context: ReportContext) -> str:
        html, _ = build_html_report(
            [self._bearish("AAPL"), self._bearish("TSLA"), self._flat_msft()],
            self._AS_OF, 1.0, context,
        )
        start = html.index("🎯 Najsilniejsze sygnały")
        return html[start:html.index("<h2", start + 1)]

    def test_bearish_signal_says_reduce_on_a_holding_and_avoid_elsewhere(self) -> None:
        section = self._signals_html(ReportContext(user_portfolio=self._portfolio()))

        assert "ZREDUKUJ (12% portfela)" in section
        assert "UNIKAJ" in section
        assert "SPRZEDAJ" not in section
        assert "nie rekomendacja" in section

    def test_plain_text_uses_the_same_position_labels(self) -> None:
        _, text = build_html_report(
            [self._bearish("AAPL"), self._bearish("TSLA"), self._flat_msft()],
            self._AS_OF, 1.0, ReportContext(user_portfolio=self._portfolio()),
        )

        signals = text.split("NAJSILNIEJSZE SYGNAŁY")[1].split("\n\n")[0]
        assert "ZREDUKUJ (12% portfela)" in signals
        assert "UNIKAJ" in signals
        assert "SPRZEDAJ" not in signals

    def test_without_a_known_portfolio_the_signal_is_unchanged(self) -> None:
        section = self._signals_html(ReportContext())

        assert "SPRZEDAJ" in section
        assert "UNIKAJ" not in section
        assert "ZREDUKUJ" not in section

    def test_avoid_signal_does_not_suggest_a_position_size(self) -> None:
        verdict = CouncilVerdict(
            final_recommendation="SELL", consensus_strength=0.9, summary="Rada: SELL.",
            dissenting_views=(),
            investor_opinions=(
                InvestorOpinion(investor_name="A", recommendation="SELL",
                                confidence=0.9, reasoning="x", key_factors=()),
            ),
        )
        unheld = dataclasses.replace(self._bearish("TSLA"), council_verdict=verdict)

        html, _ = build_html_report(
            [unheld], self._AS_OF, 1.0, ReportContext(user_portfolio=self._portfolio())
        )

        start = html.index("🎯 Najsilniejsze sygnały")
        section = html[start:html.index("<h2", start + 1)]
        assert "UNIKAJ" in section
        assert "% portfela" not in section


def test_consensus_shift_sell_to_buy_reads_sprzedaj_kup_in_the_mail() -> None:
    # MAIL-61: sekcja „Zmiany nastawienia” nie miesza angielskiego z polskim mailem.
    from src.domain.consensus_shift import ConsensusShift, ShiftKind

    shift = ConsensusShift(
        kind=ShiftKind.FLIP, previous_recommendation="SELL", current_recommendation="BUY",
        votes_changed=3, sentiment_delta=0.0, consensus_delta=0.0, gap_hours=1.0,
    )

    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(consensus_shifts=[("AAPL", shift)]),
    )

    section = html[html.index("🔄 Zmiany nastawienia"):]
    assert "SPRZEDAJ" in section.split("</table>")[0]
    assert "SELL" not in section.split("</table>")[0]
    assert "Odwrócenie SPRZEDAJ → KUP" in text
    assert "Odwrócenie SELL" not in text


def test_watchlist_mood_section_not_called_portfolio() -> None:
    # MAIL-31: sentyment liczy się po symbolach watchlisty; „portfel” to od #15
    # sekcja „💼 Twój portfel” z realnymi pozycjami.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
    )

    assert "Nastroje portfela" not in html
    assert "NASTROJE PORTFELA" not in text
    assert "📊 Nastroje watchlisty" in html
    assert "NASTROJE WATCHLISTY" in text


class TestDegradedDayMood:
    """MAIL-10 (G46): dzień bez danych nie udaje neutralnego nastroju — box
    „Nastroje watchlisty” znika, gdy nic nie zapisano, a bez sentymentów średnia
    to „—”, nie zmyślone „+0,00 (Neutralny)”."""

    _STARTED = datetime(2026, 5, 14, tzinfo=UTC)

    def test_all_ignored_cycle_has_no_fake_neutral_mood(self) -> None:
        html, text = build_html_report(
            [_ignored_result("MSFT"), _ignored_result("AAPL")], self._STARTED, 1.0
        )

        assert "Nastroje watchlisty" not in html
        assert "NASTROJE WATCHLISTY" not in text
        assert "Średni sentyment" not in html
        assert "Średni sentyment" not in text
        assert "0 / 0" not in html

    def test_saved_cycle_without_sentiment_shows_no_average(self) -> None:
        degraded = dataclasses.replace(
            _saved_result(), sentiment_score=None, sentiment_label=None
        )

        html, text = build_html_report([degraded], self._STARTED, 1.0)

        assert "Nastroje watchlisty" in html
        assert "Średni sentyment: <strong>—</strong> (brak danych)" in html
        assert "  Średni sentyment: — (brak danych)" in text
        assert "+0,00 (Neutralny)" not in text

    def test_saved_cycle_with_sentiment_keeps_the_average(self) -> None:
        html, text = build_html_report([_saved_result()], self._STARTED, 1.0)

        assert "Średni sentyment: <strong>+0,42</strong>" in html
        assert "  Średni sentyment: +0,42" in text


def test_footer_lists_data_sources() -> None:
    # MAIL-07: mail nie przypisywał danych Alpha Vantage, Finnhub ani CoinGecko.
    # Lista zależy od flag, więc stopka mówi to wprost („w zależności od konfiguracji”).
    run = RunContext(run_id="77", git_sha="abc1234def")

    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 19, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(run_context=run),
    )

    for body in (html, text):
        assert "Źródła danych (w zależności od konfiguracji):" in body
        for source in ("Alpha Vantage", "Finnhub", "CoinGecko", "SEC EDGAR", "FRED", "NBP"):
            assert source in body
    # Stopka źródeł stoi pod zastrzeżeniem, a stopka przebiegu zostaje ostatnia.
    assert html.index("nie stanowi rekomendacji") < html.index("Źródła danych")
    assert text.index("Źródła danych") < text.index("przebieg #77")


class TestPlainTextSlots:
    """MAIL-09 (G47): wersja tekstowa niesie te same sekcje co HTML tam, gdzie
    istnieje renderer tekstu — klient bez HTML dostawał raport bez kosztu cyklu,
    zmian nastawienia rady, karty modelu i historii głosów."""

    _STARTED = datetime(2026, 5, 14, tzinfo=UTC)

    def _text(self, context: ReportContext) -> str:
        _, text = build_html_report([_saved_result()], self._STARTED, 1.0, context)
        return text

    def test_plain_text_contains_finops_when_cost_given(self) -> None:
        from src.domain.finops import estimate_cycle_cost

        text = self._text(ReportContext(cycle_cost=estimate_cycle_cost({"llm": 3})))

        assert "KOSZT CYKLU" in text
        # Zaplecze, jak w HTML: koszt za separatorem, przed pominiętymi i błędami.
        assert text.index("ZAPLECZE") < text.index("KOSZT CYKLU")

    def test_plain_text_contains_consensus_shifts_in_the_today_zone(self) -> None:
        from src.domain.consensus_shift import ConsensusShift, ShiftKind

        shift = ConsensusShift(
            kind=ShiftKind.FLIP, previous_recommendation="SELL",
            current_recommendation="BUY", votes_changed=2,
            sentiment_delta=0.3, consensus_delta=0.1, gap_hours=24.0,
        )

        text = self._text(ReportContext(consensus_shifts=[("AAPL", shift)]))

        assert "=== 🔄 Zmiany nastawienia ===" in text
        assert text.index("Zmiany nastawienia") < text.index("ZAPLECZE")

    def test_plain_text_contains_model_scorecard_and_council_history(self) -> None:
        from src.application.report_council_history import InvestorHistory

        view = ModelScorecardView(
            summary=summarize_run(["trained_successfully"]),
            trained_at=datetime(2026, 5, 13, tzinfo=UTC),
        )
        history = [
            InvestorHistory(investor_name="Soros", symbol="AAPL",
                            recommendations=("HOLD", "SELL", "SELL")),
        ]

        text = self._text(ReportContext(model_scorecard=view, council_history=history))

        assert "KARTA KONDYCJI MODELU" in text
        assert "=== 🗳️ Rada — historia głosów ===" in text
        assert text.index("ZAPLECZE") < text.index("KARTA KONDYCJI MODELU")

    def test_empty_sections_leave_no_slot_markers(self) -> None:
        text = self._text(ReportContext())

        assert "SLOT" not in text
        assert "KOSZT CYKLU" not in text
        assert "\n\n\n" not in text

class TestAnalystRevisionLead:
    """DAT-13: rewizja konsensusu z sekcji alfy trafia do leadu tym samym tekstem."""

    @staticmethod
    def _alpha() -> dict[str, Any]:
        from src.application.alpha_signals import AlphaSignals
        from src.domain.analyst_consensus import AnalystConsensus

        previous = AnalystConsensus("NVDA", buy=10, hold=10)
        latest = AnalystConsensus("NVDA", strong_buy=3, buy=12, hold=5, previous=previous)
        return {"NVDA": AlphaSignals(symbol="NVDA", analyst=latest)}

    def test_lead_signal_carries_the_alpha_section_label(self) -> None:
        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]

        (signal,) = to_lead_signals(results, [], None, alpha_signals=self._alpha())

        assert signal.analyst_revision_label == (
            "analitycy podnieśli rekomendacje m/m (+3 Strong Buy, +2 Buy, -5 Hold)"
        )

    def test_cycle_lead_shows_the_revision(self) -> None:
        from src.application.report_builder import build_cycle_lead

        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]

        items = build_cycle_lead(results, ReportContext(alpha_signals=self._alpha()))

        assert [item.headline for item in items] == [
            "NVDA +1,2%: analitycy podnieśli rekomendacje m/m (+3 Strong Buy, +2 Buy, -5 Hold)"
        ]


class TestInsiderBuyLead:
    """DAT-21: znaczący zakup insidera z sekcji alfy trafia do leadu i tematu maila
    tym samym tekstem, z progiem z kapitalizacji spółki."""

    @staticmethod
    def _alpha(market_cap: float | None = None) -> dict[str, Any]:
        from src.application.alpha_signals import AlphaSignals
        from src.domain.insider_flow import InsiderBuy, InsiderFlowSnapshot

        insider = InsiderFlowSnapshot(
            "NVDA",
            10_000.0,
            1,
            0,
            90,
            largest_buy=InsiderBuy(role="CEO", value_usd=1_200_000.0),
        )
        return {"NVDA": AlphaSignals(symbol="NVDA", insider=insider, market_cap=market_cap)}

    def test_lead_signal_carries_the_alpha_section_label(self) -> None:
        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]

        (signal,) = to_lead_signals(results, [], None, alpha_signals=self._alpha())

        assert signal.insider_buy_label == "CEO kupił akcje za 1,2 mln USD"

    def test_buy_below_the_cap_scaled_threshold_is_no_lead(self) -> None:
        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]

        (signal,) = to_lead_signals(results, [], None, alpha_signals=self._alpha(market_cap=4e12))

        assert signal.insider_buy_label is None

    def test_cycle_lead_opens_with_the_insider_buy(self) -> None:
        from src.application.report_builder import build_cycle_lead

        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]

        items = build_cycle_lead(results, ReportContext(alpha_signals=self._alpha()))

        assert items[0].headline == "NVDA +1,2%: CEO kupił akcje za 1,2 mln USD"

    @staticmethod
    def _filed_alpha(filed_on: date) -> dict[str, Any]:
        from src.application.alpha_signals import AlphaSignals
        from src.domain.insider_flow import InsiderBuy, InsiderFlowSnapshot

        buy = InsiderBuy(role="CEO", value_usd=1_200_000.0, filed_on=filed_on)
        insider = InsiderFlowSnapshot("NVDA", 10_000.0, 1, 0, 90, largest_buy=buy)
        return {"NVDA": AlphaSignals(symbol="NVDA", insider=insider)}

    def test_buy_filed_before_the_lead_window_is_no_lead(self) -> None:
        # DAT-21: lead tylko przez N dni od filingu; sekcja alfy pokazuje go dalej.
        from src.application.report_builder import build_cycle_lead
        from src.domain.insider_flow import InsiderLeadWindow

        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]
        window = InsiderLeadWindow(as_of=date(2026, 10, 26), max_age_days=3)
        context = ReportContext(
            alpha_signals=self._filed_alpha(date(2026, 10, 22)), insider_lead_window=window
        )

        items = build_cycle_lead(results, context)

        assert all("kupił akcje" not in item.headline for item in items)

    def test_buy_filed_inside_the_lead_window_still_leads(self) -> None:
        from src.application.report_builder import build_cycle_lead
        from src.domain.insider_flow import InsiderLeadWindow

        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.012"))]
        window = InsiderLeadWindow(as_of=date(2026, 10, 26), max_age_days=3)
        context = ReportContext(
            alpha_signals=self._filed_alpha(date(2026, 10, 23)), insider_lead_window=window
        )

        items = build_cycle_lead(results, context)

        assert items[0].headline == "NVDA +1,2%: CEO kupił akcje za 1,2 mln USD"


class TestRegimeChangeLead:
    """ALR-05 (krok 2): zmiana reżimu z nagłówka maila otwiera też lead (i temat)."""

    def test_cycle_lead_opens_with_the_regime_change(self) -> None:
        from src.application.report_builder import build_cycle_lead
        from src.application.report_regime import RegimeBanner
        from src.domain.regime import MarketRegime, RegimeChange

        banner = RegimeBanner(
            regime=MarketRegime.RISK_OFF,
            multiplier=1.5,
            base_threshold=Decimal("0.02"),
            reason="2 głosy ELEVATED",
            change=RegimeChange(previous=MarketRegime.NEUTRAL, current=MarketRegime.RISK_OFF),
        )
        results = [SymbolResult(symbol="NVDA", status="ignored", delta=Decimal("0.01"))]

        items = build_cycle_lead(results, ReportContext(regime=banner))

        assert items[0].headline == "Rynek przeszedł w RISK_OFF, analiza rzadsza"

    def test_banner_without_a_change_adds_no_lead(self) -> None:
        from src.application.report_builder import build_cycle_lead
        from src.application.report_regime import RegimeBanner
        from src.domain.regime import MarketRegime

        banner = RegimeBanner(
            regime=MarketRegime.RISK_OFF,
            multiplier=1.5,
            base_threshold=Decimal("0.02"),
            reason="alert CRITICAL",
        )

        assert build_cycle_lead([], ReportContext(regime=banner)) == []


class TestIgnoredEffectiveThreshold:
    """MAIL-18: ⏸ Pominięte mówi, z jakim progiem porównano ruch, które ruchy były
    blisko progu i o ile reżim albo earnings podniosły próg."""

    _AT = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)

    @staticmethod
    def _ignored(
        symbol: str, delta: str, threshold: str | None, **kwargs: Any
    ) -> SymbolResult:
        return SymbolResult(
            symbol=symbol,
            status="ignored",
            delta=Decimal(delta),
            effective_threshold=None if threshold is None else Decimal(threshold),
            **kwargs,
        )

    def test_ignored_chip_shows_effective_threshold(self) -> None:
        html, text = build_html_report(
            [self._ignored("AAPL", "0.012", "0.02")], self._AT, 1.0
        )

        ignored_html = html.split("⏸ Pominięte")[1]
        assert "AAPL" in ignored_html
        assert "próg 2,0%" in ignored_html
        assert "AAPL(1,20% / próg 2,0%)" in text

    def test_near_threshold_row_lists_moves_from_three_quarters_of_the_threshold(self) -> None:
        html, text = build_html_report(
            [
                self._ignored("AAPL", "0.016", "0.02"),
                self._ignored("MSFT", "0.004", "0.02"),
            ],
            self._AT,
            1.0,
        )

        assert "Blisko progu (≥ 75%)" in html
        near_html = html.split("Blisko progu (≥ 75%)")[1]
        assert "AAPL" in near_html
        assert "MSFT" not in near_html
        near_text = text.split("Blisko progu (≥ 75%)")[1].splitlines()[0]
        assert "AAPL" in near_text
        assert "MSFT" not in near_text

    def test_raised_threshold_shows_multiplier_and_its_cause(self) -> None:
        html, text = build_html_report(
            [
                self._ignored("AAPL", "0.012", "0.03", regime_multiplier=1.5),
                self._ignored("NVDA", "0.012", "0.026", earnings_multiplier=1.3),
            ],
            self._AT,
            1.0,
        )

        assert "próg 3,0% (×1,5 reżim)" in html
        assert "próg 2,6% (×1,3 earnings)" in html
        assert "próg 3,0% (×1,5 reżim)" in text

    def test_ignored_without_threshold_renders_as_before(self) -> None:
        html, text = build_html_report([self._ignored("MSFT", "0.005", None)], self._AT, 1.0)

        assert "próg" not in html.split("⏸ Pominięte")[1]
        assert "Blisko progu" not in html
        assert "MSFT(0,50%)" in text

    def test_mapper_carries_threshold_and_multipliers_of_ignored_state(self) -> None:
        raw = {
            "status": "ignored",
            "delta": Decimal("0.012"),
            "previous_price": Decimal("100"),
            "effective_threshold": Decimal("0.03"),
            "regime_multiplier": 1.5,
            "earnings_multiplier": 1.0,
        }

        result = to_symbol_result("AAPL", raw, now=self._AT)

        assert result.effective_threshold == Decimal("0.03")
        assert result.regime_multiplier == pytest.approx(1.5)
        assert result.earnings_multiplier == pytest.approx(1.0)


def test_predictions_table_has_at_most_five_columns() -> None:
    # MAIL-23: 9 kolumn przy ~360 px telefonu łamało tekst po literze.
    html, _ = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    table = html.split("🔮 Wygenerowane predykcje")[1].split("</table>")[0]
    assert len(re.findall(r"<th\b", table)) <= 5


class TestHeldEarningsLead:
    """DAT-20: lead „wyniki spółki, którą masz” bierze wagi z portfela właściciela
    i zdarzenia z `ReportContext.earnings_soon` (flaga kalendarza wyników). Bez
    kontekstu wyników albo bez portfela lead jest jak wcześniej."""

    _STARTED = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)

    @staticmethod
    def _portfolio() -> Portfolio:
        return Portfolio(
            positions=(
                Position(
                    symbol="MSFT",
                    quantity=Decimal("10"),
                    purchase_price=Decimal("400"),
                    purchase_date=date(2026, 1, 5),
                ),
            ),
            as_of=datetime(2026, 10, 26, tzinfo=UTC),
        )

    @staticmethod
    def _quiet_msft() -> SymbolResult:
        return SymbolResult(
            symbol="MSFT",
            status="ignored",
            delta=Decimal("0.004"),
            current_price=Decimal("410.00"),
        )

    @staticmethod
    def _soon() -> tuple[EarningsEvent, ...]:
        return (
            EarningsEvent(
                "MSFT", 1, "2026-10-28", timing=EarningsTiming.AFTER_CLOSE, eps_estimate=3.1
            ),
        )

    def test_earnings_of_a_held_company_lead_the_mail(self) -> None:
        context = ReportContext(user_portfolio=self._portfolio(), earnings_soon=self._soon())

        html, text = build_html_report([self._quiet_msft()], self._STARTED, 1.0, context)

        assert "MSFT (100% portfela): wyniki jutro po sesji" in html
        assert "MSFT (100% portfela): wyniki jutro po sesji" in text

    def test_without_the_calendar_context_the_lead_is_unchanged(self) -> None:
        context = ReportContext(user_portfolio=self._portfolio())

        html, _ = build_html_report([self._quiet_msft()], self._STARTED, 1.0, context)

        assert "wyniki jutro" not in html

    def test_without_a_portfolio_earnings_do_not_lead(self) -> None:
        context = ReportContext(earnings_soon=self._soon())

        html, _ = build_html_report([self._quiet_msft()], self._STARTED, 1.0, context)

        assert "wyniki jutro" not in html
