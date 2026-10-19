from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.application import report_builder
from src.application.report_builder import (
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
    to_symbol_result,
)
from src.application.report_formatting import company_label
from src.application.report_model_scorecard import ModelScorecardView
from src.application.run_context import RunContext
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.cycle_maturity import SkipReason
from src.domain.model_scorecard import summarize_evaluations, summarize_run
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.portfolio import Portfolio, Position
from src.domain.quota import QuotaAlert, QuotaSeverity


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

    def test_to_symbol_result_sets_asset_class_from_asset(self):
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

    def test_html_has_dedicated_crypto_section(self):
        html, _ = build_html_report(
            [_saved_result(), _crypto_saved(), _crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        # Wyróżnik sekcji (emoji), niezależny od taga sektora "Krypto" w treści.
        assert "🪙" in html
        assert "BTC" in html

    def test_ignored_crypto_visible_with_price_in_section(self):
        # Ignored krypto (poniżej progu) MUSI pokazać cenę w sekcji krypto —
        # lista 'Pominięte' pokazuje tylko deltę, nie cenę.
        html, _ = build_html_report(
            [_crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        assert "🪙" in html
        assert "$3200.00" in html

    def test_crypto_section_in_plain_text(self):
        _, text = build_html_report(
            [_crypto_saved(), _crypto_ignored()],
            datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0,
        )
        assert "KRYPTO" in text.upper()
        assert "BTC" in text
        assert "ETH" in text

    def test_no_crypto_section_when_no_crypto(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        assert "🪙" not in html

    def test_crypto_section_shows_prediction_reasoning(self):
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
    def test_returns_html_and_plain_text(self):
        results = [_saved_result(), _ignored_result(), _error_result()]
        html, text = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 12.5
        )
        assert isinstance(html, str)
        assert isinstance(text, str)
        assert len(html) > 100
        assert len(text) > 50

    def test_html_contains_summary_counts(self):
        results = [_saved_result(), _ignored_result(), _ignored_result("GOOGL")]
        html, _ = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 1.0
        )
        # 1 saved, 2 ignored, 0 errors
        assert ">1<" in html  # saved count
        assert ">2<" in html  # ignored count
        assert "AAPL" in html

    def test_html_contains_reasoning_for_saved(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "macro tailwinds" in html

    def test_html_shows_sector_in_predictions(self):
        # AAPL → Big Tech, doklejony przy nazwie spółki w sekcji PREDYKCJE.
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Big Tech" in html

    def test_plain_text_shows_sector_in_predictions(self):
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Big Tech" in text

    def test_suggestions_section_appears_for_hot_sector(self):
        # CRWD +6% → gorący sektor cyber → sekcja WARTE UWAGI z peerami.
        hot = SymbolResult(
            symbol="CRWD", status="saved", delta=Decimal("0.06"),
            current_price=Decimal("769"), trend="BULLISH",
            target_price=Decimal("780"), confidence_score=0.75,
            sentiment_score=0.3, sentiment_label="Bullish",
        )
        html, text = build_html_report(
            [hot], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Warte uwagi" in html
        assert "WARTE UWAGI" in text
        # Marker slotu nie może wyciec do treści.
        assert "SUGGESTIONS_SLOT" not in html
        assert "SUGGESTIONS_SLOT" not in text

    def test_no_suggestions_section_when_no_hot_sector(self):
        # Mały ruch, neutralny sentyment → brak sekcji i brak markera.
        cold = SymbolResult(
            symbol="AAPL", status="saved", delta=Decimal("0.004"),
            current_price=Decimal("100"), trend="SIDEWAYS",
            target_price=Decimal("100"), confidence_score=0.6,
            sentiment_score=0.05, sentiment_label="Neutral",
        )
        html, text = build_html_report(
            [cold], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Warte uwagi" not in html
        assert "WARTE UWAGI" not in text
        assert "SUGGESTIONS_SLOT" not in html
        assert "SUGGESTIONS_SLOT" not in text

    def test_html_shows_errors_section_when_present(self):
        html, _ = build_html_report(
            [_error_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Błędy" in html
        assert "429" in html

    def test_plain_text_includes_all_sections(self):
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

    def test_plain_text_formatted_price_and_delta(self):
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "$298.87" in text
        assert "2.50%" in text

    def test_handles_empty_results(self):
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 0.0
        )
        # Brak crasha, nadal generuje strukturę
        assert ">0<" in html  # zero predykcji
        assert "Symboli:" in text
        assert "0" in text


class TestPolishLabels:
    def test_html_uses_polish_trend_label(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # BULLISH → "Wzrostowy"
        assert "Wzrostowy" in html
        assert "BULLISH" not in html

    def test_html_uses_polish_sentiment_label(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # "Somewhat-Bullish" → "Lekko pozytywny"
        assert "Lekko pozytywny" in html
        assert "Somewhat-Bullish" not in html

    def test_plain_text_uses_polish_trend(self):
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Wzrostowy" in text
        assert "BULLISH" not in text

    def test_polish_field_labels_in_table(self):
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

    def test_renders_inline_bars_not_a_remote_image(self):
        results = [
            SymbolResult(symbol="AAPL", status="saved", delta=Decimal("0.025")),
            SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("-0.005")),
        ]
        chart = build_delta_chart_html(results)
        assert chart
        assert "quickchart" not in chart
        assert "AAPL" in chart

    def test_returns_empty_for_empty_results(self):
        assert build_delta_chart_html([]) == ""

    def test_excludes_error_results(self):
        results = [
            SymbolResult(symbol="X", status="error", error_message="boom"),
        ]
        assert build_delta_chart_html(results) == ""

    def test_excludes_results_without_delta(self):
        results = [SymbolResult(symbol="X", status="ignored", delta=None)]
        assert build_delta_chart_html(results) == ""

    def test_report_embeds_the_chart_without_any_remote_host(self):
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

    def test_expected_change_computed_from_current_and_target(self):
        r = _saved_result()
        # (305 - 298.87) / 298.87 ≈ 0.02051
        assert r.expected_change is not None
        assert abs(r.expected_change - Decimal("0.02051")) < Decimal("0.001")

    def test_expected_change_is_none_when_target_missing(self):
        r = SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))
        assert r.expected_change is None

    def test_html_table_shows_signed_forecast_change(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Prognoza pokazana jako "$305.00 (+2.05%)"
        assert "$305.00" in html
        assert "+2.05%" in html or "+2.04%" in html

    def test_html_shows_confidence_percent(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "85%" in html
        assert "Pewność" in html

    def test_plain_text_shows_expected_move_and_confidence(self):
        _, text = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "prognoza" in text.lower()
        assert "pewność 85%" in text.lower()

    def test_reasoning_section_includes_expected_move(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Oczekiwany ruch" in html
        assert "$298.87 → $305.00" in html


class TestForecastChart:
    def test_renders_inline_bars_for_saved(self):
        results = [
            SymbolResult(
                symbol="AAPL", status="saved",
                current_price=Decimal("100"), target_price=Decimal("105"),
            )
        ]
        chart = build_forecast_chart_html(results)
        assert chart
        assert "quickchart" not in chart

    def test_skips_ignored_in_forecast_chart(self):
        results = [
            SymbolResult(symbol="X", status="ignored", delta=Decimal("0.005")),
        ]
        assert build_forecast_chart_html(results) == ""

    def test_skips_saved_without_target(self):
        results = [SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))]
        assert build_forecast_chart_html(results) == ""

    def test_report_contains_both_charts_when_saved_present(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Dwa wykresy: Δ cyklu + prognoza. Oba inline, zero <img>.
        assert "Zmiana ceny (cykl)" in html
        assert "<img" not in html


class TestPortfolioMood:
    def test_avg_sentiment_and_extremes(self):
        results = [
            SymbolResult(symbol="A", status="saved", sentiment_score=0.4, confidence_score=0.8),
            SymbolResult(symbol="B", status="saved", sentiment_score=-0.3),
            SymbolResult(symbol="C", status="ignored", sentiment_score=0.1),
        ]
        mood = build_portfolio_mood(results)
        # avg = (0.4 + -0.3 + 0.1) / 3 ≈ 0.0667
        assert abs(mood["avg_sentiment"] - 0.0667) < 0.001
        assert mood["most_bullish"].symbol == "A"
        assert mood["most_bearish"].symbol == "B"
        assert mood["high_confidence_count"] == 1
        assert mood["saved_count"] == 2

    def test_handles_empty_results(self):
        mood = build_portfolio_mood([])
        assert mood["avg_sentiment"] == pytest.approx(0.0)
        assert mood["most_bullish"] is None
        assert mood["most_bearish"] is None
        assert mood["saved_count"] == 0


class TestMarketStatus:
    def test_open_during_regular_session(self):
        # Czwartek 15:00 UTC = 10:00 ET (zima EST UTC-5)
        # 2026-02-05 to czwartek
        status = market_status(datetime(2026, 2, 5, 15, 0, tzinfo=UTC))
        assert "Otwarta" in status["label"]
        assert "Zamknięcie za" in status["detail"]

    def test_premarket_before_open(self):
        # Czwartek 13:00 UTC = 8:00 ET (przed otwarciem)
        status = market_status(datetime(2026, 2, 5, 13, 0, tzinfo=UTC))
        assert "Premarket" in status["label"]

    def test_after_hours_after_close(self):
        # Czwartek 22:00 UTC = 17:00 ET (po zamknięciu)
        status = market_status(datetime(2026, 2, 5, 22, 0, tzinfo=UTC))
        assert "After" in status["label"]

    def test_weekend_closed(self):
        # Sobota
        status = market_status(datetime(2026, 5, 16, 18, 0, tzinfo=UTC))
        assert "weekend" in status["label"].lower()


class TestReflectionAndTopNews:
    def test_reflection_insight_rendered_in_html(self):
        r = SymbolResult(
            symbol="AAPL",
            status="saved",
            sentiment_score=0.3,
            reflection_insight="Ostatni błąd: zignorowałem makro Fed.",
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Self-Reflection" in html
        assert "zignorowałem makro Fed" in html

    def test_cold_start_reflection_not_shown(self):
        r = SymbolResult(
            symbol="AAPL",
            status="saved",
            sentiment_score=0.3,
            reflection_insight=None,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Brak header'a gdy nikt nie ma reflection
        assert "Self-Reflection" not in html

    def test_top_news_rendered_in_reasoning(self):
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
    def test_html_renders_accuracy_when_provided(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(
                         accuracy_stats={
                    "mean_accuracy": 0.72,
                    "sample_count": 50,
                    "correct_count": 38,
                    "days_window": 30,
                },
                     ))
        assert "72.0%" in html or "72%" in html
        assert "50" in html
        assert "30 dni" in html

    def test_html_renders_no_data_placeholder(self):
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

    def test_no_accuracy_section_when_none(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=None))
        assert "Historia trafności" not in html


class TestCorrelationChart:
    def test_renders_quadrant_table_for_3_plus_points(self):
        results = [
            SymbolResult(symbol=s, status="saved",
                         sentiment_score=v, delta=Decimal(str(v / 2)))
            for s, v in [("A", 0.3), ("B", -0.2), ("C", 0.1)]
        ]
        chart = build_correlation_chart_html(results)
        assert chart
        assert "quickchart" not in chart
        assert "<table" in chart

    def test_returns_empty_for_less_than_3(self):
        results = [
            SymbolResult(symbol="A", status="saved",
                         sentiment_score=0.3, delta=Decimal("0.05")),
            SymbolResult(symbol="B", status="saved",
                         sentiment_score=-0.2, delta=Decimal("-0.03")),
        ]
        assert build_correlation_chart_html(results) == ""


class TestTradeSignals:
    def test_sorts_by_strength_desc(self):
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

    def test_maps_trend_to_polish_direction(self):
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

    def test_skips_results_without_confidence_or_target(self):
        results = [
            SymbolResult(symbol="A", status="saved", trend="BULLISH"),  # no conf/target
            SymbolResult(symbol="B", status="ignored", delta=Decimal("0.01")),
        ]
        assert build_trade_signals(results) == []

    def test_html_renders_strongest_signals_section(self):
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
    def test_detects_divergence_up(self):
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.04"), sentiment_score=-0.4,
        )
        signals = detect_risk_signals([r])
        types = [s.type for s in signals]
        assert "DIVERGENCE" in types

    def test_detects_divergence_down(self):
        r = SymbolResult(
            symbol="Y", status="saved",
            delta=Decimal("-0.04"), sentiment_score=0.4,
        )
        types = [s.type for s in detect_risk_signals([r])]
        assert "DIVERGENCE" in types

    def test_detects_av_llm_conflict(self):
        r = SymbolResult(
            symbol="X", status="saved",
            av_llm_agreement=0.15,
        )
        types = [s.type for s in detect_risk_signals([r])]
        assert "AV_LLM_CONFLICT" in types

    def test_detects_low_signal(self):
        r = SymbolResult(symbol="X", status="saved", news_volume=2)
        types = [s.type for s in detect_risk_signals([r])]
        assert "LOW_SIGNAL" in types

    def test_no_signals_for_healthy_prediction(self):
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.02"), sentiment_score=0.3,
            av_llm_agreement=0.85, news_volume=20,
        )
        assert detect_risk_signals([r]) == []

    def test_html_renders_risk_section(self):
        r = SymbolResult(
            symbol="X", status="saved",
            delta=Decimal("0.05"), sentiment_score=-0.4,
        )
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Sygnały ostrzegawcze" in html or "SYGNAŁY OSTRZEGAWCZE" in text


class TestDayOverDay:
    def test_parses_resolved_predictions(self):
        rows = [
            {"symbol": "NVDA", "predicted_trend": "BULLISH", "is_trend_correct": True},
            {"symbol": "SAP",  "predicted_trend": "BEARISH", "is_trend_correct": False},
            {"symbol": "Z",    "predicted_trend": "BULLISH", "is_trend_correct": None},  # filtered
        ]
        resolved = parse_resolved_predictions(rows)
        assert len(resolved) == 2
        by_sym = {r.symbol: r for r in resolved}
        assert by_sym["NVDA"].is_correct is True
        assert by_sym["SAP"].is_correct is False

    def test_correctness_reflects_trend_direction_not_price_proximity(self):
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

    def test_html_renders_resolved_section(self):
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

    def test_no_section_when_no_resolved(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(resolved_predictions=[]))
        assert "Zamknięte predykcje" not in html


class TestResolvedPostMortem:
    """Sekcja 'Zamknięte predykcje' to pełen post-mortem: dlaczego prognoza
    mówiła wzrost/spadek (oryginalne uzasadnienie), co się faktycznie stało
    (ruch ceny) oraz czy/​dlaczego się potwierdziła (diagnoza dla chybionych).
    Dotyczy akcji I krypto (wszystko leci przez prediction_logs)."""

    def test_parse_enriches_with_reasoning_move_and_insight(self):
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
        assert r.actual_change_pct == Decimal("0.06")

    def test_miss_shows_reasoning_actual_move_and_diagnosis(self):
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
        assert "+4.00%" in html
        # (3) dlaczego się NIE potwierdziła — diagnoza
        assert "Zignorowałem odbicie całego sektora chmury" in html
        # plain text też niesie pełen post-mortem
        assert "Słabe wyniki kwartalne" in text
        assert "Zignorowałem odbicie całego sektora chmury" in text

    def test_miss_with_placeholder_insight_shows_neutral_why(self):
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

    def test_miss_diagnosis_mentioning_trafiona_is_still_shown(self):
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

    def test_hit_shows_reasoning_and_confirmation(self):
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
        assert "+6.00%" in html
        assert "Trafiona" in html

    def test_crypto_closed_prediction_is_covered_and_tagged(self):
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
            ReportContext(resolved_predictions=resolved),
        )
        assert "BTC" in html
        assert "Krypto" in html  # tag klasy aktywa (company_label_with_sector)
        assert "Napływy do ETF-ów spot" in html

    def test_backward_compatible_without_new_fields(self):
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
    def test_top_news_with_url_renders_as_anchor(self):
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

    def test_news_without_url_falls_back_to_plain_text(self):
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
    def test_escapes_llm_news_and_error_text(self):
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

    def test_rejects_unsafe_news_url(self):
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

    def test_escapes_unknown_risk_signal_type_fallback(self, monkeypatch):
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
            report_builder, "detect_risk_signals", lambda results: [injected]
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

    def test_news_title_escaped(self):
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r", sentiment_score=0.3,
            current_price=Decimal("100"), target_price=Decimal("105"),
            top_news=[TopNewsItem(title=_XSS_SCRIPT, source="Reuters",
                                  url="https://example.com/1",
                                  relevance=0.9, sentiment=0.2)],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_news_source_escaped(self):
        r = SymbolResult(
            symbol="X", status="saved", reasoning="r", sentiment_score=0.3,
            current_price=Decimal("100"), target_price=Decimal("105"),
            top_news=[TopNewsItem(title="ok", source=_XSS_IMG,
                                  url="https://example.com/1",
                                  relevance=0.9, sentiment=0.2)],
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_llm_reasoning_escaped(self):
        r = SymbolResult(
            symbol="X", status="saved", reasoning=_XSS_SCRIPT,
            sentiment_score=0.3, current_price=Decimal("100"),
            target_price=Decimal("105"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_reflection_insight_escaped(self):
        r = SymbolResult(
            symbol="X", status="saved", sentiment_score=0.3,
            reflection_insight=_XSS_IMG,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_error_message_escaped(self):
        r = SymbolResult(
            symbol="X", status="error", error_message=_XSS_SCRIPT,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_SCRIPT, _XSS_SCRIPT_MARKER)

    def test_resolved_prediction_reasoning_and_insight_escaped(self):
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

    def test_unknown_symbol_in_ignored_chip_escaped(self):
        # Symbol w chipie 'Pominięte' renderowany przez _html(r.symbol).
        r = SymbolResult(
            symbol=_XSS_IMG, status="ignored", delta=Decimal("0.001"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        _assert_escaped(html, _XSS_IMG, _XSS_IMG_MARKER)

    def test_rag_precedent_summary_escaped(self):
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

    def test_provenance_badge_detail_escaped(self):
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
    def test_maps_saved_status(self):
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

    def test_cold_start_reflection_filtered(self):
        raw = {
            "status": "saved",
            "reflection_context": "Brak danych historycznych do oceny.",
            "llm_analysis": {},
            "sentiment": {},
        }
        result = to_symbol_result("AAPL", raw)
        assert result.reflection_insight is None

    def test_maps_ignored_status(self):
        raw = {"status": "ignored", "delta": Decimal("0.005")}
        result = to_symbol_result("MSFT", raw)
        assert result.status == "ignored"
        assert result.delta == Decimal("0.005")

    def test_maps_error_from_exception(self):
        result = to_symbol_result("AAPL", None, error="HTTPError 500")
        assert result.status == "error"
        assert result.error_message == "HTTPError 500"

    def test_handles_missing_optional_fields(self):
        # Defensive: niekompletny raw dict (np. agent crashował w połowie)
        raw = {"status": "saved"}
        result = to_symbol_result("X", raw)
        assert result.status == "saved"
        assert result.delta is None
        assert result.trend is None


class TestRagPrecedentReceipts:
    """Q5 — analogi RAG z cyklu wystawione jako audytowalny blok."""

    def _raw_with_precedents(self, precedents):
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

    def test_to_symbol_result_maps_precedents(self):
        raw = self._raw_with_precedents([
            {"summary": "Fed hawkish surprise", "predicted_trend": "BEARISH",
             "is_trend_correct": True},
        ])
        result = to_symbol_result("AAPL", raw)
        assert len(result.similar_precedents) == 1
        assert result.similar_precedents[0].summary == "Fed hawkish surprise"
        assert result.similar_precedents[0].predicted_trend == "BEARISH"
        assert result.similar_precedents[0].is_trend_correct is True

    def test_missing_precedents_yields_empty_list(self):
        result = to_symbol_result("AAPL", {"status": "saved", "llm_analysis": {},
                                           "sentiment": {}})
        assert result.similar_precedents == []

    def test_html_renders_analog_block_with_outcomes(self):
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

    def test_no_analog_block_when_empty(self):
        raw = self._raw_with_precedents([])
        result = to_symbol_result("AAPL", raw)
        html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Na podstawie analogów" not in html

    def test_plain_text_shows_analog_summary(self):
        raw = self._raw_with_precedents([
            {"summary": "Fed hawkish surprise spooks tech",
             "predicted_trend": "BEARISH", "is_trend_correct": True},
        ])
        result = to_symbol_result("AAPL", raw)
        _, text = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Fed hawkish surprise spooks tech" in text
        assert "analog" in text

    def test_analog_summary_escaped(self):
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

    def test_to_symbol_result_builds_degraded_badge(self):
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

    def test_clean_inputs_yield_fresh_only(self):
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

    def test_html_renders_degraded_chip(self):
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

    def test_fresh_chip_not_rendered_no_noise(self):
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

    def test_plain_text_shows_provenance_label(self):
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


class TestCompanyLabel:
    def test_known_symbol_returns_name(self):
        assert company_label("AAPL") == "AAPL (Apple Inc.)"

    def test_known_symbol_msft(self):
        assert company_label("MSFT") == "MSFT (Microsoft Corporation)"

    def test_known_symbol_nvda(self):
        assert company_label("NVDA") == "NVDA (NVIDIA Corporation)"

    def test_unknown_symbol_returns_symbol_only(self):
        assert company_label("ZZZZ") == "ZZZZ"

    def test_empty_symbol_returns_empty(self):
        assert company_label("") == ""


class TestCompanyNamesInReport:
    def test_html_shows_company_name_in_predictions_table(self):
        r = SymbolResult(
            symbol="AAPL", status="saved",
            delta=Decimal("0.025"), current_price=Decimal("298.87"),
            trend="BULLISH", target_price=Decimal("305.00"),
            confidence_score=0.85, reasoning="strong macro",
            sentiment_score=0.42,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in html

    def test_html_shows_company_name_in_reasoning_section(self):
        r = SymbolResult(
            symbol="MSFT", status="saved",
            trend="BULLISH", confidence_score=0.7,
            current_price=Decimal("400"), target_price=Decimal("420"),
            reasoning="cloud growth", sentiment_score=0.3,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Microsoft Corporation" in html

    def test_html_shows_company_name_in_trade_signals(self):
        r = SymbolResult(
            symbol="NVDA", status="saved", trend="BULLISH",
            confidence_score=0.9, current_price=Decimal("900"),
            target_price=Decimal("950"),
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "NVIDIA Corporation" in html

    def test_html_does_not_show_company_name_in_ignored_section(self):
        r = SymbolResult(symbol="AAPL", status="ignored", delta=Decimal("0.001"))
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Sekcja "Pominięte" NIE powinna zawierać nazwy spółki
        ignored_section = html.split("Pominięte")[1] if "Pominięte" in html else ""
        assert "Apple Inc." not in ignored_section

    def test_html_shows_company_name_in_errors_section(self):
        r = SymbolResult(symbol="AAPL", status="error", error_message="timeout")
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in html

    def test_plain_text_shows_company_name_in_predictions(self):
        r = SymbolResult(
            symbol="AAPL", status="saved",
            delta=Decimal("0.025"), current_price=Decimal("298.87"),
            trend="BULLISH", target_price=Decimal("305.00"),
            confidence_score=0.85, reasoning="strong macro",
            sentiment_score=0.42,
        )
        _, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "Apple Inc." in text

    def test_unknown_symbol_shows_only_ticker(self):
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
    def test_html_shows_kup_for_bullish(self):
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

    def test_html_shows_sprzedaj_for_bearish(self):
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

    def test_html_shows_wstrzymaj_for_sideways(self):
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
        assert "WSTRZYMAJ" in after_header

    def test_html_recommendation_includes_why_reason(self):
        r = SymbolResult(
            symbol="AAPL", status="saved", trend="BULLISH",
            confidence_score=0.8, current_price=Decimal("100"),
            target_price=Decimal("106"), reasoning="positive outlook",
            sentiment_score=0.4,
        )
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Powinien być powód (pewność / prognoza zmiana)
        assert "80%" in html  # pewność
        assert "+6.00%" in html  # prognozowana zmiana

    def test_plain_text_includes_recommendation(self):
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

    def _make_macro_report(self):
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

    def test_html_contains_risk_watch_section_when_report_provided(self):
        r = _saved_result()
        html, _ = build_html_report(
            [r], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            ReportContext(macro_risk_report=self._make_macro_report()),
        )
        assert "Risk Watch" in html
        assert "SH" in html
        # Slot placeholder musi być wymieniony, nie zostać surowo w outpucie.
        assert "RISK_WATCH_SLOT" not in html

    def test_html_skips_section_when_no_macro_report(self):
        r = _saved_result()
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        # Surowy komentarz HTML jest dopuszczalny — placeholder pozostaje
        # w drzewie i jest ignorowany przez klientów mailowych.
        assert "Risk Watch" not in html

    def test_plain_text_contains_risk_watch_block(self):
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

    def test_html_contains_leaderboard_and_fills_slot(self):
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

    def test_html_suppresses_section_when_no_track_record(self):
        r = _saved_result()
        html, _ = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "ranking wiarygodności" not in html
        # Slot zawsze wymieniany (na ""), żeby marker nie wyciekł do maila.
        assert "PERSONA_LEADERBOARD_SLOT" not in html

    def test_html_suppresses_section_for_empty_track_record(self):
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

    def test_lead_suppressed_when_cycle_has_no_signal(self):
        r = _ignored_result("MSFT")
        html, text = build_html_report([r], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

        assert self._HEADING not in html
        assert "LEAD_SLOT" not in html
        assert "LEAD_SLOT" not in text

    def test_critical_quota_alert_produces_a_lead(self):
        html, _ = build_html_report(
            [_ignored_result("MSFT")],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        assert self._HEADING in html
        assert "Alpha Vantage" in html
        assert "LEAD_SLOT" not in html

    def test_plain_text_carries_the_lead(self):
        _, text = build_html_report(
            [_ignored_result("MSFT")],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        # Wariant plain-text niesie nagłówek sekcji wersalikami.
        assert self._HEADING.upper() in text
        assert "Alpha Vantage" in text
        assert "LEAD_SLOT" not in text

    def test_lead_appears_before_the_body_of_the_report(self):
        # Lead to PIERWSZY ekran — musi wyprzedzać sekcje szczegółowe.
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(quota_alerts=[self._critical_alert()]))

        assert html.index(self._HEADING) < html.index("AAPL")

    def test_split_council_lead_renders_fractional_delta_as_percent(self):
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

        assert "NVDA -4.2%, rada PODZIELONA" in html
        assert "NVDA -4.2%, rada PODZIELONA" in text


class TestSkipReasonMapping:
    """#4 — `skip_reason` rozdziela przyczyny pominięcia. Bez tego powitalny
    ton „Dzień 1" przykryłby masową awarię źródeł."""

    def test_cold_start_is_detected_from_zero_previous_price(self):
        raw = {"status": "ignored", "delta": Decimal("0"), "previous_price": Decimal("0")}

        result = to_symbol_result("AAPL", raw)

        assert result.skip_reason is SkipReason.COLD_START

    def test_below_threshold_when_previous_price_exists(self):
        raw = {
            "status": "ignored",
            "delta": Decimal("0.001"),
            "previous_price": Decimal("298.87"),
        }

        result = to_symbol_result("AAPL", raw)

        assert result.skip_reason is SkipReason.BELOW_THRESHOLD

    def test_saved_symbol_has_no_skip_reason(self):
        raw = {"status": "saved", "delta": Decimal("0.05"), "previous_price": Decimal("100")}

        assert to_symbol_result("AAPL", raw).skip_reason is None

    def test_error_symbol_has_no_skip_reason(self):
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

    def test_section_suppressed_without_a_portfolio(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )

        assert "USER_PORTFOLIO_SLOT" not in html

    def test_fresh_portfolio_renders_real_pnl(self):
        html, _ = build_html_report(
            [_saved_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0, ReportContext(user_portfolio=self._portfolio(days_old=0)))

        assert "AAPL" in html
        assert "USER_PORTFOLIO_SLOT" not in html

    def test_stale_portfolio_is_badged(self):
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

    def test_mapper_carries_cold_start_flag(self):
        raw = {
            "status": "saved",
            "current_price": Decimal("100"),
            "ml_target_price": Decimal("100"),
            "ml_cold_start": True,
        }
        assert to_symbol_result("AAPL", raw).ml_cold_start is True

    def test_mapper_without_flag_is_not_cold_start(self):
        raw = {"status": "saved", "current_price": Decimal("100")}
        assert to_symbol_result("AAPL", raw).ml_cold_start is False

    def test_cold_start_forecast_is_labelled_instead_of_zero_percent(self):
        html, text = build_html_report(
            [_cold_start_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "— (ML: cold-start)" in html
        assert "Oczekiwany ruch: — (ML: cold-start)" in html
        assert "Prognozowana zmiana" not in html
        # "(+0.00%)" to dotychczasowy format kolumny prognozy i linii ruchu.
        assert "(+0.00%)" not in html
        assert "prognoza — (ML: cold-start)" in text
        assert "(+0.00%)" not in text

    def test_trained_forecast_has_no_cold_start_label(self):
        html, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)
        assert "ML: cold-start" not in html
        assert "ML: cold-start" not in text

    def test_scorecard_admits_cold_start_when_no_model_is_loaded(self):
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

    def test_scorecard_names_the_candidate_behind_the_metrics(self):
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

    def test_scorecard_shows_per_symbol_evaluations_next_to_the_pooled_verdict(self):
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

    def test_signal_section_names_its_ranking_source_on_cold_start(self):
        html, text = build_html_report(
            [_cold_start_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        assert "Najsilniejsze sygnały" in html
        assert "ranking wg pewności" in html
        assert "+0.00%" not in html
        assert "ranking wg pewności" in text
        assert "+0.00%" not in text

    def test_forecast_chart_skips_cold_start_results(self):
        assert build_forecast_chart_html([_cold_start_result()]) == ""


class TestAccuracyHistoryInterval:
    """WIA-32: historia trafności w mailu pokazuje przedział Wilsona obok procentu."""

    _STATS = {
        "mean_accuracy": 0.72,
        "sample_count": 50,
        "correct_count": 36,
        "days_window": 30,
    }

    def test_html_shows_wilson_interval(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=self._STATS)
        )

        # 36/50 → Wilson 95% ≈ 58–83%.
        assert "95% CI 58–83%" in html

    def test_plain_text_shows_wilson_interval(self):
        _, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, ReportContext(accuracy_stats=self._STATS)
        )

        assert "95% CI 58–83%" in text


class TestResolvedPredictionTimestamp:
    """WIA-17/23: klucz cyklu (dzień UTC) wymaga znacznika czasu predykcji."""

    def test_parse_keeps_the_prediction_timestamp(self):
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

    def test_unparseable_timestamp_is_none(self):
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


def test_trade_signal_shows_calibrated_confidence():
    """WIA-12: ranking liczy się na pewności skalibrowanej, a mail pokazywał
    tylko surową — `TradeSignal` obiecywał format „85% → 58%”."""
    html = report_builder._render_trade_signals_html([
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

    def test_html_shows_raw_to_calibrated_with_bucket_n(self):
        html = report_builder._render_trade_signals_html(
            [_signal(calibration_bucket_count=12)]
        )

        assert "85%" in html
        assert "→" in html
        assert "58%" in html
        assert "po kalibracji" in html
        assert "n kubełka: 12" in html

    def test_html_replaces_bare_strength_with_rank_and_components(self):
        html = report_builder._render_trade_signals_html([
            _signal(symbol="AAPL"),
            _signal(symbol="MSFT", calibrated_confidence=None, strength=0.85 * 3.1),
        ])

        assert "siła sygnału" not in html
        assert "#1" in html
        assert "#2" in html
        assert "ruch 3.1% × pewność skalibrowana 58%" in html
        assert "ruch 3.1% × pewność 85%" in html

    def test_html_uncalibrated_signal_shows_raw_confidence_only(self):
        html = report_builder._render_trade_signals_html(
            [_signal(calibrated_confidence=None)]
        )

        assert "pewność <strong>85%</strong>" in html
        assert "po kalibracji" not in html

    def test_html_cold_start_strength_is_the_confidence_alone(self):
        html = report_builder._render_trade_signals_html([
            _signal(
                expected_change=Decimal("0"),
                target_price=Decimal("100"),
                strength=0.58,
                ml_cold_start=True,
            )
        ])

        assert "wg pewności skalibrowanej 58%" in html
        assert "ruch 0.0%" not in html

    def test_text_shows_calibration_and_rank_components(self):
        lines = report_builder._render_trade_signals_text(
            [_signal(calibration_bucket_count=12)]
        )
        text = "\n".join(lines)

        assert "pewność 85% → 58% po kalibracji (n kubełka: 12)" in text
        assert "#1 · ruch 3.1% × pewność skalibrowana 58%" in text
        assert "siła " not in text


class TestAccuracyHistoryBothDefinitions:
    """WIA-29 (c): „Historia trafności” liczyła SIDEWAYS do trafności, a Track
    Record pokazuje trafność kierunkową — dwie różne liczby pod jedną nazwą.
    Mail pokazuje obie obok siebie, z etykietami mówiącymi, co liczą."""

    _STATS = {
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

    def test_html_shows_directional_next_to_total(self):
        html = self._html(self._STATS)

        assert "Trafność kierunkowa (bez prognoz „bez ruchu”)" in html
        assert "65.0%" in html
        assert "n=40" in html
        assert "Trafność łączna (z prognozami „bez ruchu”)" in html
        assert "72.0%" in html
        assert "95% CI 58–83%, n=50" in html
        assert "Średnia trafność" not in html

    def test_text_shows_directional_next_to_total(self):
        text = self._text(self._STATS)

        assert "Trafność kierunkowa (bez prognoz „bez ruchu”): 65.0%" in text
        assert "Trafność łączna (z prognozami „bez ruchu”): 72.0%" in text
        assert "Średnia trafność" not in text

    def test_no_directional_calls_is_said_plainly(self):
        stats = {**self._STATS, "directional_count": 0, "directional_correct_count": 0}

        assert "brak prognoz kierunkowych" in self._html(stats)
        assert "brak prognoz kierunkowych" in self._text(stats)

    def test_stats_without_directional_counts_show_the_labelled_total_only(self):
        stats = {
            key: value
            for key, value in self._STATS.items()
            if not key.startswith("directional")
        }

        html = self._html(stats)

        assert "Trafność łączna (z prognozami „bez ruchu”)" in html
        assert "Trafność kierunkowa" not in html


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

    def test_summary_shows_directional_next_to_total(self):
        html, text = self._report([
            self._resolved("NVDA", "BULLISH", True),
            self._resolved("AMD", "BEARISH", False),
            self._resolved("SPY", "SIDEWAYS", True),
        ])

        for body in (html, text):
            assert "Trafność kierunkowa (bez prognoz „bez ruchu”): 50% (n=2)" in body
            assert "Trafność łączna (z prognozami „bez ruchu”): 67% (n=3)" in body
            assert "% accuracy" not in body

    def test_summary_without_directional_calls_says_so(self):
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


def test_full_band_not_suggested_on_tiny_accuracy_sample():
    # PORT-01 (G31): jedno trafienie (100%, n_eff = 1) nie jest track recordem.
    html, text = _sizing_report({
        "mean_accuracy": 1.0, "sample_count": 1, "correct_count": 1,
        "days_window": 30, "n_eff": 1.0,
    })

    assert "Pozycja pełna" not in html
    assert "Pozycja pełna" not in text
    assert "Pozycja standardowa" in html


def test_full_band_not_suggested_when_accuracy_stats_have_no_effective_sample():
    # Bez n_eff nie wiadomo, ile niezależnych zakładów stoi za hit-rate —
    # 40 trafień jednego dnia spadku NASDAQ to jeden zakład.
    html, _ = _sizing_report({
        "mean_accuracy": 0.9, "sample_count": 40, "correct_count": 36,
        "days_window": 30,
    })

    assert "Pozycja pełna" not in html


def test_full_band_suggested_on_a_sufficient_effective_sample():
    html, _ = _sizing_report({
        "mean_accuracy": 0.7, "sample_count": 200, "correct_count": 140,
        "days_window": 30, "n_eff": 25.0,
    })

    assert "Pozycja pełna 4-5% portfela" in html


def test_plain_text_report_says_var_lacks_history():
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


def test_user_portfolio_lists_a_position_the_cycle_could_not_price():
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

    assert "Bez wyceny w tym cyklu: XDWD.DE (≈ 44.4% kosztu)" in html


def test_held_price_outside_the_watchlist_prices_the_position():
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
    assert "$820.00" in html


def test_lead_warns_about_a_council_signal_against_a_large_position():
    # PORT-10: rada SPRZEDAJ na pozycji ważącej ≥ 10% wyprzedza silny konsensus
    # na spółce, której użytkownik nie ma.
    def verdict(rec: str) -> CouncilVerdict:
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

    headline = "Sygnał przeciw Twojej pozycji: rada SPRZEDAJ na NVDA (25% portfela) -4.2%"
    assert headline in html
    assert html.index(headline) < html.index("AMD +8.0%, rada zgodna")
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


def test_plain_text_email_carries_the_user_portfolio_section():
    # Decyzja 2026-10-19: wersja tekstowa maila ma sekcję portfela jak HTML.
    _, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(user_portfolio=_held_aapl_portfolio()),
    )

    assert "=== 💼 Twój portfel ===" in text
    assert "$2,988.70" in text
    assert "USER_PORTFOLIO_SLOT" not in text


def test_plain_text_without_user_portfolio_keeps_the_html_section():
    # Stary push (pełny tekst do Telegram/Slack) dostaje tekst BEZ kwot portfela.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
        ReportContext(user_portfolio=_held_aapl_portfolio(), user_portfolio_in_text=False),
    )

    assert "Twój portfel" not in text
    assert "$2,988.70" not in text
    assert "USER_PORTFOLIO_SLOT" not in text
    assert "Twój portfel" in html


@pytest.mark.parametrize(
    "context",
    [None, ReportContext(macro_risk_report=MacroRiskReport())],
    ids=["no-risk-watch-report", "empty-risk-watch-report"],
)
def test_plain_text_without_risk_watch_data_has_no_slot_marker(context):
    # Pusty slot = brak sekcji. Komentarz HTML jest w mailu HTML niewidoczny,
    # ale w wersji tekstowej był dosłowną linią „<!-- RISK_WATCH_SLOT -->”.
    _, text = build_html_report(
        [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0, context
    )

    assert "RISK_WATCH_SLOT" not in text
    assert "<!--" not in text


def test_plain_text_without_quota_alerts_has_no_slot_marker():
    # Ten sam mechanizm co Risk Watch: bez alertów kwot tekst zaczynał się od
    # dosłownego „<!-- QUOTA_BANNER_SLOT -->”.
    _, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    assert "QUOTA_BANNER_SLOT" not in text
    assert text.startswith("=" * 64)


def test_plain_text_with_quota_alert_still_starts_with_the_banner():
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


def test_header_shows_the_cycle_start_in_the_display_timezone():
    # MAIL-14: nagłówek HTML i tekstu w czasie lokalnym, nie „… UTC”.
    html, text = build_html_report(
        [_saved_result()], datetime(2026, 10, 16, 5, 30, tzinfo=UTC), 1.0,
        ReportContext(display_timezone="Europe/Warsaw"),
    )

    assert "Czas rozpoczęcia: pt 16.10 07:30" in html
    assert "Czas rozpoczęcia:  pt 16.10 07:30" in text
    assert "05:30" not in html
    assert "05:30" not in text


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
    return html[: html.index("</div>") + len("</div>")]


def test_html_starts_with_hidden_preheader_from_lead():
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


def test_preheader_escapes_the_lead_detail():
    # Detail pochodzi z rady i newsów — w preheaderze przechodzi przez html.escape.
    result = SymbolResult(
        symbol="AMD", status="saved", delta=Decimal("0.08"), current_price=Decimal("50"),
        council_verdict=_consensus("<img src=x onerror=alert(1)> & marże"),
    )

    html, _ = build_html_report([result], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    first = _first_element(html)
    assert "&lt;img src=x onerror=alert(1)&gt; &amp; marże" in first
    assert "<img" not in html


def test_preheader_without_lead_counts_closed_predictions():
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


def test_plain_text_has_no_preheader():
    _, text = build_html_report([_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0)

    assert "&zwnj;" not in text
    assert "Brak sygnałów ponad próg" not in text


class TestResolvedSectionWindowHeading:
    """WIA-19: za flagą `settled_at_windows_enabled` sekcja pokazuje predykcje
    rozliczone w TYM cyklu (po `settled_at`), więc nagłówek nie może dalej
    obiecywać „ostatnich 24h”, których okno już nie liczy."""

    _RESOLVED = [ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH", is_correct=True)]

    def test_cycle_window_heading(self):
        html, text = build_html_report(
            [], datetime(2026, 10, 19, 5, 30, tzinfo=UTC), 1.0,
            ReportContext(resolved_predictions=self._RESOLVED, resolved_this_cycle=True),
        )

        assert "Zamknięte predykcje (rozliczone w tym cyklu)" in html
        assert "ZAMKNIĘTE PREDYKCJE (rozliczone w tym cyklu)" in text
        assert "ostatnie 24h" not in html
        assert "ostatnie 24h" not in text

    def test_default_heading_keeps_the_24h_window(self):
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

    def test_footer_links_run_and_commit(self):
        html, _ = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=self._GHA)
        )

        assert 'href="https://github.com/x/y/actions/runs/77"' in html
        assert "przebieg #77" in html
        assert "commit abc1234" in html
        # Krótki SHA, nie pełny.
        assert "abc1234def" not in html

    def test_plain_text_footer_carries_run_commit_and_logs_url(self):
        _, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=self._GHA)
        )

        assert text.rstrip().endswith(
            "przebieg #77 · commit abc1234 · logi: https://github.com/x/y/actions/runs/77"
        )

    def test_no_run_context_means_no_link_and_unchanged_text(self):
        html, text = build_html_report([], self._STARTED, 1.0, ReportContext())

        assert "actions/runs" not in html
        assert "przebieg #" not in html
        assert "przebieg #" not in text
        assert "raport automatyczny" in html

    def test_local_run_has_no_logs_link(self):
        html, text = build_html_report(
            [], self._STARTED, 1.0, ReportContext(run_context=RunContext())
        )

        assert "commit local" in html
        assert ">logi</a>" not in html
        assert "commit local" in text
        assert "logi:" not in text

    def test_footer_escapes_values_and_drops_unsafe_url(self):
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
