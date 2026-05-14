from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import (
    ResolvedPrediction,
    SymbolResult,
    TopNewsItem,
    build_chart_url,
    build_correlation_chart_url,
    build_forecast_chart_url,
    build_html_report,
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    market_status,
    parse_resolved_predictions,
    to_symbol_result,
)


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


class TestBuildHtmlReport:
    def test_returns_html_and_plain_text(self):
        results = [_saved_result(), _ignored_result(), _error_result()]
        html, text = build_html_report(
            results, datetime(2026, 5, 14, 12, 0, tzinfo=UTC), 12.5
        )
        assert isinstance(html, str) and isinstance(text, str)
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
        assert "Symboli:" in text and "0" in text


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
        assert "Zmiana 12h" in html
        assert "Prognoza" in html
        assert "Sentyment" in html
        assert "Target" not in html  # stara wersja


class TestBuildChartUrl:
    def test_returns_quickchart_url(self):
        results = [
            SymbolResult(symbol="AAPL", status="saved", delta=Decimal("0.025")),
            SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("-0.005")),
        ]
        url = build_chart_url(results)
        assert url is not None
        assert "quickchart.io/chart" in url

    def test_returns_none_for_empty_results(self):
        assert build_chart_url([]) is None

    def test_excludes_error_results(self):
        results = [
            SymbolResult(symbol="X", status="error", error_message="boom"),
        ]
        assert build_chart_url(results) is None

    def test_excludes_results_without_delta(self):
        results = [SymbolResult(symbol="X", status="ignored", delta=None)]
        assert build_chart_url(results) is None

    def test_html_embeds_chart_image(self):
        html, _ = build_html_report(
            [_saved_result(), _ignored_result()],
            datetime(2026, 5, 14, tzinfo=UTC),
            1.0,
        )
        assert "<img" in html
        assert "quickchart.io" in html


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
    def test_returns_quickchart_url_for_saved(self):
        results = [
            SymbolResult(
                symbol="AAPL", status="saved",
                current_price=Decimal("100"), target_price=Decimal("105"),
            )
        ]
        url = build_forecast_chart_url(results)
        assert url is not None and "quickchart.io" in url

    def test_skips_ignored_in_forecast_chart(self):
        results = [
            SymbolResult(symbol="X", status="ignored", delta=Decimal("0.005")),
        ]
        assert build_forecast_chart_url(results) is None

    def test_skips_saved_without_target(self):
        results = [SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))]
        assert build_forecast_chart_url(results) is None

    def test_html_embeds_forecast_chart_when_saved_present(self):
        html, _ = build_html_report(
            [_saved_result()], datetime(2026, 5, 14, tzinfo=UTC), 1.0
        )
        # Powinny być DWA <img> w raporcie: Δ12h + prognoza
        assert html.count("<img") >= 2


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
        assert mood["avg_sentiment"] == 0.0
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
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            accuracy_stats={
                "mean_accuracy": 0.72,
                "sample_count": 50,
                "correct_count": 38,
                "days_window": 30,
            },
        )
        assert "72.0%" in html or "72%" in html
        assert "50" in html
        assert "30 dni" in html

    def test_html_renders_no_data_placeholder(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            accuracy_stats={
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": 30,
            },
        )
        assert "brak ocenionych predykcji" in html.lower()

    def test_no_accuracy_section_when_none(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, accuracy_stats=None,
        )
        assert "Historia trafności" not in html


class TestCorrelationChart:
    def test_returns_url_for_3_plus_points(self):
        results = [
            SymbolResult(symbol=s, status="saved",
                         sentiment_score=v, delta=Decimal(str(v / 2)))
            for s, v in [("A", 0.3), ("B", -0.2), ("C", 0.1)]
        ]
        url = build_correlation_chart_url(results)
        assert url is not None
        assert "scatter" in url.lower() or "quickchart" in url

    def test_returns_none_for_less_than_3(self):
        results = [
            SymbolResult(symbol="A", status="saved",
                         sentiment_score=0.3, delta=Decimal("0.05")),
            SymbolResult(symbol="B", status="saved",
                         sentiment_score=-0.2, delta=Decimal("-0.03")),
        ]
        assert build_correlation_chart_url(results) is None


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
            {"symbol": "NVDA", "predicted_trend": "BULLISH", "accuracy_score": 0.91},
            {"symbol": "SAP",  "predicted_trend": "BEARISH", "accuracy_score": 0.21},
            {"symbol": "Z",    "predicted_trend": "BULLISH", "accuracy_score": None},  # filtered
        ]
        resolved = parse_resolved_predictions(rows)
        assert len(resolved) == 2
        by_sym = {r.symbol: r for r in resolved}
        assert by_sym["NVDA"].is_correct is True
        assert by_sym["SAP"].is_correct is False

    def test_html_renders_resolved_section(self):
        resolved = [
            ResolvedPrediction(symbol="NVDA", predicted_trend="BULLISH",
                               accuracy_score=0.91, is_correct=True),
            ResolvedPrediction(symbol="SAP", predicted_trend="BEARISH",
                               accuracy_score=0.21, is_correct=False),
        ]
        html, text = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0,
            resolved_predictions=resolved,
        )
        assert "Zamknięte predykcje" in html
        assert "NVDA" in html and "SAP" in html
        assert "ZAMKNIĘTE PREDYKCJE" in text

    def test_no_section_when_no_resolved(self):
        html, _ = build_html_report(
            [], datetime(2026, 5, 14, tzinfo=UTC), 1.0, resolved_predictions=[],
        )
        assert "Zamknięte predykcje" not in html


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
        assert result.confidence_score == 0.72
        assert result.reflection_insight == "Ostatni błąd: nie doceniłem hawkish Fed."
        assert len(result.top_news) == 1
        assert result.top_news[0].source == "Reuters"
        assert result.top_news[0].relevance == 0.95

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
