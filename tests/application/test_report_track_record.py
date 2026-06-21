"""Testy renderu sekcji 📈 Track Record (T1 equity curve, T2 calibration,
T4 lessons). Czysta prezentacja — bierze gotowe obiekty domeny/aplikacji i
produkuje fragment HTML. Brak danych → pusty string (sekcja się chowa)."""

from __future__ import annotations

from decimal import Decimal

from src.application.report_lessons import LessonItem, LessonsReport
from src.application.report_track_record import render_track_record_html
from src.domain.calibration_curve import CalibrationBucket
from src.domain.equity_curve import EquityCurve, EquityPoint


def _curve() -> EquityCurve:
    return EquityCurve(
        points=(
            EquityPoint(index=0, period_return=0.05, equity=1.05),
            EquityPoint(index=1, period_return=-0.02, equity=1.029),
            EquityPoint(index=2, period_return=0.04, equity=1.07),
        ),
        total_return=0.07,
        win_rate=0.667,
        max_drawdown=0.02,
        n_trades=3,
    )


class TestEmpty:
    def test_all_none_renders_nothing(self) -> None:
        assert render_track_record_html(None, None, None) == ""

    def test_empty_objects_render_nothing(self) -> None:
        empty_curve = EquityCurve(
            points=(), total_return=0.0, win_rate=0.0, max_drawdown=0.0, n_trades=0
        )
        empty_lessons = LessonsReport(items=(), total_mistakes=0, recurring_theme=None)
        assert render_track_record_html(empty_curve, [], empty_lessons) == ""


class TestEquityCurve:
    def test_renders_summary_stats(self) -> None:
        html = render_track_record_html(_curve(), None, None)
        assert "<h2" in html
        assert "Track Record" in html
        # Wynik łączny, trafność i obsunięcie jako procenty.
        assert "+7.0%" in html      # total_return
        assert "66.7%" in html or "67%" in html  # win_rate
        assert "2.0%" in html       # max_drawdown


class TestCalibration:
    def test_renders_buckets_and_ece(self) -> None:
        buckets = [
            CalibrationBucket(
                lower=0.0, upper=0.1, count=10, mean_confidence=0.05, hit_rate=0.1
            ),
            CalibrationBucket(
                lower=0.9, upper=1.0, count=10, mean_confidence=0.95, hit_rate=0.9
            ),
        ]
        html = render_track_record_html(None, buckets, None)
        assert "<h2" in html
        assert "kalibracj" in html.lower()
        # Hit-rate i pewność per bucket renderowane jako procenty.
        assert "90" in html
        # ECE (well-calibrated → niski) renderowany.
        assert "ECE" in html or "Błąd kalibracji" in html


class TestLessons:
    def test_renders_lessons_and_theme(self) -> None:
        lessons = LessonsReport(
            items=(
                LessonItem(
                    symbol="AAPL",
                    predicted_trend="BULLISH",
                    insight="rynek wycenił słabszy popyt",
                    actual_change_pct=Decimal("-0.08"),
                ),
            ),
            total_mistakes=1,
            recurring_theme="popyt",
        )
        html = render_track_record_html(None, None, lessons)
        assert "<h2" in html
        assert "AAPL" in html
        assert "rynek wycenił słabszy popyt" in html
        assert "popyt" in html

    def test_lesson_insight_html_escaped(self) -> None:
        lessons = LessonsReport(
            items=(
                LessonItem(
                    symbol="<x>",
                    predicted_trend="BULLISH",
                    insight="<script>alert(1)</script>",
                    actual_change_pct=None,
                ),
            ),
            total_mistakes=1,
            recurring_theme=None,
        )
        html = render_track_record_html(None, None, lessons)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestComposition:
    def test_all_three_sections_present(self) -> None:
        buckets = [
            CalibrationBucket(
                lower=0.9, upper=1.0, count=5, mean_confidence=0.95, hit_rate=0.8
            ),
        ]
        lessons = LessonsReport(
            items=(
                LessonItem(
                    symbol="MSFT",
                    predicted_trend="BEARISH",
                    insight="zaskoczenie wynikami",
                    actual_change_pct=Decimal("0.05"),
                ),
            ),
            total_mistakes=1,
            recurring_theme=None,
        )
        html = render_track_record_html(_curve(), buckets, lessons)
        assert "Track Record" in html
        assert "+7.0%" in html
        assert "MSFT" in html
        assert "kalibracj" in html.lower()
