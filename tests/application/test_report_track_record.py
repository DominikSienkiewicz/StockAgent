"""Testy renderu sekcji 📈 Track Record (T1 equity curve, T2 calibration,
T4 lessons). Czysta prezentacja — bierze gotowe obiekty domeny/aplikacji i
produkuje fragment HTML. Brak danych → pusty string (sekcja się chowa)."""

from __future__ import annotations

from decimal import Decimal

from src.application.report_lessons import LessonItem, LessonsReport
from src.application.report_track_record import (
    TrackRecordStats,
    render_track_record_html,
)
from src.domain.calibration_curve import CalibrationBucket, probability_scores
from src.domain.equity_curve import EquityCurve, EquityPoint
from src.domain.track_record import (
    HitRateEvidence,
    ScoredCall,
    assess_directional_hit_rate,
)


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
        assert "+7,0%" in html      # total_return
        assert "66,7%" in html or "67%" in html  # win_rate
        assert "2,0%" in html       # max_drawdown


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
        assert "+7,0%" in html
        assert "MSFT" in html
        assert "kalibracj" in html.lower()


def _evidence() -> HitRateEvidence:
    calls = [
        ScoredCall(
            cycle=f"2026-10-{day:02d}",
            predicted_trend="BULLISH",
            is_correct=day % 2 == 0,
            realized_change=0.01 if day % 2 == 0 else -0.01,
        )
        for day in range(1, 9)
        for _ in range(3)
    ]
    return assess_directional_hit_rate(calls)


class TestHonestStats:
    """WIA-17/22/23/31/32/36: Track Record mówi, na jakich założeniach stoi krzywa
    i czy trafność w ogóle odróżnia się od naiwnej strategii."""

    def test_equity_summary_states_portfolio_assumptions_and_method_change(self) -> None:
        html = render_track_record_html(_curve(), None, None)

        assert "równa waga" in html
        assert "gotówka" in html
        # Zmiana metodologii przesuwa historyczne liczby — musi być oznaczona.
        assert "Metodologia zmieniona 2026-10-18" in html

    def test_hit_rate_evidence_line_has_interval_baseline_and_verdict(self) -> None:
        html = render_track_record_html(
            _curve(), None, None, TrackRecordStats(hit_rate=_evidence())
        )

        assert "Trafność kierunkowa: 50,0%" in html
        assert "95% CI" in html
        assert "n=24, n_eff≈8" in html
        assert "zawsze-BULLISH: 50%" in html
        assert "nieodróżnialne od rzutu monetą" in html

    def test_calibration_block_shows_brier_and_skill(self) -> None:
        buckets = [
            CalibrationBucket(lower=0.6, upper=0.7, count=10, mean_confidence=0.6, hit_rate=0.6)
        ]
        scores = probability_scores([(0.6, True)] * 6 + [(0.6, False)] * 4)

        html = render_track_record_html(None, buckets, None, TrackRecordStats(scores=scores))

        assert "Brier = 0,24 " in html
        assert "BSS = +0,0%" in html
        assert "ECE" in html

    def test_stats_alone_do_not_resurrect_a_hidden_section(self) -> None:
        stats = TrackRecordStats(hit_rate=_evidence())

        assert render_track_record_html(None, None, None, stats) == ""


class TestIntervalCoverage:
    """ML-34 — pokrycie zrealizowane vs nominalne przedziałów 80% (kolumny
    z migracji 032, za flagą). Linia mówi wprost, że gwarancja zakłada
    wymienność, której rynek nie spełnia."""

    def test_coverage_line_compares_realised_with_nominal(self) -> None:
        from src.domain.conformal import IntervalCoverage

        html = render_track_record_html(
            None, None, None, TrackRecordStats(coverage=IntervalCoverage(hits=37, n=50))
        )

        assert "Track Record" in html
        assert "Pokrycie przedziałów 80%: zrealizowane 74% (37/50) vs nominalne 80%" in html
        assert "wymienność" in html
        assert "ACI" in html

    def test_no_settled_intervals_no_line(self) -> None:
        from src.domain.conformal import IntervalCoverage

        stats = TrackRecordStats(coverage=IntervalCoverage(hits=0, n=0))

        assert render_track_record_html(None, None, None, stats) == ""
