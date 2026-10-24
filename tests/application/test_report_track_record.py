"""Testy renderu sekcji 📈 Track Record (T1 equity curve, T2 calibration,
T4 lessons). Czysta prezentacja — bierze gotowe obiekty domeny/aplikacji i
produkuje fragment HTML. Brak danych → pusty string (sekcja się chowa)."""

from __future__ import annotations

import re
from decimal import Decimal

from src.application.report_evidence import PaperCosts
from src.application.report_lessons import LessonItem, LessonsReport
from src.application.report_track_record import (
    TrackRecordStats,
    render_track_record_html,
)
from src.domain.benchmark import BenchmarkComparison, compare_with_benchmarks
from src.domain.calibration_curve import CalibrationBucket, probability_scores
from src.domain.equity_curve import EquityCurve, EquityPoint, PaperTrade
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


class TestEntryPriceNote:
    """WIA-34: trafność i paper-PnL liczą lukę na otwarciu, której czytelnik maila
    przed sesją nie kupi — notka pod podsumowaniem krzywej mówi to wprost."""

    def test_equity_summary_says_the_entry_is_the_last_close(self) -> None:
        html = render_track_record_html(_curve(), None, None)

        assert "ostatnie zamknięcie" in html
        assert "lukę na otwarciu" in html

    def test_no_curve_no_entry_note(self) -> None:
        assert "ostatnie zamknięcie" not in render_track_record_html(None, None, None)


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

    def test_renders_the_category_histogram(self) -> None:
        # WIA-57: „3 z 5 błędów: makro” obok listy lekcji.
        from src.domain.mistake_taxonomy import MistakeCategory

        lessons = LessonsReport(
            items=(
                LessonItem(
                    symbol="AAPL",
                    predicted_trend="BULLISH",
                    insight="zignorowałem decyzję Fed",
                    actual_change_pct=None,
                ),
            ),
            total_mistakes=6,
            recurring_theme=None,
            category_counts={MistakeCategory.MACRO: 3, MistakeCategory.EARNINGS: 2},
        )

        html = render_track_record_html(None, None, lessons)

        assert "3 z 5 błędów: makro" in html
        assert "2 z 5: wyniki spółki" in html

    def test_without_categories_there_is_no_histogram(self) -> None:
        lessons = LessonsReport(
            items=(
                LessonItem(
                    symbol="AAPL",
                    predicted_trend="BULLISH",
                    insight="rynek wycenił słabszy popyt",
                    actual_change_pct=None,
                ),
            ),
            total_mistakes=1,
            recurring_theme=None,
        )

        assert "błędów:" not in render_track_record_html(None, None, lessons)

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

    def test_significance_verdict_writes_the_p_value_with_a_comma(self) -> None:
        # MAIL-58: werdykt w mailu po polsku — „p=0,41”, nie „p=0.41”.
        html = render_track_record_html(
            _curve(), None, None, TrackRecordStats(hit_rate=_evidence())
        )

        assert re.search(r"p=\d,\d", html)
        assert not re.search(r"p=\d\.\d", html)

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


class TestPaperCostLabel:
    """WIA-50: krzywa kapitału mówi, z jakim kosztem transakcyjnym ją liczono."""

    _COSTS = PaperCosts(cost_bps=10, crypto_cost_bps=25, crypto_symbols=frozenset({"BTC"}))

    def test_label_and_assumptions_name_the_cost(self) -> None:
        html = render_track_record_html(
            _curve(), None, None, TrackRecordStats(costs=self._COSTS)
        )

        assert "paper-PnL (koszt 10 bps)" in html
        assert "krypto 25 bps" in html
        assert "tylko przy zmianie pozycji" in html
        assert "bez kosztów transakcyjnych" not in html

    def test_mail_uses_the_polish_decimal_comma_for_bps(self) -> None:
        html = render_track_record_html(
            _curve(), None, None, TrackRecordStats(costs=PaperCosts(cost_bps=12.5))
        )

        assert "paper-PnL (koszt 12,5 bps)" in html

    def test_without_stated_costs_the_curve_says_it_is_gross(self) -> None:
        html = render_track_record_html(_curve(), None, None)

        assert "bez kosztów transakcyjnych" in html
        assert "koszt 10 bps" not in html


def _benchmark(*, with_index: bool = True) -> BenchmarkComparison:
    """Agent short na trzech cyklach, koszyk i IVV long w tych samych oknach."""
    changes = (0.01, -0.005, 0.02)
    cycles = [
        [PaperTrade("AAA", "BEARISH", change, benchmark_change=0.008 if with_index else None)]
        for change in changes
    ]
    return compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=0)


class TestBenchmarkLine:
    """WIA-21: krzywa agenta obok koszyka (buy&hold tych samych symboli) i IVV."""

    def test_line_compares_agent_basket_and_ivv(self) -> None:
        comparison = _benchmark()

        html = render_track_record_html(
            comparison.agent, None, None, TrackRecordStats(benchmark=comparison)
        )

        assert "agent -2,5% · koszyk +2,5% · IVV +2,4%" in html
        assert "Zwrot nadwyżkowy vs koszyk: -5,0 p.p." in html
        assert "hit-rate względem rynku (IVV): 33,3% (n=3)" in html
        assert "buy&amp;hold" in html

    def test_missing_ivv_data_is_stated_explicitly(self) -> None:
        comparison = _benchmark(with_index=False)

        html = render_track_record_html(
            comparison.agent, None, None, TrackRecordStats(benchmark=comparison)
        )

        assert "IVV: brak danych benchmarku" in html

    def test_without_a_comparison_the_section_says_there_is_no_benchmark(self) -> None:
        html = render_track_record_html(_curve(), None, None)

        assert "Na tle rynku: brak danych benchmarku" in html

    def test_sparkline_draws_the_basket_as_a_second_row(self) -> None:
        comparison = _benchmark()

        html = render_track_record_html(
            comparison.agent, None, None, TrackRecordStats(benchmark=comparison)
        )

        sparkline = html[html.index("<table role='presentation'") :]
        assert sparkline.count("<tr>") == 2
        assert sparkline.count("background:#9ca3af") == 3

    def test_sparkline_without_a_benchmark_keeps_one_row(self) -> None:
        html = render_track_record_html(_curve(), None, None)

        sparkline = html[html.index("<table role='presentation'") :]
        assert sparkline.count("<tr>") == 1
        assert "#9ca3af" not in sparkline


class TestCompactTrackRecord:
    """Decyzja 2026-10-19 (budżet MAIL-06): w duży dzień Track Record ma wersję
    zwartą — same kluczowe liczby, bez sparklinii, kalibracji i lekcji."""

    @staticmethod
    def _render(*, compact: bool) -> str:
        comparison = _benchmark()
        buckets = [
            CalibrationBucket(lower=0.6, upper=0.7, count=10, mean_confidence=0.6, hit_rate=0.6)
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
        stats = TrackRecordStats(
            hit_rate=_evidence(),
            scores=probability_scores([(0.6, True)] * 6 + [(0.6, False)] * 4),
            costs=PaperCosts(cost_bps=10),
            benchmark=comparison,
        )
        return render_track_record_html(
            comparison.agent, buckets, lessons, stats, compact=compact
        )

    def test_compact_keeps_the_key_numbers(self) -> None:
        html = self._render(compact=True)

        assert "Track Record" in html
        assert "paper-PnL (koszt 10 bps)" in html
        assert "agent -2,5% · koszyk +2,5% · IVV +2,4%" in html
        assert "Trafność kierunkowa: " in html
        assert "95% CI" in html
        assert "n=" in html

    def test_compact_drops_sparklines_calibration_and_lessons(self) -> None:
        html = self._render(compact=True)

        assert "<table" not in html
        assert "Lessons learned" not in html
        assert "zaskoczenie wynikami" not in html
        assert "Krzywa kalibracji" not in html
        assert "Założenia:" not in html

    def test_compact_says_where_the_rest_went(self) -> None:
        assert "wersja zwarta" in self._render(compact=True)

    def test_compact_is_much_smaller_than_the_full_section(self) -> None:
        full = self._render(compact=False)
        compact = self._render(compact=True)

        assert "Lessons learned" in full
        assert len(compact.encode("utf-8")) < len(full.encode("utf-8")) / 2

    def test_compact_without_data_renders_nothing(self) -> None:
        assert render_track_record_html(None, None, None, compact=True) == ""

    def test_compact_without_a_comparison_states_the_missing_benchmark(self) -> None:
        html = render_track_record_html(_curve(), None, None, compact=True)

        assert "Na tle rynku: brak danych benchmarku" in html
        assert "Krzywa kapitału (paper-PnL)" in html


class TestSidewaysCalibration:
    """WIA-39: SIDEWAYS liczone osobno od krzywej kierunkowej — n i Brier."""

    _BUCKETS = [
        CalibrationBucket(lower=0.6, upper=0.7, count=10, mean_confidence=0.65, hit_rate=0.6)
    ]

    def test_sideways_line_shows_n_and_brier(self) -> None:
        sideways = probability_scores([(0.8, True), (0.8, False)])
        stats = TrackRecordStats(sideways_scores=sideways)

        html = render_track_record_html(None, self._BUCKETS, None, stats)

        # Brier = (0.2² + 0.8²) / 2 = 0.34.
        assert "SIDEWAYS" in html
        assert "n=2, Brier 0,340" in html

    def test_no_sideways_scores_no_sideways_line(self) -> None:
        html = render_track_record_html(None, self._BUCKETS, None, TrackRecordStats())

        assert "SIDEWAYS" not in html

def test_expired_predictions_count_is_rendered() -> None:
    """WIA-49: cichy brak oceny wygląda jak brak predykcji — sekcja mówi wprost,
    ile predykcji wygasło bez oceny, nawet gdy reszta Track Record jest pusta.
    Copy idzie polską odmianą liczebnika (3 predykcje wygasły / 5 predykcji wygasło)."""
    html = render_track_record_html(None, None, None, expired_count=3)

    assert "Track Record" in html
    assert "3 predykcje wygasły bez oceny" in html
    assert "5 predykcji wygasło bez oceny" in render_track_record_html(
        None, None, None, expired_count=5
    )
    assert "1 predykcja wygasła bez oceny" in render_track_record_html(
        None, None, None, expired_count=1
    )


def test_no_expired_predictions_no_line() -> None:
    assert render_track_record_html(None, None, None, expired_count=0) == ""
    assert "wygas" not in render_track_record_html(_curve(), None, None)


def test_compact_track_record_keeps_the_expired_count() -> None:
    html = render_track_record_html(_curve(), None, None, compact=True, expired_count=2)

    assert "2 predykcje wygasły bez oceny" in html
    assert "wersja zwarta" in html
