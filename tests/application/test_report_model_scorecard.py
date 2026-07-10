"""Testy sekcji "Karta kondycji modelu" w raporcie e-mail (roadmap #12).

Sekcja ma zdjąć z niewidzialnej bramki retreningu ("nie shipuj modelu gorszego
od baseline'u") welon: pokazać użytkownikowi, że model faktycznie bije naiwny
baseline persystencji — albo uczciwie przyznać, że kandydat został odrzucony.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.application.report_model_scorecard import (
    build_scorecard_view,
    render_model_scorecard_html,
    render_model_scorecard_text,
)
from src.domain.model_scorecard import METRIC_NOTE, RunSummary, summarize_run

_NOW = datetime(2026, 7, 10, tzinfo=UTC)
_FRESH = _NOW - timedelta(days=3)
_STALE = _NOW - timedelta(days=30)


def _summary(accepted: int, rejected: int, skipped: int) -> RunSummary:
    outcomes = (
        ["trained_successfully"] * accepted
        + ["skipped_validation_failed"] * rejected
        + ["za_malo_probek"] * skipped
    )
    return summarize_run(outcomes)


class TestSelfSuppression:
    def test_none_summary_suppresses_html(self) -> None:
        # Brak danych = brak sekcji, nie pusty nagłówek.
        out = render_model_scorecard_html(
            None, trained_at=_FRESH, now=_NOW
        )
        assert out == ""

    def test_none_summary_suppresses_text(self) -> None:
        out = render_model_scorecard_text(
            None, trained_at=_FRESH, now=_NOW
        )
        assert out == ""


class TestMetricNote:
    def test_metric_note_present_in_html(self) -> None:
        # Sekcja "trust" bez tego zastrzeżenia sama byłaby nieuczciwa.
        out = render_model_scorecard_html(
            _summary(3, 0, 0),
            trained_at=_FRESH,
            now=_NOW,
            candidate_rmse=0.0123,
            baseline_rmse=0.0200,
            directional_hit_rate=0.61,
            folds=5,
        )
        assert METRIC_NOTE in out

    def test_metric_note_present_in_text(self) -> None:
        out = render_model_scorecard_text(
            _summary(3, 0, 0),
            trained_at=_FRESH,
            now=_NOW,
            candidate_rmse=0.0123,
            baseline_rmse=0.0200,
            directional_hit_rate=0.61,
            folds=5,
        )
        assert METRIC_NOTE in out


class TestBrutalHonestyOnRejects:
    def test_all_rejected_renders_and_does_not_fake_success(self) -> None:
        # accepted == 0: sekcja MUSI się wyrenderować i jawnie przyznać odrzut.
        summary = _summary(0, 4, 0)
        html = render_model_scorecard_html(
            summary, trained_at=_FRESH, now=_NOW
        )
        text = render_model_scorecard_text(
            summary, trained_at=_FRESH, now=_NOW
        )
        assert html != ""
        assert text != ""
        # Jawny komunikat: nowy model odrzucony, jedziemy na sprawdzonym.
        assert "odrzucon" in html.lower()
        assert "sprawdzonym" in html.lower()
        assert "odrzucon" in text.lower()
        assert "sprawdzonym" in text.lower()
        # Nie udaje sukcesu — liczba zaakceptowanych to 0.
        assert summary.summary_line in html
        assert "zaakceptowano 0" in html

    def test_rejects_shown_even_when_some_accepted(self) -> None:
        summary = _summary(2, 1, 0)
        html = render_model_scorecard_html(
            summary, trained_at=_FRESH, now=_NOW
        )
        assert "odrzucon" in html.lower()


class TestStaleBadge:
    def test_model_older_than_21_days_shows_stale_badge_html(self) -> None:
        out = render_model_scorecard_html(
            _summary(1, 0, 0), trained_at=_STALE, now=_NOW
        )
        assert "STALE" in out

    def test_model_older_than_21_days_shows_stale_badge_text(self) -> None:
        out = render_model_scorecard_text(
            _summary(1, 0, 0), trained_at=_STALE, now=_NOW
        )
        assert "STALE" in out

    def test_fresh_model_has_no_stale_badge(self) -> None:
        out = render_model_scorecard_html(
            _summary(1, 0, 0), trained_at=_FRESH, now=_NOW
        )
        assert "STALE" not in out


class TestCandidateMetrics:
    def test_shows_improvement_over_baseline(self) -> None:
        out = render_model_scorecard_html(
            _summary(1, 0, 0),
            trained_at=_FRESH,
            now=_NOW,
            candidate_rmse=0.0100,
            baseline_rmse=0.0200,
            directional_hit_rate=0.61,
            folds=5,
        )
        # (0.02 - 0.01) / 0.02 * 100 = 50%
        assert "50.0%" in out
        assert "61%" in out
        assert "5" in out

    def test_negative_improvement_is_not_hidden(self) -> None:
        # Kandydat gorszy od baseline'u — brutalna prawda, ujemny procent.
        out = render_model_scorecard_html(
            _summary(0, 1, 0),
            trained_at=_FRESH,
            now=_NOW,
            candidate_rmse=0.0300,
            baseline_rmse=0.0200,
            directional_hit_rate=0.40,
            folds=5,
        )
        assert "-50.0%" in out

    def test_metrics_optional_render_without_them(self) -> None:
        # Bez metryk kandydata sekcja i tak pokazuje przebieg treningu.
        out = render_model_scorecard_html(
            _summary(2, 0, 0), trained_at=_FRESH, now=_NOW
        )
        assert out != ""
        assert "zaakceptowano 2" in out


class TestHtmlEscaping:
    def test_escapes_metric_note_context(self) -> None:
        # Sekcja nie wstrzykuje surowego HTML — sanity check na braku '<script>'.
        out = render_model_scorecard_html(
            _summary(1, 0, 0), trained_at=_FRESH, now=_NOW
        )
        assert "<script>" not in out


class TestBuildScorecardView:
    """#12 — surowe wiersze z `model_scorecards` → widok sekcji."""

    def test_no_rows_yields_no_view(self) -> None:
        assert build_scorecard_view([]) is None

    def test_aggregates_run_and_takes_latest_metrics(self) -> None:
        rows = [
            {
                "symbol": "AAPL",
                "status": "trained_successfully",
                "timestamp": "2026-07-09T10:00:00+00:00",
                "candidate_holdout_rmse": 0.021,
                "baseline_rmse": 0.024,
                "candidate_holdout_directional_hit_rate": 0.58,
                "n_folds": 7,
            },
            {
                "symbol": "MSFT",
                "status": "skipped_validation_failed",
                "timestamp": "2026-07-09T09:00:00+00:00",
            },
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.summary.total == 2
        assert view.summary.accepted == 1
        assert view.summary.rejected == 1
        assert view.candidate_rmse == 0.021
        assert view.folds == 7

    def test_all_rejected_run_is_not_dressed_up(self) -> None:
        rows = [
            {"symbol": s, "status": "skipped_validation_failed",
             "timestamp": "2026-07-09T10:00:00+00:00"}
            for s in ("AAPL", "MSFT")
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.summary.accepted == 0
        assert view.summary.rejected == 2

    def test_unparsable_timestamp_yields_no_view(self) -> None:
        # Bez daty treningu nie da się orzec o świeżości → sekcja się chowa.
        assert build_scorecard_view([{"symbol": "AAPL", "status": "x"}]) is None
