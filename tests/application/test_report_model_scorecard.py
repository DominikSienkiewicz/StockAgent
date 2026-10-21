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
        assert "50,0%" in out
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
        assert "-50,0%" in out

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

    def test_counts_only_the_latest_training_run(self) -> None:
        # ML-03: okno 30 dni niesie kilka tygodniowych przebiegów; karta liczyła
        # je razem („zaakceptowano 2 z 4"), choć ostatni trening miał 2 wiersze.
        rows = [
            {"symbol": "AAPL", "status": "trained_successfully",
             "timestamp": "2026-10-11T03:40:00+00:00"},
            {"symbol": "MSFT", "status": "skipped_validation_failed",
             "timestamp": "2026-10-11T03:35:00+00:00"},
            {"symbol": "AAPL", "status": "trained_successfully",
             "timestamp": "2026-10-04T03:40:00+00:00"},
            {"symbol": "MSFT", "status": "skipped_validation_failed",
             "timestamp": "2026-10-04T03:35:00+00:00"},
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.summary.total == 2
        assert view.summary.accepted == 1

    def test_rerun_on_the_same_day_is_its_own_run_by_run_id(self) -> None:
        # ML-24: re-run 10 minut po cronie sklejał się z nim („zaakceptowano 1,
        # odrzucono 1" z dwóch przebiegów); z `run_id` liczy się tylko ostatni.
        rows = [
            {"symbol": "ALL", "status": "trained_successfully", "run_id": "rerun",
             "timestamp": "2026-10-11T03:50:00+00:00"},
            {"symbol": "ALL", "status": "skipped_validation_failed", "run_id": "cron",
             "timestamp": "2026-10-11T03:40:00+00:00"},
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.summary.total == 1
        assert view.summary.accepted == 1

    def test_metrics_come_from_the_newest_accepted_row_with_its_symbol_and_n(self) -> None:
        # Najnowszy wiersz z metrykami to odrzucony kandydat MSFT — dawniej karta
        # pokazywała jego RMSE bez nazwy, jakby to był wdrożony model.
        rows = [
            {"symbol": "MSFT", "status": "skipped_validation_failed",
             "timestamp": "2026-10-11T03:45:00+00:00",
             "candidate_holdout_rmse": 0.030, "baseline_rmse": 0.025,
             "n_folds": 4, "n_samples": 80},
            {"symbol": "ALL", "status": "trained_successfully",
             "timestamp": "2026-10-11T03:40:00+00:00",
             "candidate_holdout_rmse": 0.021, "baseline_rmse": 0.024,
             "candidate_holdout_directional_hit_rate": 0.58,
             "n_folds": 4, "n_samples": 243},
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.candidate_rmse == 0.021
        assert view.metrics_symbol == "ALL"
        assert view.metrics_n_samples == 243

    def test_rejected_only_run_still_names_the_candidate(self) -> None:
        rows = [
            {"symbol": "ALL", "status": "skipped_validation_failed",
             "timestamp": "2026-10-11T03:45:00+00:00",
             "candidate_holdout_rmse": 0.030, "baseline_rmse": 0.025, "n_samples": 243},
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.candidate_rmse == 0.030
        assert view.metrics_symbol == "ALL"
        assert view.metrics_n_samples == 243


class TestColdStartHonesty:
    """MAIL-01 (G03): bez załadowanego modelu nie ma "sprawdzonego modelu", na
    którym można by jechać — noga ML to baseline "cena bez zmian"."""

    def test_rejected_run_without_loaded_model_does_not_claim_proven_model(self) -> None:
        summary = _summary(0, 4, 0)
        html = render_model_scorecard_html(summary, trained_at=_FRESH, now=_NOW, model_loaded=False)
        text = render_model_scorecard_text(summary, trained_at=_FRESH, now=_NOW, model_loaded=False)
        for out in (html, text):
            assert "sprawdzonym" not in out.lower()
            assert "odrzucon" in out.lower()
            assert "cold-start" in out

    def test_skipped_run_without_loaded_model_says_cold_start(self) -> None:
        # Stan produkcji z 2026-10-11: wszystkie symbole pominięte (za mało próbek).
        summary = _summary(0, 0, 45)
        html = render_model_scorecard_html(summary, trained_at=_FRESH, now=_NOW, model_loaded=False)
        assert "Brak wytrenowanego modelu" in html
        assert "cold-start" in html


class TestMetricsProvenance:
    """ML-03: metryki kandydata podpisane symbolem i liczbą próbek."""

    def test_html_names_the_symbol_and_sample_size(self) -> None:
        html = render_model_scorecard_html(
            _summary(1, 0, 0), trained_at=_FRESH, now=_NOW,
            candidate_rmse=0.021, baseline_rmse=0.024,
            metrics_symbol="ALL", n_samples=243,
        )
        assert "ALL" in html
        assert "n=243" in html

    def test_text_names_the_symbol_and_sample_size(self) -> None:
        text = render_model_scorecard_text(
            _summary(1, 0, 0), trained_at=_FRESH, now=_NOW,
            candidate_rmse=0.021, baseline_rmse=0.024,
            metrics_symbol="ALL", n_samples=243,
        )
        assert "ALL" in text
        assert "n=243" in text

    def test_symbol_is_escaped(self) -> None:
        html = render_model_scorecard_html(
            _summary(1, 0, 0), trained_at=_FRESH, now=_NOW,
            candidate_rmse=0.021, baseline_rmse=0.024, metrics_symbol="<b>X</b>",
        )
        assert "<b>X</b>" not in html


def _pooled_run_rows(pooled_status: str) -> list[dict[str, object]]:
    """Przebieg po ML-13: trzy ewaluacje per symbol, potem wiersz modelu wspólnego."""
    return [
        {"symbol": "AAPL", "status": "evaluated_passed_gate",
         "timestamp": "2026-10-11T03:30:00+00:00"},
        {"symbol": "MSFT", "status": "evaluated_passed_gate",
         "timestamp": "2026-10-11T03:31:00+00:00"},
        {"symbol": "NVDA", "status": "evaluated_failed_gate",
         "timestamp": "2026-10-11T03:32:00+00:00"},
        {"symbol": "ALL", "status": pooled_status,
         "timestamp": "2026-10-11T03:35:00+00:00",
         "candidate_holdout_rmse": 0.030, "baseline_rmse": 0.025, "n_samples": 243},
    ]


class TestPooledVerdict:
    """Decyzja 3 (2026-10-18): werdykt główny karty bierze się z wiersza modelu
    wspólnego `ALL`; ewaluacje per symbol (bez zapisu wag) idą osobną linią.
    Dawniej liczyły się jako „pominięte” obok jedynego modelu, który shipuje wagi."""

    def test_main_verdict_comes_from_the_pooled_row_only(self) -> None:
        view = build_scorecard_view(_pooled_run_rows("skipped_validation_failed"))

        assert view is not None
        assert (view.summary.total, view.summary.accepted, view.summary.rejected) == (1, 0, 1)
        assert view.summary.skipped == 0

    def test_per_symbol_evaluations_are_counted_separately(self) -> None:
        view = build_scorecard_view(_pooled_run_rows("trained_successfully"))

        assert view is not None
        assert view.evaluations is not None
        assert (view.evaluations.passed, view.evaluations.failed) == (2, 1)

    def test_metrics_come_only_from_the_pooled_row_when_it_exists(self) -> None:
        # A1c (2026-10-18): przy wierszu ALL bez metryk (trening wspólny pominięty)
        # nagłówek nie może pokazywać RMSE ewaluacji per symbol jak modelu wspólnego.
        rows = _pooled_run_rows("skipped")
        metric_keys = ("candidate_holdout_rmse", "baseline_rmse")
        rows[-1] = {k: v for k, v in rows[-1].items() if k not in metric_keys}
        rows[0] = {**rows[0], "candidate_holdout_rmse": 0.011, "baseline_rmse": 0.020}

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.candidate_rmse is None
        assert view.metrics_symbol is None

    def test_run_without_pooled_row_keeps_the_legacy_aggregate(self) -> None:
        rows = [
            {"symbol": "AAPL", "status": "trained_successfully",
             "timestamp": "2026-07-09T10:00:00+00:00"},
        ]

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.evaluations is None

    def test_html_and_text_show_the_evaluations_line(self) -> None:
        view = build_scorecard_view(_pooled_run_rows("trained_successfully"))
        assert view is not None

        for render in (render_model_scorecard_html, render_model_scorecard_text):
            out = render(
                view.summary, trained_at=_FRESH, now=_NOW, evaluations=view.evaluations
            )
            assert "model wspólny" in out
            assert "zaakceptowano 1, odrzucono 0, pominięto 0" in out
            assert "Ewaluacje per symbol" in out
            assert "bramkę przeszło 2, nie przeszło 1, pominięto 0" in out

    def test_no_evaluations_line_without_pooled_split(self) -> None:
        for render in (render_model_scorecard_html, render_model_scorecard_text):
            out = render(_summary(1, 0, 0), trained_at=_FRESH, now=_NOW)
            assert "Ewaluacje per symbol" not in out
            assert "model wspólny" not in out


class TestThreeBaselines:
    """ML-10: bramka bije najsilniejszy z baseline'ów „zwrot 0", „dryf średniej"
    i „sam LLM" — karta pokazuje wszystkie trzy, gdy wiersz je ma (migracja 028)."""

    def test_view_reads_the_drift_and_llm_baselines_from_the_pooled_row(self) -> None:
        rows = _pooled_run_rows("skipped_validation_failed")
        rows[-1] = {**rows[-1], "drift_baseline_rmse": 0.024, "llm_baseline_rmse": 0.021}

        view = build_scorecard_view(rows)

        assert view is not None
        assert view.drift_baseline_rmse == 0.024
        assert view.llm_baseline_rmse == 0.021

    def test_html_and_text_show_all_three_baselines(self) -> None:
        for render in (render_model_scorecard_html, render_model_scorecard_text):
            out = render(
                _summary(1, 0, 0),
                trained_at=_FRESH,
                now=_NOW,
                candidate_rmse=0.0180,
                baseline_rmse=0.0200,
                drift_baseline_rmse=0.0195,
                llm_baseline_rmse=0.0190,
            )
            assert "zwrot 0" in out
            assert "0,0200" in out
            assert "dryf średniej" in out
            assert "0,0195" in out
            assert "sam LLM" in out
            assert "0,0190" in out

    def test_rows_without_the_new_baselines_render_as_before(self) -> None:
        # Wiersze sprzed migracji 028 albo zapis bez flagi: brak linii baseline'ów.
        for render in (render_model_scorecard_html, render_model_scorecard_text):
            out = render(
                _summary(1, 0, 0),
                trained_at=_FRESH,
                now=_NOW,
                candidate_rmse=0.0100,
                baseline_rmse=0.0200,
            )
            assert "dryf średniej" not in out
            assert "sam LLM" not in out
