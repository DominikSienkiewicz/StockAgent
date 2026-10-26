from __future__ import annotations

import json
import math
from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from src.domain.experiment import Experiment, ExperimentMetric
from src.tools.evaluate import (
    BaselineVerdict,
    EvalReport,
    format_report,
    render_evaluation_json,
    summarize_evaluation,
)


def _row(
    trend: str,
    correct: bool | None,
    price_at: float,
    predicted: float,
    actual: float,
) -> dict[str, Any]:
    return {
        "predicted_trend": trend,
        "is_trend_correct": correct,
        "price_at_prediction": price_at,
        "predicted_target_price": predicted,
        "actual_price_after_12h": actual,
    }


class TestSummarizeEvaluation:
    def test_empty_rows_returns_zeroed_report(self) -> None:
        report = summarize_evaluation([])
        assert report.sample_count == 0
        assert report.hit_rate is None
        assert report.model_rmse is None
        assert report.baseline_rmse is None
        assert report.beats_baseline is False

    def test_directional_hit_rate(self) -> None:
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", False, 100, 95, 101),
            _row("BULLISH", True, 100, 102, 103),
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 3
        assert report.hit_rate == 2 / 3

    def test_model_rmse_and_baseline_rmse(self) -> None:
        # 2 rekordy. Model: |pred-actual| = 1 i 1 → RMSE = 1.
        # Baseline (last-value = price_at): |price_at-actual| = 5 i 5 → RMSE = 5.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            _row("BULLISH", True, 100, 106, 105),
        ]
        report = summarize_evaluation(rows)
        assert report.model_rmse is not None
        assert report.baseline_rmse is not None
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)
        assert math.isclose(report.baseline_rmse, 5.0, rel_tol=1e-9)
        assert report.beats_baseline is True

    def test_model_worse_than_baseline_does_not_beat(self) -> None:
        rows = [
            _row("BULLISH", True, 100, 120, 101),  # model bardzo daleko
            _row("BULLISH", True, 100, 80, 99),
        ]
        report = summarize_evaluation(rows)
        assert report.beats_baseline is False

    def test_skips_rows_with_missing_values_for_rmse(self) -> None:
        # Wiersz bez actual nie psuje RMSE (jest pomijany), ale wlicza się do
        # hit-rate jeśli ma is_trend_correct.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            {
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "price_at_prediction": 100,
                "predicted_target_price": None,
                "actual_price_after_12h": None,
            },
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 2
        assert report.hit_rate == pytest.approx(1.0)
        # tylko 1 wiersz miał komplet cen → RMSE z jednego punktu
        assert report.model_rmse is not None
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)

    def test_report_is_dataclass(self) -> None:
        assert isinstance(summarize_evaluation([]), EvalReport)


class TestDirectionalHitRateSplit:
    """Audyt z czerwca 2026: blended `hit_rate` zalicza SIDEWAYS jako trafienie zawsze, gdy
    ruch < ±0.5% — czyli „nic się nie ruszyło" myli się z „trafiłem kierunek".
    Rozdzielamy: directional-only hit-rate (BULLISH/BEARISH) + pokrycie SIDEWAYS."""

    def test_directional_hit_rate_distinct_from_blended(self) -> None:
        # Mieszanka: 2 kierunkowe (1 trafiona) + 2 SIDEWAYS (obie „trafione",
        # bo dni spokojne). Blended = 3/4. Directional-only = 1/2 — różne liczby.
        rows = [
            _row("BULLISH", True, 100, 105, 106),  # kierunkowa, trafiona
            _row("BEARISH", False, 100, 95, 101),  # kierunkowa, pudło
            _row("SIDEWAYS", True, 100, 100, 100.1),  # cisza, „trafiona"
            _row("SIDEWAYS", True, 100, 100, 99.9),  # cisza, „trafiona"
        ]
        report = summarize_evaluation(rows)

        assert report.hit_rate == 3 / 4  # blended (back-compat)
        assert report.directional_hit_rate == 1 / 2  # tylko BULLISH/BEARISH
        assert report.directional_count == 2
        assert report.sideways_count == 2

    def test_directional_hit_rate_none_when_no_directional_rows(self) -> None:
        rows = [
            _row("SIDEWAYS", True, 100, 100, 100.1),
            _row("SIDEWAYS", True, 100, 100, 99.9),
        ]
        report = summarize_evaluation(rows)

        assert report.directional_hit_rate is None
        assert report.directional_count == 0
        assert report.sideways_count == 2
        # Blended wciąż liczone dla back-compat.
        assert report.hit_rate == pytest.approx(1.0)

    def test_directional_count_ignores_rows_without_flag(self) -> None:
        # Wiersz bez is_trend_correct nie wlicza się do żadnego licznika trafień.
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", None, 100, 95, 101),
        ]
        report = summarize_evaluation(rows)

        assert report.directional_count == 1
        assert report.directional_hit_rate == pytest.approx(1.0)
        assert report.sideways_count == 0


class TestOneHitRateDefinition:
    """WIA-63: evaluate.py liczy kierunkowość jedną funkcją domenową
    (`directional_hit_rate`) — SIDEWAYS i etykieta bez zakładu nie wchodzą do
    hit-rate'u kierunkowego, a ich liczba pochodzi z tej samej definicji."""

    def test_sideways_and_unknown_labels_stay_out_of_the_directional_rate(self) -> None:
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", False, 100, 95, 101),
            _row("SIDEWAYS", True, 100, 100, 100.1),
            _row(" sideways ", True, 100, 100, 99.9),
            _row("NEUTRAL", True, 100, 100, 100.0),
        ]

        report = summarize_evaluation(rows)

        assert report.directional_hit_rate == 1 / 2
        assert report.directional_count == 2

    def test_sideways_count_follows_the_domain_definition(self) -> None:
        # Etykieta spoza BULLISH/BEARISH nie stawia zakładu kierunkowego — domena
        # liczy ją jako brak zakładu, więc raport nie może jej gubić ani liczyć inaczej.
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row(" sideways ", True, 100, 100, 99.9),
            _row("NEUTRAL", True, 100, 100, 100.0),
        ]

        report = summarize_evaluation(rows)

        assert report.sideways_count == 2


class TestReturnSpaceRmse:
    """WIA-40: RMSE na zwrotach ((cel − p0)/p0 vs (rzeczywista − p0)/p0), bo BTC
    ~100k dominuje sumę kwadratów w cenach, a cold-start daje identyczne RMSE."""

    def test_rmse_is_scale_free_btc_does_not_dominate(self) -> None:
        rows = [
            _row("BULLISH", True, 100_000, 101_000, 100_000),  # błąd zwrotu 1%
            _row("BULLISH", False, 200, 200, 210),  # błąd zwrotu 5%
        ]

        report = summarize_evaluation(rows)

        assert report.model_rmse_return == pytest.approx(math.sqrt((0.01**2 + 0.05**2) / 2))
        # Baseline „brak zmiany” myli się o rzeczywisty zwrot: 0% i 5%.
        assert report.baseline_rmse_return == pytest.approx(math.sqrt((0.0**2 + 0.05**2) / 2))

    def test_verdict_undetermined_when_targets_equal_entry_prices(self) -> None:
        # Cold-start: model bez wag zwraca cenę wejścia jako cel, więc jego błąd
        # jest błędem baseline'u — „NIE bije” byłoby artefaktem, nie pomiarem.
        rows = [
            _row("SIDEWAYS", True, 100, 100, 100.2),
            _row("SIDEWAYS", False, 50, 50, 53),
        ]

        report = summarize_evaluation(rows)

        assert report.verdict is BaselineVerdict.UNDETERMINED
        assert report.beats_baseline is False
        assert "nieokreślony" in format_report(report, 30)
        assert "NIE bije" not in format_report(report, 30)

    def test_verdict_without_complete_prices_is_undetermined(self) -> None:
        report = summarize_evaluation([])

        assert report.model_rmse_return is None
        assert report.verdict is BaselineVerdict.UNDETERMINED

    def test_model_closer_to_the_outcome_beats_the_baseline(self) -> None:
        rows = [_row("BULLISH", True, 100, 104, 105), _row("BULLISH", True, 100, 106, 105)]

        report = summarize_evaluation(rows)

        assert report.verdict is BaselineVerdict.BEATS
        assert report.beats_baseline is True
        assert "✅ bije baseline" in format_report(report, 30)

    def test_model_further_than_no_change_does_not_beat(self) -> None:
        rows = [_row("BULLISH", True, 100, 120, 101), _row("BULLISH", True, 100, 80, 99)]

        report = summarize_evaluation(rows)

        assert report.verdict is BaselineVerdict.DOES_NOT_BEAT
        assert "⚠️ NIE bije baseline" in format_report(report, 30)

    def test_model_and_baseline_are_scored_on_the_same_rows(self) -> None:
        # Wiersz bez celu nie może zaniżyć błędu modelu, a zostać w baseline'ie.
        rows = [
            _row("BULLISH", True, 100, 110, 110),
            {**_row("BULLISH", True, 100, 0, 150), "predicted_target_price": None},
        ]

        report = summarize_evaluation(rows)

        assert report.model_rmse_return == pytest.approx(0.0)
        assert report.baseline_rmse_return == pytest.approx(0.10)


class TestRmsePerAssetClass:
    def _rows(self) -> list[dict[str, Any]]:
        return [
            {**_row("BULLISH", True, 100_000, 101_000, 100_000), "asset_class": "CRYPTO"},
            {**_row("BULLISH", False, 200, 200, 210), "asset_class": "STOCK"},
            {**_row("BEARISH", True, 100, 98, 98), "asset_class": "STOCK"},
        ]

    def test_each_class_gets_its_own_scale_free_rmse(self) -> None:
        report = summarize_evaluation(self._rows())

        crypto = report.by_asset_class["CRYPTO"]
        stock = report.by_asset_class["STOCK"]
        assert (crypto.n, stock.n) == (1, 2)
        assert crypto.model_rmse == pytest.approx(0.01)
        assert stock.model_rmse == pytest.approx(math.sqrt((0.05**2 + 0.0**2) / 2))

    def test_rows_without_a_class_produce_no_breakdown(self) -> None:
        report = summarize_evaluation([_row("BULLISH", True, 100, 104, 105)])

        assert report.by_asset_class == {}

    def test_formatted_report_lists_the_classes(self) -> None:
        text = format_report(summarize_evaluation(self._rows()), 30)

        assert "per klasa aktywa" in text
        assert "CRYPTO" in text
        assert "STOCK" in text


class TestConfidenceScores:
    """WIA-40: Brier i log loss deklarowanej pewności LLM na zakładach kierunkowych."""

    def test_brier_and_log_loss_of_the_declared_confidence(self) -> None:
        rows = [
            {**_row("BULLISH", True, 100, 105, 106), "confidence_score": 0.9},
            {**_row("BEARISH", False, 100, 95, 101), "confidence_score": 0.8},
            {**_row("SIDEWAYS", True, 100, 100, 100.1), "confidence_score": 0.99},
        ]

        scores = summarize_evaluation(rows).confidence_scores

        assert scores is not None
        assert scores.n == 2  # SIDEWAYS nie stawia zakładu kierunkowego
        assert scores.brier == pytest.approx(((0.9 - 1) ** 2 + (0.8 - 0) ** 2) / 2)
        assert scores.log_loss == pytest.approx(-(math.log(0.9) + math.log(0.2)) / 2)

    def test_no_confidence_column_means_no_scores(self) -> None:
        report = summarize_evaluation([_row("BULLISH", True, 100, 105, 106)])

        assert report.confidence_scores is None

    def test_formatted_report_prints_brier_and_log_loss(self) -> None:
        rows = [{**_row("BULLISH", True, 100, 105, 106), "confidence_score": 0.9}]

        text = format_report(summarize_evaluation(rows), 30)

        assert "Brier pewności" in text
        assert "Log loss pewności" in text


class TestJsonOutput:
    """WIA-40: `--json` — te same liczby maszynowo, do porównania przebiegów."""

    def _rows(self) -> list[dict[str, Any]]:
        return [
            {
                **_row("BULLISH", True, 100_000, 101_000, 100_000),
                "asset_class": "CRYPTO",
                "confidence_score": 0.9,
                "timestamp": "2026-10-01T14:00:00+00:00",
            },
            {
                **_row("BEARISH", False, 200, 190, 210),
                "asset_class": "STOCK",
                "confidence_score": 0.7,
                "timestamp": "2026-10-02T14:00:00+00:00",
            },
        ]

    def test_json_carries_the_headline_numbers_and_a_machine_verdict(self) -> None:
        payload = json.loads(render_evaluation_json(self._rows(), days=30))

        assert payload["days"] == 30
        assert payload["sample_count"] == 2
        assert payload["directional"] == {"hit_rate": 0.5, "n": 2, "sideways": 0}
        assert payload["rmse_return"]["verdict"] in {"beats", "does_not_beat", "undetermined"}
        assert payload["rmse_return"]["model"] == pytest.approx(math.sqrt((0.01**2 + 0.1**2) / 2))

    def test_json_lists_classes_and_confidence_scores(self) -> None:
        payload = json.loads(render_evaluation_json(self._rows(), days=30))

        assert sorted(payload["by_asset_class"]) == ["CRYPTO", "STOCK"]
        assert payload["by_asset_class"]["CRYPTO"]["n"] == 1
        assert payload["confidence"]["n"] == 2
        assert payload["confidence"]["brier"] == pytest.approx(((0.9 - 1) ** 2 + 0.7**2) / 2)

    def test_empty_run_is_valid_json_with_null_metrics(self) -> None:
        payload = json.loads(render_evaluation_json([], days=7))

        assert payload["sample_count"] == 0
        assert payload["rmse_return"] == {
            "model": None,
            "baseline": None,
            "verdict": "undetermined",
        }
        assert payload["confidence"] is None

    def test_json_includes_requested_segments(self) -> None:
        rows = [
            {
                **_row("BULLISH", True, 100, 101, 101),
                "asset_class": "STOCK",
                "timestamp": f"2026-10-{day:02d}T14:00:00+00:00",
            }
            for day in range(1, 4)
        ]

        payload = json.loads(render_evaluation_json(rows, days=30, segmentations=("asset_class",)))

        segment = payload["segments"]["asset_class"][0]
        assert segment["key"] == "STOCK"
        assert segment["n"] == 3


class TestScoresCli:
    """WIA-40: `--scores` czyta symbol i pewność (każde istnieje od swojej migracji,
    więc tylko na żądanie), a `--json` oddaje wyłącznie JSON."""

    def test_scores_flag_reads_symbol_and_confidence_and_classifies_symbols(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.crypto_symbols = ["BTC"]
        settings.symbols_etf = []
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {
                **_row("BULLISH", True, 100_000, 101_000, 100_000),
                "symbol": "BTC",
                "confidence_score": 0.9,
            },
            {**_row("BULLISH", False, 200, 200, 210), "symbol": "AAPL", "confidence_score": 0.6},
        ]

        main_tools.main(["evaluate", "--scores"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("confidence_score", "symbol")
        )
        out = capsys.readouterr().out
        assert "per klasa aktywa" in out
        assert "CRYPTO" in out
        assert "Brier pewności" in out

    def test_json_flag_prints_only_json(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            _row("BULLISH", True, 100, 104, 105)
        ]

        code = main_tools.main(["evaluate", "--json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sample_count"] == 1

    def test_json_flag_keeps_the_missing_column_notice_off_stdout(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools
        from src.infrastructure.adapters.supabase_repo import MissingColumnError

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.side_effect = [
            MissingColumnError("gate_delta"),
            [],
        ]

        main_tools.main(["evaluate", "--echo", "--json"])

        captured = capsys.readouterr()
        json.loads(captured.out)
        assert "gate_delta" in captured.err


class TestSignificanceVerdict:
    """WIA-22: ewaluacja offline mówi, czy trafność kierunkowa bije najlepszy
    naiwny baseline na tym samym zbiorze — nie tylko goły procent."""

    def test_report_carries_directional_evidence(self) -> None:
        rows = [
            {**_row("BULLISH", True, 100, 101, 101), "timestamp": "2026-10-01T14:00:00+00:00"},
            {**_row("BULLISH", False, 100, 101, 99), "timestamp": "2026-10-02T14:00:00+00:00"},
        ]

        report = summarize_evaluation(rows)

        assert report.evidence is not None
        assert report.evidence.hit_rate.directional == 2
        assert report.evidence.cycles == 2

    def test_formatted_report_states_the_verdict(self) -> None:
        rows = [
            {
                **_row("BULLISH", i % 2 == 0, 100, 101, 101 if i % 2 == 0 else 99),
                "timestamp": f"2026-10-{i + 1:02d}T14:00:00+00:00",
            }
            for i in range(6)
        ]

        text = format_report(summarize_evaluation(rows), 30)

        assert "nieodróżnialne od rzutu monetą" in text
        assert "zawsze-BULLISH: 50%" in text


class TestRecipeSegments:
    """WIA-20: ewaluacja segmentuje trafność po wersji przepisu — inaczej wynik
    miesza prompty, modele i persony, a zmiana przepisu udaje umiejętność."""

    def test_segments_directional_hit_rate_by_recipe_version(self) -> None:
        from src.domain.equity_curve import DirectionalHitRate
        from src.tools.evaluate import summarize_by_version

        rows = [
            {**_row("BULLISH", True, 100, 101, 101), "recipe_version": "r1" * 32},
            {**_row("BEARISH", False, 100, 99, 101), "recipe_version": "r1" * 32},
            {**_row("BULLISH", True, 100, 101, 101), "recipe_version": "r2" * 32},
            {**_row("BULLISH", None, 100, 101, 101), "recipe_version": "r2" * 32},
        ]

        segments = summarize_by_version(rows, "recipe_version")

        assert segments == {
            "r1" * 32: DirectionalHitRate(hits=1, directional=2, sideways=0),
            "r2" * 32: DirectionalHitRate(hits=1, directional=1, sideways=0),
        }

    def test_rows_without_a_version_are_reported_as_unversioned(self) -> None:
        from src.domain.recipe import UNVERSIONED
        from src.tools.evaluate import summarize_by_version

        segments = summarize_by_version([_row("BULLISH", True, 100, 101, 101)], "recipe_version")

        assert list(segments) == [UNVERSIONED]

    def test_formatted_segments_show_short_versions_and_counts(self) -> None:
        from src.tools.evaluate import format_segments, summarize_by_version

        rows = [
            {**_row("BULLISH", True, 100, 101, 101), "recipe_version": "abcdef123456" + "0" * 52},
            {**_row("BULLISH", False, 100, 101, 99), "recipe_version": "abcdef123456" + "0" * 52},
        ]

        text = format_segments(summarize_by_version(rows, "recipe_version"), "przepis")

        assert "abcdef123456" in text
        assert "0" * 52 not in text
        assert "50.0%" in text
        assert "n=2" in text

    def test_by_recipe_flag_reads_the_recipe_column(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 101, 101), "recipe_version": "f" * 64},
        ]

        main_tools.main(["evaluate", "--by-recipe"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, with_recipe=True
        )
        assert "f" * 12 in capsys.readouterr().out

    def test_default_run_does_not_touch_the_recipe_column(self, mocker: MockerFixture) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = []

        main_tools.main(["evaluate"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(days=30)


class TestUniverseSegments:
    """WIA-48: trafność per wersja uniwersum — efekt „dodałem symbole” oddzielony
    od zmiany trafności. Wersja jest w kwicie v2 (`decision_receipts`)."""

    def test_by_universe_flag_reads_the_receipt_universe(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 101, 101), "universe_version": "e" * 64},
            {**_row("BULLISH", False, 100, 101, 99)},
        ]

        main_tools.main(["evaluate", "--by-universe"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, with_universe=True
        )
        out = capsys.readouterr().out
        assert "e" * 12 in out
        assert "unversioned" in out


def _day(i: int) -> str:
    return f"2026-08-{i + 1:02d}T14:00:00+00:00"


class TestComponentSegmentation:
    """WIA-55: ablacja komponentów na tych samych wierszach — trend LLM,
    `final_recommendation` rady (z JSONB) i kierunek XGBoost. Bez niej nie da
    się powiedzieć, która noga niesie trafność, a która ją rozcieńcza."""

    def _rows(self) -> list[dict[str, Any]]:
        # Trzy wzrosty z radą BUY: rada trafia 3/3, trend LLM tylko 1/3.
        # Trzy wiersze bez zakładu (LLM SIDEWAYS, rada HOLD albo pominięta).
        # XGBoost bez wag: cel = cena (baseline „cena bez zmian”) → brak zakładu.
        return [
            {
                **_row("BULLISH", True, 100, 100, 102),
                "council_recommendation": "BUY",
                "timestamp": _day(0),
            },
            {
                **_row("BEARISH", False, 100, 100, 102),
                "council_recommendation": "BUY",
                "timestamp": _day(1),
            },
            {
                **_row("BEARISH", False, 100, 100, 101),
                "council_recommendation": "BUY",
                "timestamp": _day(2),
            },
            {
                **_row("SIDEWAYS", True, 100, 100, 100.2),
                "council_recommendation": "HOLD",
                "timestamp": _day(3),
            },
            {
                **_row("SIDEWAYS", True, 100, 100, 99.9),
                "council_recommendation": "HOLD",
                "timestamp": _day(4),
            },
            {
                **_row("SIDEWAYS", False, 100, 100, 101),
                "council_recommendation": None,
                "timestamp": _day(5),
            },
        ]

    def test_component_segment_scores_council_and_llm_on_the_same_rows(self) -> None:
        report = summarize_evaluation(self._rows(), segment_by="component")

        segments = {s.key: s for s in report.segments["component"]}
        assert segments["council"].rate == pytest.approx(1.0)
        assert segments["council"].n == 3
        assert segments["llm"].rate == pytest.approx(1 / 3)
        assert segments["llm"].n == 3

    def test_model_without_weights_places_no_directional_bet(self) -> None:
        report = summarize_evaluation(self._rows(), segment_by="component")

        model = {s.key: s for s in report.segments["component"]}["xgboost"]
        assert model.n == 0
        assert model.rate is None

    def test_model_direction_comes_from_its_target_price(self) -> None:
        rows = [
            {**_row("BULLISH", True, 100, 103, 102), "timestamp": _day(0)},
            {**_row("BULLISH", False, 100, 97, 99), "timestamp": _day(1)},
            {**_row("BULLISH", True, 100, 98, 101), "timestamp": _day(2)},
        ]

        report = summarize_evaluation(rows, segment_by="component")

        model = {s.key: s for s in report.segments["component"]}["xgboost"]
        assert (model.n, cast(Any, model.rate)) == (3, pytest.approx(2 / 3))

    def test_segments_below_ten_bets_are_flagged_as_insufficient(self) -> None:
        report = summarize_evaluation(self._rows(), segment_by="component")

        assert all(s.insufficient for s in report.segments["component"])

    def test_components_are_listed_in_a_fixed_order(self) -> None:
        report = summarize_evaluation(self._rows(), segment_by="component")

        assert [s.key for s in report.segments["component"]] == ["llm", "council", "xgboost"]

    def test_formatted_segments_print_n_and_the_insufficient_flag(self) -> None:
        from src.tools.evaluate import format_segment_evidence

        report = summarize_evaluation(self._rows(), segment_by="component")

        text = format_segment_evidence("component", report.segments["component"])

        assert "council" in text
        assert "100.0%" in text
        assert "n=3" in text
        assert "za mało danych" in text

    def test_default_report_has_no_segments(self) -> None:
        assert summarize_evaluation(self._rows()).segments == {}

    def test_unknown_segmentation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="segment"):
            summarize_evaluation(self._rows(), segment_by="persona")


class TestContextSegmentation:
    """WIA-55: segmenty kontekstu z istniejących kolumn — reżim (017), pewność
    (016), znak Alpha Fusion (022) i etykieta edge vs opcje (025)."""

    def test_regime_segments_correct_for_multiple_comparisons(self) -> None:
        # RISK_ON: 20 trafień na zmianę BULLISH/BEARISH w 20 dniach (baseline 50%)
        # → istotne. RISK_OFF: 10 zakładów, połowa trafiona → nieistotne.
        strong = [
            {
                **_row(trend, True, 100, 100, 101 if trend == "BULLISH" else 99),
                "regime": "RISK_ON",
                "timestamp": _day(i),
            }
            for i, trend in enumerate(["BULLISH", "BEARISH"] * 10)
        ]
        weak = [
            {
                **_row("BULLISH", i % 2 == 0, 100, 100, 101 if i % 2 == 0 else 99),
                "regime": "RISK_OFF",
                "timestamp": _day(i),
            }
            for i in range(10)
        ]

        report = summarize_evaluation(strong + weak, segment_by="regime")

        segments = {s.key: s for s in report.segments["regime"]}
        assert segments["RISK_ON"].fdr_significant is True
        assert segments["RISK_ON"].insufficient is False
        assert segments["RISK_OFF"].fdr_significant is False
        assert segments["RISK_OFF"].evidence.baselines.always_bullish == pytest.approx(0.5)

    def test_untagged_rows_form_the_unknown_regime(self) -> None:
        report = summarize_evaluation([_row("BULLISH", True, 100, 100, 101)], segment_by="regime")

        assert [s.key for s in report.segments["regime"]] == ["UNKNOWN"]

    def test_confidence_buckets_split_declared_confidence(self) -> None:
        rows = [
            {**_row("BULLISH", True, 100, 100, 101), "confidence_score": 0.55},
            {**_row("BULLISH", True, 100, 100, 101), "confidence_score": 0.85},
            {**_row("BULLISH", False, 100, 100, 99), "confidence_score": None},
        ]

        report = summarize_evaluation(rows, segment_by="confidence_bucket")

        assert {s.key: s.n for s in report.segments["confidence_bucket"]} == {
            "0.00-0.60": 1,
            "0.80-1.00": 1,
            "unknown": 1,
        }

    def test_alpha_fusion_sign_and_edge_label_segments(self) -> None:
        rows = [
            {**_row("BULLISH", True, 100, 100, 101), "alpha_fusion_score": 0.4, "edge_sigma": 2.0},
            {**_row("BEARISH", True, 100, 100, 99), "alpha_fusion_score": -0.2, "edge_sigma": None},
        ]

        report = summarize_evaluation(rows, segment_by=("alpha_fusion_sign", "edge_label"))

        assert [s.key for s in report.segments["alpha_fusion_sign"]] == ["negative", "positive"]
        assert [s.key for s in report.segments["edge_label"]] == ["MODEL_AHEAD", "NO_SIGNAL"]


class TestSegmentationCli:
    """WIA-55: `--by` czyta kolumny segmentu wyłącznie na żądanie — domyślny
    przebieg nie dotyka żadnej z nich (każda istnieje dopiero po swojej migracji)."""

    def test_by_component_reads_the_council_recommendation(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 100, 101), "council_recommendation": "BUY"},
        ]

        main_tools.main(["evaluate", "--by", "component"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("council_recommendation",)
        )
        out = capsys.readouterr().out
        assert "council" in out
        assert "za mało danych" in out

    def test_by_asset_class_classifies_symbols_from_the_config(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.crypto_symbols = ["BTC"]
        settings.symbols_etf = ["VOO"]
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "BTC"},
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "VOO"},
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "AAPL"},
        ]

        main_tools.main(["evaluate", "--by", "asset_class"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("symbol",)
        )
        out = capsys.readouterr().out
        assert "CRYPTO" in out
        assert "ETF" in out
        assert "STOCK" in out

    def test_several_segmentations_read_their_fields_once(self, mocker: MockerFixture) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = []

        main_tools.main(
            ["evaluate", "--by", "regime", "--by", "confidence_bucket", "--by", "regime"]
        )

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("confidence_score", "regime")
        )


def _gated(trend: str, gate_delta: float | None, actual: float, day: int = 0) -> dict[str, Any]:
    """Rozliczona predykcja z deltą bramki (EXP-01); cena przy predykcji 100."""
    return {
        **_row(trend, _trend_correct(trend, actual), 100, 100, actual),
        "gate_delta": gate_delta,
        "timestamp": _day(day),
    }


def _trend_correct(trend: str, actual: float) -> bool:
    if trend == "BULLISH":
        return actual > 100
    if trend == "BEARISH":
        return actual < 100
    return abs(actual - 100) <= 0.5


def test_echo_rate_counts_trend_matching_gate_delta_sign() -> None:
    # EXP-01: dwa z trzech trendów powtarzają znak ruchu, który otworzył bramkę.
    from src.tools.evaluate import summarize_echo

    rows = [
        _gated("BULLISH", 0.031, 101, day=0),
        _gated("BEARISH", -0.022, 99, day=1),
        _gated("BULLISH", -0.050, 101, day=2),
    ]

    echo = summarize_echo(rows)

    assert echo.with_delta == 3
    assert echo.echo_rate == pytest.approx(2 / 3)
    assert echo.reversal_rate == pytest.approx(1 / 3)


def test_momentum_baseline_hit_rate_uses_gate_delta_sign() -> None:
    # „Znak ruchu” trafia, gdy ruch po predykcji ma znak delty bramki;
    # „przeciwny znak” — gdy odwrotny. Te same wiersze co hit-rate LLM.
    from src.tools.evaluate import summarize_echo

    rows = [
        _gated("BULLISH", 0.030, 101, day=0),  # momentum ✓, LLM ✓ (echo)
        _gated("BULLISH", -0.030, 101, day=1),  # momentum ✗, LLM ✓ (odwrócenie)
        _gated("BEARISH", -0.025, 98, day=2),  # momentum ✓, LLM ✓ (echo)
        _gated("BEARISH", 0.040, 101, day=3),  # momentum ✓, LLM ✗ (odwrócenie)
    ]

    echo = summarize_echo(rows)

    assert echo.compared == 4
    assert echo.momentum_rate == pytest.approx(3 / 4)
    assert echo.contrarian_rate == pytest.approx(1 / 4)
    assert echo.llm_rate == pytest.approx(3 / 4)


class TestMomentumEcho:
    """EXP-01: test echa liczony WYŁĄCZNIE w `evaluate` (nie w recapie ani mailu)."""

    def test_llm_hit_rate_where_it_disagrees_with_momentum(self) -> None:
        from src.tools.evaluate import summarize_echo

        rows = [
            _gated("BULLISH", 0.030, 101, day=0),
            _gated("BULLISH", -0.030, 101, day=1),
            _gated("BEARISH", 0.040, 101, day=2),
        ]

        echo = summarize_echo(rows)

        assert echo.disagreement == 2
        assert echo.disagreement_rate == pytest.approx(1 / 2)

    def test_rows_without_gate_delta_or_directional_call_are_not_compared(self) -> None:
        from src.tools.evaluate import summarize_echo

        rows = [
            _gated("BULLISH", None, 101, day=0),
            _gated("SIDEWAYS", 0.030, 100.2, day=1),
            _gated("BULLISH", 0.030, 101, day=2),
        ]

        echo = summarize_echo(rows)

        assert echo.with_delta == 2
        assert echo.echo_rate == pytest.approx(1 / 2)
        assert echo.compared == 1

    def test_formatted_echo_prints_n_n_eff_and_bh(self) -> None:
        from src.tools.evaluate import format_echo, summarize_echo

        rows = [
            _gated("BULLISH", 0.030, 101, day=0),
            _gated("BULLISH", -0.030, 101, day=1),
            _gated("BEARISH", 0.040, 101, day=2),
        ]

        text = format_echo(summarize_echo(rows))

        assert "Echo momentum" in text
        assert "66.7%" in text
        assert "n=3" in text
        assert "n_eff" in text
        assert "BH" in text

    def test_echo_flag_reads_the_gate_delta_column(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            _gated("BULLISH", 0.030, 101),
        ]

        main_tools.main(["evaluate", "--echo"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("gate_delta",)
        )
        assert "Echo momentum" in capsys.readouterr().out

    def test_missing_gate_delta_column_prints_the_migration_instead_of_failing(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools
        from src.infrastructure.adapters.supabase_repo import MissingColumnError

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        read = repo_cls.return_value.get_resolved_predictions_for_eval
        read.side_effect = [
            MissingColumnError("gate_delta"),
            [_row("BULLISH", True, 100, 101, 101)],
        ]

        code = main_tools.main(["evaluate", "--echo", "--by", "regime"])

        assert code == 0
        assert read.call_args_list[1].kwargs == {"days": 30, "extra_fields": ("regime",)}
        out = capsys.readouterr().out
        assert "kolumna gate_delta niedostępna — uruchom migrację 034" in out
        assert "Próbek (zamkniętych predykcji): 1" in out
        assert "Echo momentum" not in out


class TestLlmFallbackRowsAreSkipped:
    """Atrapa SIDEWAYS po awarii LLM (`llm_analysis_failed`) nie wchodzi do
    wyniku nagłówkowego, segmentów ani mianownika testu echa — odczyt z prawdziwego
    repozytorium (mock wyłącznie klienta Supabase), jak mail i Track Record."""

    def _rows_from_repo(self, mocker: MockerFixture) -> list[dict[str, Any]]:
        from src.infrastructure.adapters.supabase_repo import SupabaseRepository

        create_client = mocker.patch(
            "src.infrastructure.adapters.supabase_repo.build_supabase_client"
        )
        client = create_client.return_value
        settled = client.table.return_value.select.return_value.gte.return_value.not_.is_
        order = settled.return_value.order.return_value
        order.range.return_value.execute.return_value.data = [
            {
                **_gated("BULLISH", 0.030, 101, day=0),
                "council_recommendation": "BUY",
                "data_quality_flags": [],
            },
            {
                **_gated("SIDEWAYS", 0.030, 101, day=1),
                "council_recommendation": "HOLD",
                "data_quality_flags": ["llm_analysis_failed"],
            },
        ]
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")
        return repo.get_resolved_predictions_for_eval(
            30, extra_fields=("council_recommendation", "gate_delta")
        )

    def test_fallback_row_is_not_in_the_echo_denominator(self, mocker: MockerFixture) -> None:
        from src.tools.evaluate import summarize_echo

        echo = summarize_echo(self._rows_from_repo(mocker))

        assert echo.with_delta == 1
        assert echo.echo_rate == pytest.approx(1.0)

    def test_fallback_row_is_not_in_the_headline_or_a_segment(self, mocker: MockerFixture) -> None:
        report = summarize_evaluation(self._rows_from_repo(mocker), segment_by="component")

        assert report.sample_count == 1
        assert report.sideways_count == 0
        llm = {s.key: s for s in report.segments["component"]}["llm"]
        assert (llm.n, llm.evidence.hit_rate.sideways) == (1, 0)


def test_default_confidence_is_segmented_apart_from_the_low_band() -> None:
    # Pewność 0.5 to wartość domyślna przy braku pewności — nie zaniża pasma 0.00-0.60.
    rows = [
        {**_row("BULLISH", False, 100, 100, 99), "confidence_score": 0.5},
        {**_row("BULLISH", True, 100, 100, 101), "confidence_score": 0.55},
    ]

    report = summarize_evaluation(rows, segment_by="confidence_bucket")

    segments = {s.key: s for s in report.segments["confidence_bucket"]}
    assert (segments["0.50 (domyślna)"].n, segments["0.50 (domyślna)"].rate) == (1, 0.0)
    assert (segments["0.00-0.60"].n, segments["0.00-0.60"].rate) == (1, 1.0)


def _council_experiment() -> Experiment:
    from datetime import date

    return Experiment(
        id="EXP-02",
        hypothesis="Rada bez werdyktu LLM jest trafniejsza.",
        arms=("sighted", "blind"),
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        primary_metric=ExperimentMetric.COUNCIL_HIT_RATE,
    )


def _arm_row(arm: str | None, day: int, *, source: str = "column", **fields: Any) -> dict[str, Any]:
    """Rozliczona predykcja BULLISH z ramieniem EXP-02 w kolumnie albo w kwicie."""
    row = {
        **_row("BULLISH", True, 100, 100, 101),
        "timestamp": _day(day),
        "confidence_score": 0.7,
        "council_recommendation": "BUY",
        "council_consensus": "0.8",
        **fields,
    }
    if arm is not None:
        key = "experiment_arms" if source == "column" else "receipt_experiment_arms"
        row[key] = {"EXP-02": arm}
    return row


class TestExperimentEvaluation:
    """EXP-05: `evaluate --experiment <id>` — różnica metryk między ramionami
    (zabieg − kontrola) z n i n_eff per ramię i CI z bootstrapu blokowego po dniach."""

    def test_rows_are_split_by_the_arm_from_column_or_receipt(self) -> None:
        from src.tools.evaluate import summarize_experiment

        rows = [
            _arm_row("sighted", 0),
            _arm_row("blind", 0, source="receipt"),
            _arm_row("blind", 1, council_recommendation="SELL"),
            _arm_row(None, 2),
        ]

        report = summarize_experiment(rows, _council_experiment())

        assert report.arm_counts == {"sighted": 1, "blind": 2}
        hit = {c.metric.value: c for c in report.comparisons}["council_hit_rate"]
        assert (hit.control.n, hit.treatment.n) == (1, 2)
        assert hit.control.mean == pytest.approx(1.0)
        assert hit.treatment.mean == pytest.approx(0.5)
        assert hit.difference == pytest.approx(-0.5)

    def test_column_wins_over_the_receipt(self) -> None:
        from src.tools.evaluate import summarize_experiment

        row = {**_arm_row("blind", 0), "receipt_experiment_arms": {"EXP-02": "sighted"}}

        assert summarize_experiment([row], _council_experiment()).arm_counts == {
            "sighted": 0,
            "blind": 1,
        }

    def test_council_brier_uses_the_consensus_strength(self) -> None:
        from src.tools.evaluate import summarize_experiment

        report = summarize_experiment([_arm_row("sighted", 0)], _council_experiment())

        brier = {c.metric.value: c for c in report.comparisons}["council_brier"]
        assert brier.control.mean == pytest.approx((0.8 - 1.0) ** 2)

    def test_formatted_section_names_the_hypothesis_primary_metric_and_n(self) -> None:
        from src.tools.evaluate import format_experiment, summarize_experiment

        rows = [
            *(_arm_row("sighted", day) for day in range(25)),
            *(_arm_row("blind", day, council_recommendation="SELL") for day in range(25)),
        ]

        text = format_experiment(summarize_experiment(rows, _council_experiment()))

        assert "EXP-02" in text
        assert "Rada bez werdyktu LLM jest trafniejsza." in text
        assert "★ trafność rady" in text
        assert "blind − sighted" in text
        assert "-100.0 pp" in text
        assert "n=25" in text
        assert "n_eff" in text
        assert "CI blokowe" in text

    def test_small_sample_is_flagged(self) -> None:
        from src.tools.evaluate import format_experiment, summarize_experiment

        text = format_experiment(
            summarize_experiment([_arm_row("sighted", 0)], _council_experiment())
        )

        assert "za mało danych" in text
        assert "za mało cykli" in text

    def test_experiment_flag_reads_arms_and_council_fields(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.experiment_definitions.return_value = (_council_experiment(),)
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        read = repo_cls.return_value.get_resolved_predictions_for_eval
        read.return_value = [_arm_row("sighted", 0)]

        code = main_tools.main(["evaluate", "--experiment", "EXP-02"])

        assert code == 0
        assert read.call_args.kwargs == {
            "days": 30,
            "extra_fields": (
                "confidence_score",
                "council_consensus",
                "council_recommendation",
                "experiment_arms",
                "receipt_experiment_arms",
            ),
        }
        assert "Eksperyment EXP-02" in capsys.readouterr().out

    def test_missing_arms_column_falls_back_to_the_receipt(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools
        from src.infrastructure.adapters.supabase_repo import MissingColumnError

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.experiment_definitions.return_value = (_council_experiment(),)
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        read = repo_cls.return_value.get_resolved_predictions_for_eval
        read.side_effect = [
            MissingColumnError("experiment_arms"),
            [_arm_row("blind", 0, source="receipt")],
        ]

        code = main_tools.main(["evaluate", "--experiment", "EXP-02"])

        assert code == 0
        assert "experiment_arms" not in read.call_args_list[1].kwargs["extra_fields"]
        out = capsys.readouterr().out
        assert "kolumna experiment_arms niedostępna — uruchom migrację 037" in out
        assert "blind 1" in out

    def test_unknown_experiment_is_reported_without_reading_the_database(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.experiment_definitions.return_value = (_council_experiment(),)
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")

        code = main_tools.main(["evaluate", "--experiment", "EXP-99"])

        assert code == 1
        repo_cls.return_value.get_resolved_predictions_for_eval.assert_not_called()
        assert "EXP-99" in capsys.readouterr().err


def _discrimination_row(
    trend: str,
    correct: bool,
    confidence: float,
    *,
    day: int = 0,
    council: str | None = None,
    consensus: float | None = None,
    actual: float = 101.0,
) -> dict[str, Any]:
    row = _row(trend, correct, 100, 101, actual)
    row["timestamp"] = f"2026-09-{day + 1:02d}T05:30:00+00:00"
    row["confidence_score"] = confidence
    row["council_recommendation"] = council
    row["council_consensus"] = consensus
    return row


class TestDiscrimination:
    """WIA-38: czy pewność LLM i siła konsensusu rady odróżniają trafienia od pudeł."""

    def test_llm_and_council_auc_on_directional_calls(self) -> None:
        from src.tools.evaluate import summarize_discrimination

        rows = [
            _discrimination_row("BULLISH", True, 0.9, council="BUY", consensus=0.9, actual=103),
            _discrimination_row("BULLISH", False, 0.6, council="BUY", consensus=0.5, actual=98),
            _discrimination_row(
                "BEARISH", True, 0.8, day=1, council="SELL", consensus=0.4, actual=97
            ),
            _discrimination_row(
                "BEARISH", False, 0.7, day=1, council="SELL", consensus=0.8, actual=102
            ),
            # SIDEWAYS i HOLD nie stawiają zakładu na kierunek — poza AUC.
            _discrimination_row("SIDEWAYS", True, 0.95, day=1, council="HOLD", consensus=0.9),
        ]

        report = summarize_discrimination(rows)

        assert report.llm is not None and report.llm.n == 4
        assert report.llm.auc == pytest.approx(1.0)
        assert report.council is not None and report.council.n == 4
        assert report.council.auc == pytest.approx(0.5)

    def test_rows_without_confidence_or_council_are_left_out(self) -> None:
        from src.tools.evaluate import summarize_discrimination

        rows = [_discrimination_row("BULLISH", True, 0.9)]
        rows[0]["confidence_score"] = None

        report = summarize_discrimination(rows)

        assert report.llm is None
        assert report.council is None

    def test_formatted_section_prints_auc_n_eff_and_coverage(self) -> None:
        from src.tools.evaluate import format_discrimination, summarize_discrimination

        rows = [
            _discrimination_row("BULLISH", i % 2 == 0, 0.9 if i % 2 == 0 else 0.6, day=i)
            for i in range(10)
        ]

        text = format_discrimination(summarize_discrimination(rows))

        assert "Czy pewność coś znaczy" in text
        assert "AUC 1.00" in text
        assert "n=10" in text
        assert "trafność top 20%: 100.0%" in text
        assert "rada" in text and "za mało danych" in text

    def test_discrimination_flag_reads_confidence_and_council_fields(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        read = repo_cls.return_value.get_resolved_predictions_for_eval
        read.return_value = [_discrimination_row("BULLISH", True, 0.9)]

        code = main_tools.main(["evaluate", "--discrimination"])

        assert code == 0
        assert read.call_args.kwargs == {
            "days": 30,
            "extra_fields": ("confidence_score", "council_consensus", "council_recommendation"),
        }
        assert "Czy pewność coś znaczy" in capsys.readouterr().out


class TestRiskSignalSegmentation:
    """WIA-47: heurystyki RiskSignal z maila (DIVERGENCE, AV_LLM_CONFLICT,
    LOW_SIGNAL) sprawdzone na wynikach — trafność LLM z sygnałem i bez, z n,
    Wilsonem na n_eff i BH-FDR w rodzinie segmentów."""

    @staticmethod
    def _signals(
        day: int,
        *,
        correct: bool,
        delta: float | None = 0.0,
        sentiment: float | None = 0.0,
        agreement: float | None = 0.9,
        news: int | None = 10,
    ) -> dict[str, Any]:
        actual = 101 if correct else 99
        return {
            **_row("BULLISH", correct, 100, 100, actual),
            "timestamp": _day(day),
            "gate_delta": delta,
            "sentiment_score": sentiment,
            "av_llm_agreement": agreement,
            "news_volume_24h": news,
        }

    def test_segment_table_splits_by_risk_signal_presence(self) -> None:
        rows = [
            # AV↔LLM w konflikcie (zgoda < 0.3): oba chybione.
            self._signals(0, correct=False, agreement=0.1),
            self._signals(1, correct=False, agreement=0.2),
            # Rozbieżność cena↑ / sentyment↓ i mało newsów: trafione.
            self._signals(2, correct=True, delta=0.03, sentiment=-0.5, news=1),
            # Bez sygnału.
            self._signals(3, correct=True),
        ]

        report = summarize_evaluation(rows, segment_by="risk_signal")

        segments: dict[str, tuple[int, Any]] = {
            s.key: (s.n, s.rate) for s in report.segments["risk_signal"]
        }
        assert segments["AV_LLM_CONFLICT: z sygnałem"] == (2, pytest.approx(0.0))
        assert segments["AV_LLM_CONFLICT: bez"] == (2, pytest.approx(1.0))
        assert segments["DIVERGENCE: z sygnałem"] == (1, pytest.approx(1.0))
        assert segments["DIVERGENCE: bez"] == (3, pytest.approx(1 / 3))
        assert segments["LOW_SIGNAL: z sygnałem"] == (1, pytest.approx(1.0))
        assert segments["LOW_SIGNAL: bez"] == (3, pytest.approx(1 / 3))

    def test_missing_inputs_form_an_unknown_segment_not_a_negative(self) -> None:
        # Bez delty bramki (migracja 034) rozbieżności nie da się stwierdzić.
        rows = [self._signals(0, correct=True, delta=None, agreement=None, news=None)]

        report = summarize_evaluation(rows, segment_by="risk_signal")

        keys = {s.key: s.n for s in report.segments["risk_signal"]}
        assert keys["DIVERGENCE: brak danych"] == 1
        assert keys["AV_LLM_CONFLICT: brak danych"] == 1
        assert keys["LOW_SIGNAL: brak danych"] == 1
        assert keys["DIVERGENCE: z sygnałem"] == 0

    def test_council_agreement_splits_llm_hit_rate(self) -> None:
        rows = [
            {**self._signals(0, correct=True), "council_recommendation": "BUY"},
            {**self._signals(1, correct=False), "council_recommendation": "SELL"},
            {**self._signals(2, correct=False), "council_recommendation": "HOLD"},
            {**self._signals(3, correct=True), "council_recommendation": None},
        ]

        report = summarize_evaluation(rows, segment_by="council_agreement")

        segments: dict[str, tuple[int, Any]] = {
            s.key: (s.n, s.rate) for s in report.segments["council_agreement"]
        }
        assert segments["rada zgodna z LLM"] == (1, pytest.approx(1.0))
        assert segments["rada niezgodna z LLM"] == (2, pytest.approx(0.0))
        assert segments["bez rady"] == (1, pytest.approx(1.0))

    def test_cli_reads_the_signal_inputs_on_request(
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main_tools

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            self._signals(0, correct=True, agreement=0.1),
        ]

        main_tools.main(["evaluate", "--by", "risk_signal"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30,
            extra_fields=("av_llm_agreement", "gate_delta", "news_volume_24h", "sentiment_score"),
        )
        assert "AV_LLM_CONFLICT: z sygnałem" in capsys.readouterr().out
