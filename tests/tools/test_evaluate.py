from __future__ import annotations

import math

import pytest

from src.tools.evaluate import EvalReport, format_report, summarize_evaluation


def _row(
    trend: str,
    correct: bool | None,
    price_at: float,
    predicted: float,
    actual: float,
) -> dict:
    return {
        "predicted_trend": trend,
        "is_trend_correct": correct,
        "price_at_prediction": price_at,
        "predicted_target_price": predicted,
        "actual_price_after_12h": actual,
    }


class TestSummarizeEvaluation:
    def test_empty_rows_returns_zeroed_report(self):
        report = summarize_evaluation([])
        assert report.sample_count == 0
        assert report.hit_rate is None
        assert report.model_rmse is None
        assert report.baseline_rmse is None
        assert report.beats_baseline is False

    def test_directional_hit_rate(self):
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", False, 100, 95, 101),
            _row("BULLISH", True, 100, 102, 103),
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 3
        assert report.hit_rate == 2 / 3

    def test_model_rmse_and_baseline_rmse(self):
        # 2 rekordy. Model: |pred-actual| = 1 i 1 → RMSE = 1.
        # Baseline (last-value = price_at): |price_at-actual| = 5 i 5 → RMSE = 5.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            _row("BULLISH", True, 100, 106, 105),
        ]
        report = summarize_evaluation(rows)
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)
        assert math.isclose(report.baseline_rmse, 5.0, rel_tol=1e-9)
        assert report.beats_baseline is True

    def test_model_worse_than_baseline_does_not_beat(self):
        rows = [
            _row("BULLISH", True, 100, 120, 101),  # model bardzo daleko
            _row("BULLISH", True, 100, 80, 99),
        ]
        report = summarize_evaluation(rows)
        assert report.beats_baseline is False

    def test_skips_rows_with_missing_values_for_rmse(self):
        # Wiersz bez actual nie psuje RMSE (jest pomijany), ale wlicza się do
        # hit-rate jeśli ma is_trend_correct.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            {"predicted_trend": "BULLISH", "is_trend_correct": True,
             "price_at_prediction": 100, "predicted_target_price": None,
             "actual_price_after_12h": None},
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 2
        assert report.hit_rate == pytest.approx(1.0)
        # tylko 1 wiersz miał komplet cen → RMSE z jednego punktu
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)

    def test_report_is_dataclass(self):
        assert isinstance(summarize_evaluation([]), EvalReport)


class TestDirectionalHitRateSplit:
    """Finding #24: blended `hit_rate` zalicza SIDEWAYS jako trafienie zawsze, gdy
    ruch < ±0.5% — czyli „nic się nie ruszyło" myli się z „trafiłem kierunek".
    Rozdzielamy: directional-only hit-rate (BULLISH/BEARISH) + pokrycie SIDEWAYS."""

    def test_directional_hit_rate_distinct_from_blended(self):
        # Mieszanka: 2 kierunkowe (1 trafiona) + 2 SIDEWAYS (obie „trafione",
        # bo dni spokojne). Blended = 3/4. Directional-only = 1/2 — różne liczby.
        rows = [
            _row("BULLISH", True, 100, 105, 106),    # kierunkowa, trafiona
            _row("BEARISH", False, 100, 95, 101),    # kierunkowa, pudło
            _row("SIDEWAYS", True, 100, 100, 100.1),  # cisza, „trafiona"
            _row("SIDEWAYS", True, 100, 100, 99.9),   # cisza, „trafiona"
        ]
        report = summarize_evaluation(rows)

        assert report.hit_rate == 3 / 4               # blended (back-compat)
        assert report.directional_hit_rate == 1 / 2   # tylko BULLISH/BEARISH
        assert report.directional_count == 2
        assert report.sideways_count == 2

    def test_directional_hit_rate_none_when_no_directional_rows(self):
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

    def test_directional_count_ignores_rows_without_flag(self):
        # Wiersz bez is_trend_correct nie wlicza się do żadnego licznika trafień.
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", None, 100, 95, 101),
        ]
        report = summarize_evaluation(rows)

        assert report.directional_count == 1
        assert report.directional_hit_rate == pytest.approx(1.0)
        assert report.sideways_count == 0


class TestSignificanceVerdict:
    """WIA-22: ewaluacja offline mówi, czy trafność kierunkowa bije najlepszy
    naiwny baseline na tym samym zbiorze — nie tylko goły procent."""

    def test_report_carries_directional_evidence(self):
        rows = [
            {**_row("BULLISH", True, 100, 101, 101), "timestamp": "2026-10-01T14:00:00+00:00"},
            {**_row("BULLISH", False, 100, 101, 99), "timestamp": "2026-10-02T14:00:00+00:00"},
        ]

        report = summarize_evaluation(rows)

        assert report.evidence is not None
        assert report.evidence.hit_rate.directional == 2
        assert report.evidence.cycles == 2

    def test_formatted_report_states_the_verdict(self):
        rows = [
            {**_row("BULLISH", i % 2 == 0, 100, 101, 101 if i % 2 == 0 else 99),
             "timestamp": f"2026-10-{i + 1:02d}T14:00:00+00:00"}
            for i in range(6)
        ]

        text = format_report(summarize_evaluation(rows), 30)

        assert "nieodróżnialne od rzutu monetą" in text
        assert "zawsze-BULLISH: 50%" in text


class TestRecipeSegments:
    """WIA-20: ewaluacja segmentuje trafność po wersji przepisu — inaczej wynik
    miesza prompty, modele i persony, a zmiana przepisu udaje umiejętność."""

    def test_segments_directional_hit_rate_by_recipe_version(self):
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

    def test_rows_without_a_version_are_reported_as_unversioned(self):
        from src.domain.recipe import UNVERSIONED
        from src.tools.evaluate import summarize_by_version

        segments = summarize_by_version([_row("BULLISH", True, 100, 101, 101)], "recipe_version")

        assert list(segments) == [UNVERSIONED]

    def test_formatted_segments_show_short_versions_and_counts(self):
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

    def test_by_recipe_flag_reads_the_recipe_column(self, mocker, capsys):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 101, 101), "recipe_version": "f" * 64},
        ]

        evaluate.main(["--by-recipe"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, with_recipe=True
        )
        assert "f" * 12 in capsys.readouterr().out

    def test_default_run_does_not_touch_the_recipe_column(self, mocker):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = []

        evaluate.main([])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(days=30)


class TestUniverseSegments:
    """WIA-48: trafność per wersja uniwersum — efekt „dodałem symbole” oddzielony
    od zmiany trafności. Wersja jest w kwicie v2 (`decision_receipts`)."""

    def test_by_universe_flag_reads_the_receipt_universe(self, mocker, capsys):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 101, 101), "universe_version": "e" * 64},
            {**_row("BULLISH", False, 100, 101, 99)},
        ]

        evaluate.main(["--by-universe"])

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

    def _rows(self) -> list[dict]:
        # Trzy wzrosty z radą BUY: rada trafia 3/3, trend LLM tylko 1/3.
        # Trzy wiersze bez zakładu (LLM SIDEWAYS, rada HOLD albo pominięta).
        # XGBoost bez wag: cel = cena (baseline „cena bez zmian”) → brak zakładu.
        return [
            {**_row("BULLISH", True, 100, 100, 102), "council_recommendation": "BUY",
             "timestamp": _day(0)},
            {**_row("BEARISH", False, 100, 100, 102), "council_recommendation": "BUY",
             "timestamp": _day(1)},
            {**_row("BEARISH", False, 100, 100, 101), "council_recommendation": "BUY",
             "timestamp": _day(2)},
            {**_row("SIDEWAYS", True, 100, 100, 100.2), "council_recommendation": "HOLD",
             "timestamp": _day(3)},
            {**_row("SIDEWAYS", True, 100, 100, 99.9), "council_recommendation": "HOLD",
             "timestamp": _day(4)},
            {**_row("SIDEWAYS", False, 100, 100, 101), "council_recommendation": None,
             "timestamp": _day(5)},
        ]

    def test_component_segment_scores_council_and_llm_on_the_same_rows(self):
        report = summarize_evaluation(self._rows(), segment_by="component")

        segments = {s.key: s for s in report.segments["component"]}
        assert segments["council"].rate == pytest.approx(1.0)
        assert segments["council"].n == 3
        assert segments["llm"].rate == pytest.approx(1 / 3)
        assert segments["llm"].n == 3

    def test_model_without_weights_places_no_directional_bet(self):
        report = summarize_evaluation(self._rows(), segment_by="component")

        model = {s.key: s for s in report.segments["component"]}["xgboost"]
        assert model.n == 0
        assert model.rate is None

    def test_model_direction_comes_from_its_target_price(self):
        rows = [
            {**_row("BULLISH", True, 100, 103, 102), "timestamp": _day(0)},
            {**_row("BULLISH", False, 100, 97, 99), "timestamp": _day(1)},
            {**_row("BULLISH", True, 100, 98, 101), "timestamp": _day(2)},
        ]

        report = summarize_evaluation(rows, segment_by="component")

        model = {s.key: s for s in report.segments["component"]}["xgboost"]
        assert (model.n, model.rate) == (3, pytest.approx(2 / 3))

    def test_segments_below_ten_bets_are_flagged_as_insufficient(self):
        report = summarize_evaluation(self._rows(), segment_by="component")

        assert all(s.insufficient for s in report.segments["component"])

    def test_components_are_listed_in_a_fixed_order(self):
        report = summarize_evaluation(self._rows(), segment_by="component")

        assert [s.key for s in report.segments["component"]] == ["llm", "council", "xgboost"]

    def test_formatted_segments_print_n_and_the_insufficient_flag(self):
        from src.tools.evaluate import format_segment_evidence

        report = summarize_evaluation(self._rows(), segment_by="component")

        text = format_segment_evidence("component", report.segments["component"])

        assert "council" in text
        assert "100.0%" in text
        assert "n=3" in text
        assert "za mało danych" in text

    def test_default_report_has_no_segments(self):
        assert summarize_evaluation(self._rows()).segments == {}

    def test_unknown_segmentation_is_rejected(self):
        with pytest.raises(ValueError, match="segment"):
            summarize_evaluation(self._rows(), segment_by="persona")


class TestContextSegmentation:
    """WIA-55: segmenty kontekstu z istniejących kolumn — reżim (017), pewność
    (016), znak Alpha Fusion (022) i etykieta edge vs opcje (025)."""

    def test_regime_segments_correct_for_multiple_comparisons(self):
        # RISK_ON: 20 trafień na zmianę BULLISH/BEARISH w 20 dniach (baseline 50%)
        # → istotne. RISK_OFF: 10 zakładów, połowa trafiona → nieistotne.
        strong = [
            {**_row(trend, True, 100, 100, 101 if trend == "BULLISH" else 99),
             "regime": "RISK_ON", "timestamp": _day(i)}
            for i, trend in enumerate(["BULLISH", "BEARISH"] * 10)
        ]
        weak = [
            {**_row("BULLISH", i % 2 == 0, 100, 100, 101 if i % 2 == 0 else 99),
             "regime": "RISK_OFF", "timestamp": _day(i)}
            for i in range(10)
        ]

        report = summarize_evaluation(strong + weak, segment_by="regime")

        segments = {s.key: s for s in report.segments["regime"]}
        assert segments["RISK_ON"].fdr_significant is True
        assert segments["RISK_ON"].insufficient is False
        assert segments["RISK_OFF"].fdr_significant is False
        assert segments["RISK_OFF"].evidence.baselines.always_bullish == pytest.approx(0.5)

    def test_untagged_rows_form_the_unknown_regime(self):
        report = summarize_evaluation([_row("BULLISH", True, 100, 100, 101)], segment_by="regime")

        assert [s.key for s in report.segments["regime"]] == ["UNKNOWN"]

    def test_confidence_buckets_split_declared_confidence(self):
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

    def test_alpha_fusion_sign_and_edge_label_segments(self):
        rows = [
            {**_row("BULLISH", True, 100, 100, 101),
             "alpha_fusion_score": 0.4, "edge_sigma": 2.0},
            {**_row("BEARISH", True, 100, 100, 99),
             "alpha_fusion_score": -0.2, "edge_sigma": None},
        ]

        report = summarize_evaluation(rows, segment_by=("alpha_fusion_sign", "edge_label"))

        assert [s.key for s in report.segments["alpha_fusion_sign"]] == ["negative", "positive"]
        assert [s.key for s in report.segments["edge_label"]] == ["MODEL_AHEAD", "NO_SIGNAL"]


class TestSegmentationCli:
    """WIA-55: `--by` czyta kolumny segmentu wyłącznie na żądanie — domyślny
    przebieg nie dotyka żadnej z nich (każda istnieje dopiero po swojej migracji)."""

    def test_by_component_reads_the_council_recommendation(self, mocker, capsys):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 100, 101), "council_recommendation": "BUY"},
        ]

        evaluate.main(["--by", "component"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("council_recommendation",)
        )
        out = capsys.readouterr().out
        assert "council" in out
        assert "za mało danych" in out

    def test_by_asset_class_classifies_symbols_from_the_config(self, mocker, capsys):
        from src.tools import evaluate

        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.crypto_symbols = ["BTC"]
        settings.symbols_etf = ["VOO"]
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "BTC"},
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "VOO"},
            {**_row("BULLISH", True, 100, 100, 101), "symbol": "AAPL"},
        ]

        evaluate.main(["--by", "asset_class"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("symbol",)
        )
        out = capsys.readouterr().out
        assert "CRYPTO" in out
        assert "ETF" in out
        assert "STOCK" in out

    def test_several_segmentations_read_their_fields_once(self, mocker):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = []

        evaluate.main(["--by", "regime", "--by", "confidence_bucket", "--by", "regime"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("confidence_score", "regime")
        )


def _gated(trend: str, gate_delta: float | None, actual: float, day: int = 0) -> dict:
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


def test_echo_rate_counts_trend_matching_gate_delta_sign():
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


def test_momentum_baseline_hit_rate_uses_gate_delta_sign():
    # „Znak ruchu” trafia, gdy ruch po predykcji ma znak delty bramki;
    # „przeciwny znak” — gdy odwrotny. Te same wiersze co hit-rate LLM.
    from src.tools.evaluate import summarize_echo

    rows = [
        _gated("BULLISH", 0.030, 101, day=0),   # momentum ✓, LLM ✓ (echo)
        _gated("BULLISH", -0.030, 101, day=1),  # momentum ✗, LLM ✓ (odwrócenie)
        _gated("BEARISH", -0.025, 98, day=2),   # momentum ✓, LLM ✓ (echo)
        _gated("BEARISH", 0.040, 101, day=3),   # momentum ✓, LLM ✗ (odwrócenie)
    ]

    echo = summarize_echo(rows)

    assert echo.compared == 4
    assert echo.momentum_rate == pytest.approx(3 / 4)
    assert echo.contrarian_rate == pytest.approx(1 / 4)
    assert echo.llm_rate == pytest.approx(3 / 4)


class TestMomentumEcho:
    """EXP-01: test echa liczony WYŁĄCZNIE w `evaluate` (nie w recapie ani mailu)."""

    def test_llm_hit_rate_where_it_disagrees_with_momentum(self):
        from src.tools.evaluate import summarize_echo

        rows = [
            _gated("BULLISH", 0.030, 101, day=0),
            _gated("BULLISH", -0.030, 101, day=1),
            _gated("BEARISH", 0.040, 101, day=2),
        ]

        echo = summarize_echo(rows)

        assert echo.disagreement == 2
        assert echo.disagreement_rate == pytest.approx(1 / 2)

    def test_rows_without_gate_delta_or_directional_call_are_not_compared(self):
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

    def test_formatted_echo_prints_n_n_eff_and_bh(self):
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

    def test_echo_flag_reads_the_gate_delta_column(self, mocker, capsys):
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_resolved_predictions_for_eval.return_value = [
            _gated("BULLISH", 0.030, 101),
        ]

        evaluate.main(["--echo"])

        repo_cls.return_value.get_resolved_predictions_for_eval.assert_called_once_with(
            days=30, extra_fields=("gate_delta",)
        )
        assert "Echo momentum" in capsys.readouterr().out

    def test_missing_gate_delta_column_prints_the_migration_instead_of_failing(
        self, mocker, capsys
    ):
        from src.infrastructure.adapters.supabase_repo import MissingColumnError
        from src.tools import evaluate

        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        read = repo_cls.return_value.get_resolved_predictions_for_eval
        read.side_effect = [
            MissingColumnError("gate_delta"),
            [_row("BULLISH", True, 100, 101, 101)],
        ]

        code = evaluate.main(["--echo", "--by", "regime"])

        assert code == 0
        assert read.call_args_list[1].kwargs == {"days": 30, "extra_fields": ("regime",)}
        out = capsys.readouterr().out
        assert "kolumna gate_delta niedostępna — uruchom migrację 034" in out
        assert "Próbek (zamkniętych predykcji): 1" in out
        assert "Echo momentum" not in out
