from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort, RepositoryPort
from src.application.use_cases.train_model import (
    MIN_SAMPLES_FOR_TRAINING,
    POOLED_MODEL_SYMBOL,
    TrainModelUseCase,
)

EXPECTED_ML_FEATURES = [
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
]


@pytest.fixture
def ml_port() -> Mock:
    return Mock(spec=MLPredictionPort)


@pytest.fixture
def repository_port() -> Mock:
    return Mock(spec=RepositoryPort)


@pytest.fixture
def use_case(ml_port, repository_port) -> TrainModelUseCase:
    return TrainModelUseCase(ml_port=ml_port, db_port=repository_port)


def _feature_store_rows(n: int) -> list[dict]:
    """Generuje n rekordów jak z widoku ml_feature_store.

    Widok dostarcza GOTOWE `price_delta` (NaN-guarded) i `target_return` (zwrot
    12h) — train_model ich NIE przelicza, tylko selekcjonuje i robi dropna.
    """
    return [
        {
            "price_current": 100.0 + i * 0.5,
            "price_prev_12h": 99.0 + i * 0.5,
            "price_delta": (100.0 + i * 0.5 - (99.0 + i * 0.5)) / (99.0 + i * 0.5),
            "av_sentiment_score": -0.5 + (i % 10) / 10,
            "av_relevance_avg": 0.5 + (i % 5) / 10,
            "news_volume_24h": 1 + (i % 8),
            "high_relevance_count": i % 3,
            "llm_trend_signal": (i % 3) - 1,
            "av_llm_agreement": 0.4 + (i % 6) / 10,
            "target_return": 0.01 * ((i % 5) - 2),
        }
        for i in range(n)
    ]


class TestRun:
    def test_refresh_feature_store_delegates_to_repository(
        self, use_case, repository_port
    ):
        use_case.refresh_feature_store()

        repository_port.refresh_feature_store.assert_called_once()

    def test_returns_not_enough_data_when_below_minimum(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING - 1
        )

        result = use_case.run("AAPL")

        assert result["status"] == "skipped"
        assert "not enough" in result["reason"].lower() or "za mało" in result["reason"].lower()
        ml_port.train.assert_not_called()

    def test_returns_not_enough_data_when_empty(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        result = use_case.run("AAPL")

        assert result["status"] == "skipped"
        ml_port.train.assert_not_called()

    def test_can_skip_feature_store_refresh(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL", refresh_view=False)

        repository_port.refresh_feature_store.assert_not_called()
        ml_port.train.assert_not_called()

    def test_trains_model_with_correct_features_and_target(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING + 10
        )
        ml_port.train.return_value = {"status": "trained_successfully", "n_samples": 60}

        use_case.run("AAPL")

        ml_port.train.assert_called_once()
        features, target = ml_port.train.call_args.args
        assert isinstance(features, pd.DataFrame)
        assert list(features.columns) == EXPECTED_ML_FEATURES
        # Target = target_return (zwrot 12h z widoku), nie cena bezwzględna
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)

    def test_uses_view_price_delta_without_recompute(
        self, use_case, repository_port, ml_port
    ):
        # Widok dostarcza price_delta GOTOWE — train_model nie wolno go przeliczać
        # (stary recompute (current-prev)/prev dublował logikę i robił +inf przy
        # prev=0). Ustawiamy price_delta na wartość, której recompute NIGDY by nie
        # dał, i sprawdzamy, że trafia do modelu bez zmian.
        rows = _feature_store_rows(MIN_SAMPLES_FOR_TRAINING + 5)
        for r in rows:
            r["price_delta"] = 0.5
        repository_port.get_feature_store_data.return_value = rows
        ml_port.train.return_value = {"status": "trained_successfully"}

        use_case.run("AAPL")

        features, _ = ml_port.train.call_args.args
        # Wszystkie wiersze dostały price_delta=0.5 — porównanie tolerancyjne na float.
        assert np.allclose(features["price_delta"], 0.5)

    def test_drops_rows_when_view_price_delta_is_null(
        self, use_case, repository_port, ml_port
    ):
        # Widok emituje price_delta=NULL gdy price_prev_12h=0/NULL (guard anty-inf).
        # train_model NIE przelicza (stary recompute dawał +inf przeżywające
        # dropna) — wiersz z NULL price_delta musi zostać odrzucony, a do modelu
        # trafiają wyłącznie skończone wartości.
        rows = _feature_store_rows(MIN_SAMPLES_FOR_TRAINING + 5)
        rows[0]["price_delta"] = None  # type: ignore[assignment]
        repository_port.get_feature_store_data.return_value = rows
        ml_port.train.return_value = {"status": "trained_successfully"}

        use_case.run("AAPL")

        features, _ = ml_port.train.call_args.args
        assert len(features) == len(rows) - 1
        assert bool(np.isfinite(features["price_delta"]).all())

    def test_returns_ml_train_result(self, use_case, repository_port, ml_port):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING + 10
        )
        ml_port.train.return_value = {
            "status": "trained_successfully",
            "n_samples": 60,
            "model_path": "data/models/price_predictor.ubj",
        }

        result = use_case.run("AAPL")

        assert result["status"] == "trained_successfully"
        assert result["n_samples"] == 60


def test_skip_result_reports_n_samples(use_case, repository_port, ml_port):
    # ML-04: liczba próbek była tylko w tekście powodu ("3 < 50") — scorecard
    # i logi nie miały jej jako pola, więc brak wag dało się wyjaśnić tylko z logu GHA.
    repository_port.get_feature_store_data.return_value = _feature_store_rows(3)

    result = use_case.run("AAPL", refresh_view=False)

    assert result["status"] == "skipped"
    assert result["n_samples"] == 3
    ml_port.train.assert_not_called()


def test_skip_after_dropna_reports_valid_n_samples(use_case, repository_port, ml_port):
    rows = _feature_store_rows(MIN_SAMPLES_FOR_TRAINING + 2)
    for row in rows[:5]:
        row["price_delta"] = None
    repository_port.get_feature_store_data.return_value = rows

    result = use_case.run("AAPL", refresh_view=False)

    assert result["status"] == "skipped"
    assert result["n_samples"] == MIN_SAMPLES_FOR_TRAINING - 3
    ml_port.train.assert_not_called()


def _stamped_rows(symbol: str, n: int, start_hour: int, step_hours: int) -> list[dict]:
    """Wiersze widoku z `symbol` i `timestamp` (ISO jak z PostgREST). `target_return`
    koduje (symbol, kolejność), żeby test widział porządek wierszy w modelu."""
    rows = _feature_store_rows(n)
    base = pd.Timestamp("2026-09-01T00:00:00+00:00")
    for i, row in enumerate(rows):
        hour = start_hour + i * step_hours
        row["symbol"] = symbol
        row["timestamp"] = (base + pd.Timedelta(hours=hour)).isoformat()
        row["target_return"] = float(hour)
    return rows


def _rows_by_symbol(mapping: dict[str, list[dict]]):
    return lambda symbol: mapping[symbol]


def test_pooled_training_calls_train_once_with_rows_of_all_symbols(
    use_case, repository_port, ml_port
):
    # ML-13 (decyzja 4): JEDEN wspólny model. Dziś trener pomija wszystkie symbole
    # (max 17 próbek przy progu 50), choć łącznie jest ich kilkaset.
    repository_port.get_feature_store_data.side_effect = _rows_by_symbol(
        {
            "AAPL": _stamped_rows("AAPL", 30, start_hour=0, step_hours=2),
            "BTC": _stamped_rows("BTC", 30, start_hour=1, step_hours=2),
        }
    )
    ml_port.train.return_value = {"status": "trained_successfully", "n_samples": 60}

    result = use_case.run_pooled(["AAPL", "BTC", "AAPL"])

    ml_port.train.assert_called_once()
    features, target = ml_port.train.call_args.args
    assert list(features.columns) == EXPECTED_ML_FEATURES
    assert len(features) == len(target) == 60
    # Jeden zapis wag: model wspólny trenuje w trybie domyślnym (persist=True).
    assert ml_port.train.call_args.kwargs.get("persist", True) is True
    assert result["status"] == "trained_successfully"
    # Każdy symbol czytany raz, mimo duplikatu na wejściu.
    assert repository_port.get_feature_store_data.call_count == 2


def test_pooled_rows_follow_global_time_not_symbol_blocks(
    use_case, repository_port, ml_port
):
    # Konkatenacja per symbol dałaby AAPL(0..58h), potem BTC(1..59h) — fold
    # walidowałby AAPL na chwilach wcześniejszych niż trening na BTC.
    repository_port.get_feature_store_data.side_effect = _rows_by_symbol(
        {
            "AAPL": _stamped_rows("AAPL", 30, start_hour=0, step_hours=2),
            "BTC": _stamped_rows("BTC", 30, start_hour=1, step_hours=2),
        }
    )
    ml_port.train.return_value = {"status": "trained_successfully"}

    use_case.run_pooled(["AAPL", "BTC"])

    _, target = ml_port.train.call_args.args
    assert list(target) == [float(h) for h in range(60)]
    stamps = list(ml_port.train.call_args.kwargs["timestamps"])
    assert stamps == sorted(stamps)
    assert len(stamps) == 60


def test_pooled_training_skips_below_minimum_and_reports_n_samples(
    use_case, repository_port, ml_port
):
    repository_port.get_feature_store_data.side_effect = _rows_by_symbol(
        {
            "AAPL": _stamped_rows("AAPL", 10, start_hour=0, step_hours=1),
            "BTC": _stamped_rows("BTC", 10, start_hour=0, step_hours=1),
        }
    )

    result = use_case.run_pooled(["AAPL", "BTC"])

    assert result["status"] == "skipped"
    assert result["n_samples"] == 20
    ml_port.train.assert_not_called()


def test_per_symbol_run_is_evaluation_only(use_case, repository_port, ml_port):
    # Pętla per symbol zostaje do ewaluacji — zapis wag robi wyłącznie model wspólny,
    # więc kolejne symbole nie nadpisują już jednego pliku .ubj (last-writer-wins).
    repository_port.get_feature_store_data.return_value = _feature_store_rows(
        MIN_SAMPLES_FOR_TRAINING + 10
    )
    ml_port.train.return_value = {"status": "evaluated_passed_gate"}

    use_case.run("AAPL", refresh_view=False)

    assert ml_port.train.call_args.kwargs["persist"] is False


def test_pooled_scorecard_row_is_recorded_as_all(use_case, repository_port, ml_port):
    repository_port.get_feature_store_data.side_effect = _rows_by_symbol(
        {"AAPL": _stamped_rows("AAPL", 60, start_hour=0, step_hours=1)}
    )
    ml_port.train.return_value = {"status": "skipped_validation_failed", "n_samples": 60}

    result = use_case.run_pooled(["AAPL"], record_scorecard=True)

    repository_port.save_model_scorecard.assert_called_once_with(POOLED_MODEL_SYMBOL, result)
    assert POOLED_MODEL_SYMBOL == "ALL"


def test_pooled_scorecard_row_carries_the_run_id(ml_port, repository_port):
    # ML-24: jeden `run_id` na przebieg Slow Loopa — wiersz ALL i wiersze per symbol
    # grupuje on, a nie przerwa czasowa (re-run tego samego dnia).
    repository_port.get_feature_store_data.return_value = []
    use_case = TrainModelUseCase(ml_port=ml_port, db_port=repository_port, run_id="run-1")

    result = use_case.run_pooled(["AAPL"], record_scorecard=True)

    repository_port.save_model_scorecard.assert_called_once_with(
        POOLED_MODEL_SYMBOL, result, run_id="run-1"
    )


def test_pooled_skip_is_recorded_too(use_case, repository_port, ml_port):
    # Deadman czyta najnowszy wiersz scorecardu — przebieg bez modelu też jest śladem.
    repository_port.get_feature_store_data.return_value = []

    use_case.run_pooled(["AAPL"], record_scorecard=True)

    saved_symbol, saved = repository_port.save_model_scorecard.call_args.args
    assert saved_symbol == "ALL"
    assert saved["status"] == "skipped"


def test_pooled_scorecard_write_failure_does_not_fail_training(
    use_case, repository_port, ml_port
):
    repository_port.get_feature_store_data.return_value = []
    repository_port.save_model_scorecard.side_effect = Exception("no table")

    result = use_case.run_pooled(["AAPL"], record_scorecard=True)

    assert result["status"] == "skipped"


def test_pooled_scorecard_is_not_recorded_by_default(use_case, repository_port, ml_port):
    repository_port.get_feature_store_data.return_value = []

    use_case.run_pooled(["AAPL"])

    repository_port.save_model_scorecard.assert_not_called()


def test_per_symbol_run_purges_label_horizon_with_row_timestamps(
    use_case, repository_port, ml_port
):
    # Decyzja 4 (2026-10-18): purge horyzontu etykiety także w ewaluacji per symbol.
    # Bez znaczników czasu fold uczył się na wierszach, których zwrot 12h dojrzewa
    # dopiero w oknie walidacji.
    rows = _stamped_rows("AAPL", MIN_SAMPLES_FOR_TRAINING + 10, start_hour=0, step_hours=1)
    repository_port.get_feature_store_data.return_value = rows[::-1]
    ml_port.train.return_value = {"status": "evaluated_passed_gate"}

    use_case.run("AAPL", refresh_view=False)

    _, target = ml_port.train.call_args.args
    stamps = list(ml_port.train.call_args.kwargs["timestamps"])
    assert len(stamps) == len(target) == MIN_SAMPLES_FOR_TRAINING + 10
    assert stamps == sorted(stamps)
    # `target_return` koduje godzinę wiersza → cechy idą w tej samej kolejności co czas.
    assert list(target) == sorted(target)
    assert ml_port.train.call_args.kwargs["persist"] is False


def test_per_symbol_run_without_row_timestamps_trains_unpurged(
    use_case, repository_port, ml_port
):
    # Zgodność wstecz: wiersze bez kolumny `timestamp` ewaluują się jak dotąd.
    repository_port.get_feature_store_data.return_value = _feature_store_rows(
        MIN_SAMPLES_FOR_TRAINING + 10
    )
    ml_port.train.return_value = {"status": "evaluated_passed_gate"}

    use_case.run("AAPL", refresh_view=False)

    assert ml_port.train.call_args.kwargs.get("timestamps") is None


# -- DAT-09: trening tylko na bieżącej wersji zestawu cech -------------------


def test_without_feature_set_filter_training_reads_the_whole_history(
    use_case, repository_port, ml_port
):
    # Bez flagi (i bez migracji 027) odczyt nie może dotknąć kolumny wersji.
    repository_port.get_feature_store_data.return_value = []

    use_case.run_pooled(["AAPL"])
    use_case.run("AAPL", refresh_view=False)

    assert [c.args for c in repository_port.get_feature_store_data.call_args_list] == [
        ("AAPL",),
        ("AAPL",),
    ]
    assert all(not c.kwargs for c in repository_port.get_feature_store_data.call_args_list)


def test_feature_set_filter_reads_only_the_current_version(repository_port, ml_port):
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, feature_set_version=2
    )
    repository_port.get_feature_store_data.return_value = []

    use_case.run_pooled(["AAPL", "BTC"])
    use_case.run("AAPL", refresh_view=False)

    calls = repository_port.get_feature_store_data.call_args_list
    assert [c.args for c in calls] == [("AAPL",), ("BTC",), ("AAPL",)]
    assert all(c.kwargs == {"feature_set_version": 2} for c in calls)


def test_version_cut_below_the_minimum_skips_the_pooled_model_explicitly(
    repository_port, ml_port, caplog
):
    # Odcięcie historii po wersji zmniejsza próbkę — pominięcie ma mówić, ile
    # próbek zostało i dlaczego, a nie wyglądać jak zwykły brak danych.
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, feature_set_version=2
    )
    repository_port.get_feature_store_data.return_value = _stamped_rows(
        "AAPL", 12, start_hour=0, step_hours=1
    )

    with caplog.at_level("WARNING"):
        result = use_case.run_pooled(["AAPL"], record_scorecard=True)

    assert result["status"] == "skipped"
    assert result["n_samples"] == 12
    assert result["feature_set_version"] == 2
    assert "feature set version 2" in result["reason"]
    assert "feature set version 2 leaves 12 samples" in caplog.text
    ml_port.train.assert_not_called()
    _, saved = repository_port.save_model_scorecard.call_args.args
    assert saved["n_samples"] == 12


def test_pooled_training_logs_how_many_labels_settle_after_the_next_row(
    use_case, repository_port, ml_port, caplog
):
    # ML-25: widok z migracji 050 wystawia `settled_at`; model wspólny mierzy, ile
    # etykiet dojrzewa dopiero po następnym wierszu symbolu (purge 12 h ich nie łapie).
    rows = _stamped_rows("AAPL", MIN_SAMPLES_FOR_TRAINING, start_hour=0, step_hours=24)
    for i, row in enumerate(rows):
        hours = 30 if i % 5 == 0 else 20
        settled = pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=hours)
        row["settled_at"] = settled.isoformat()
    repository_port.get_feature_store_data.return_value = rows
    ml_port.train.return_value = {"status": "trained_successfully"}

    with caplog.at_level("INFO", logger="src.application.use_cases.train_model"):
        use_case.run_pooled(["AAPL"])

    assert "label overlap: 10 of 49 rows" in caplog.text


def test_pooled_training_without_settled_at_logs_no_overlap(
    use_case, repository_port, ml_port, caplog
):
    # Przed migracją 050 widok nie ma kolumny — odczyt `*` jest tolerancyjny, pomiaru brak.
    repository_port.get_feature_store_data.return_value = _stamped_rows(
        "AAPL", MIN_SAMPLES_FOR_TRAINING, start_hour=0, step_hours=24
    )
    ml_port.train.return_value = {"status": "trained_successfully"}

    with caplog.at_level("INFO", logger="src.application.use_cases.train_model"):
        use_case.run_pooled(["AAPL"])

    assert "label overlap" not in caplog.text
