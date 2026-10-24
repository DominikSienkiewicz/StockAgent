from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.application.ml_features import FREE_MODEL_FEATURE_COLUMNS, MARKET_FEATURE_SET_VERSION
from src.application.ports import MLPredictionPort, RepositoryPort, TrainingDataPort
from src.application.use_cases.train_model import (
    MIN_SAMPLES_FOR_TRAINING,
    POOLED_MODEL_SYMBOL,
    TrainingSet,
    TrainModelUseCase,
)
from src.domain.model_scorecard import (
    FREE_SNAPSHOT_MODEL_SYMBOL,
    LLM_SNAPSHOT_MODEL_SYMBOL,
    TargetUnit,
    TrainingOutcome,
    TrainingReport,
)
from src.domain.price_features import return_sigma
from src.domain.training_labels import FeatureSnapshot
from src.domain.training_provenance import DatasetDescription

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
        # ARCH-09: port dostaje wiersze cech w kolejności kontraktu, nie DataFrame.
        assert isinstance(features, list)
        assert all(list(row) == EXPECTED_ML_FEATURES for row in features)
        # Target = target_return (zwrot 12h z widoku), nie cena bezwzględna
        assert isinstance(target, list)
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
        assert np.allclose([row["price_delta"] for row in features], 0.5)

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
        assert bool(np.isfinite([row["price_delta"] for row in features]).all())

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
    assert list(features[0]) == EXPECTED_ML_FEATURES
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


# -- ML-52: trening z wektorów `feature_snapshots` z etykietą z `price_snapshots` --

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)
SERVED_VECTOR = {name: (i + 1) / 10 for i, name in enumerate(EXPECTED_ML_FEATURES)}


def _cycle_vector(i: int, *, gated: bool, flags: list[str] | None = None) -> dict:
    """Wektor cyklu jak z `_persist_feature_snapshot`: cechy darmowe zawsze, wektor
    modelu i dodatki tylko za bramką."""
    free = {
        "price": 100.0 + i,
        "previous_price": 99.0 + i,
        "snapshot_delta": 0.01 * (i % 3),
        "abs_snapshot_delta": 0.01 * (i % 3),
        "effective_threshold": 0.02,
        "regime": "NEUTRAL",
        "regime_multiplier": 1.0,
        "earnings_multiplier": 1.0,
        "alpha_fusion_score": None,
        "asset_type": "CRYPTO",
        "valuation_verdict": None,
        "ret_1": 0.001 * i,
    }
    if not gated:
        return free
    return {
        **free,
        **SERVED_VECTOR,
        "confidence_score": 0.6,
        "data_quality_flags": flags or [],
    }


def _cycles(
    symbol: str, n: int, *, gated: bool = True, flags: list[str] | None = None
) -> tuple[list[FeatureSnapshot], list[tuple[datetime, float]]]:
    """`n` dziennych cykli i ich snapshoty cen (zapisane tuż przed wektorem)."""
    snapshots = [
        FeatureSnapshot(
            symbol=symbol,
            as_of=T0 + timedelta(days=i),
            gated=gated,
            features=_cycle_vector(i, gated=gated, flags=flags),
            feature_set_version=MARKET_FEATURE_SET_VERSION,
        )
        for i in range(n)
    ]
    prices = [(T0 + timedelta(days=i, seconds=-1), 100.0 + i) for i in range(n + 1)]
    return snapshots, prices


@pytest.fixture
def training_data() -> Mock:
    return Mock(spec=TrainingDataPort)


@pytest.fixture
def snapshot_use_case(ml_port, repository_port, training_data) -> TrainModelUseCase:
    ml_port.train.return_value = TrainingReport(TrainingOutcome.EVALUATION_PASSED, n_samples=60)
    return TrainModelUseCase(ml_port=ml_port, db_port=repository_port, training_data=training_data)


def _train_call(ml_port: Mock, *, llm: bool):
    calls = [
        c for c in ml_port.train.call_args_list
        if ("llm_trend_signal" in c.args[0][0]) is llm
    ]
    assert len(calls) == 1
    return calls[0]


def test_snapshot_llm_model_trains_on_the_served_vector_with_price_snapshot_label(
    snapshot_use_case, ml_port, training_data
):
    # ML-52: trening czyta wektor zapisany przez cykl (ten, który dostał serving),
    # a etykieta to zwrot do pierwszego snapshotu ceny po 12 h — bez widoku SQL.
    snapshots, prices = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING + 1)
    training_data.get_feature_snapshots.return_value = snapshots
    training_data.get_price_series.return_value = {"BTC": prices}

    snapshot_use_case.run_snapshot_models(["BTC"])

    call = _train_call(ml_port, llm=True)
    rows, target = call.args
    assert rows[0] == SERVED_VECTOR
    assert list(rows[0]) == EXPECTED_ML_FEATURES
    assert len(rows) == len(target) == MIN_SAMPLES_FOR_TRAINING + 1
    assert target[0] == pytest.approx(101.0 / 100.0 - 1)
    assert call.kwargs["persist"] is False
    assert list(call.kwargs["timestamps"]) == [s.as_of for s in snapshots]


def test_free_feature_model_trains_on_every_labeled_cycle(
    snapshot_use_case, ml_port, training_data
):
    gated, gated_prices = _cycles("BTC", 30)
    quiet, quiet_prices = _cycles("ETH", 30, gated=False)
    training_data.get_feature_snapshots.return_value = gated + quiet
    training_data.get_price_series.return_value = {"BTC": gated_prices, "ETH": quiet_prices}

    result = snapshot_use_case.run_snapshot_models(["BTC", "ETH"])

    rows, target = _train_call(ml_port, llm=False).args
    assert len(rows) == len(target) == 60
    assert list(rows[0]) == list(FREE_MODEL_FEATURE_COLUMNS)
    # Bezwymiarowe cechy cyklu; brak źródła zostaje brakiem (None), nie zerem.
    assert rows[0]["is_crypto"] == 1.0
    assert rows[0]["alpha_fusion_score"] is None
    assert "price" not in rows[0]
    # Model LLM: 30 wierszy gated < minimum — uczciwe „za mało danych”.
    assert result.llm_model.outcome is TrainingOutcome.SKIPPED
    assert result.llm_model.n_samples == 30
    assert result.selection.gated.n == 30
    assert result.selection.non_gated.n == 30


def test_llm_model_drops_gated_cycles_with_failed_llm_or_feed(
    snapshot_use_case, ml_port, training_data
):
    clean, prices = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING)
    broken, broken_prices = _cycles("ETH", 10, flags=["llm_analysis_failed"])
    training_data.get_feature_snapshots.return_value = clean + broken
    training_data.get_price_series.return_value = {"BTC": prices, "ETH": broken_prices}

    snapshot_use_case.run_snapshot_models(["BTC", "ETH"])

    rows, _ = _train_call(ml_port, llm=True).args
    assert len(rows) == MIN_SAMPLES_FOR_TRAINING


def test_empty_feature_snapshots_report_not_enough_data(
    snapshot_use_case, ml_port, training_data
):
    # Baza produkcyjna jest pusta: obie karty mówią wprost, że danych za mało.
    training_data.get_feature_snapshots.return_value = []
    training_data.get_price_series.return_value = {}

    result = snapshot_use_case.run_snapshot_models(["BTC"])

    ml_port.train.assert_not_called()
    for report in (result.free_model, result.llm_model):
        assert report.outcome is TrainingOutcome.SKIPPED
        assert report.n_samples == 0
        assert "not enough" in (report.reason or "").lower()
    assert result.n_snapshots == result.n_labeled == 0


def test_snapshot_training_reads_feature_set_version_4_of_listed_symbols_only(
    snapshot_use_case, ml_port, training_data
):
    # Decyzja 2026-10-21: trening z wektorów w wersji zestawu cech 4.
    btc, prices = _cycles("BTC", 5)
    other, _ = _cycles("DOGE", 5)
    training_data.get_feature_snapshots.return_value = btc + other
    training_data.get_price_series.return_value = {"BTC": prices}

    result = snapshot_use_case.run_snapshot_models(["BTC"])

    training_data.get_feature_snapshots.assert_called_once_with(MARKET_FEATURE_SET_VERSION)
    assert training_data.get_price_series.call_args.args[0] == ["BTC"]
    assert result.n_snapshots == 5


def test_snapshot_models_are_recorded_as_two_cards(
    ml_port, repository_port, training_data
):
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, training_data=training_data, run_id="run-1"
    )
    training_data.get_feature_snapshots.return_value = []
    training_data.get_price_series.return_value = {}

    result = use_case.run_snapshot_models(["BTC"], record_scorecard=True)

    calls = repository_port.save_model_scorecard.call_args_list
    assert [c.args[0] for c in calls] == [FREE_SNAPSHOT_MODEL_SYMBOL, LLM_SNAPSHOT_MODEL_SYMBOL]
    assert [c.args[1] for c in calls] == [result.free_model, result.llm_model]
    assert all(c.kwargs == {"run_id": "run-1"} for c in calls)


# -- ML-52 (4): model wspólny z wektorów zamiast widoku SQL (flaga) ----------------


def test_snapshot_source_trains_the_shipped_model_on_gated_served_vectors(
    ml_port, repository_port, training_data
):
    gated, prices = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING + 1)
    quiet, quiet_prices = _cycles("ETH", 20, gated=False)
    training_data.get_feature_snapshots.return_value = gated + quiet
    training_data.get_price_series.return_value = {"BTC": prices, "ETH": quiet_prices}
    ml_port.train.return_value = TrainingReport(TrainingOutcome.ACCEPTED, n_samples=51)
    use_case = TrainModelUseCase(
        ml_port=ml_port,
        db_port=repository_port,
        training_data=training_data,
        snapshot_source=True,
    )

    result = use_case.run_pooled(["BTC", "ETH"])

    repository_port.get_feature_store_data.assert_not_called()
    rows, target = ml_port.train.call_args.args
    assert rows[0] == SERVED_VECTOR
    assert len(rows) == len(target) == MIN_SAMPLES_FOR_TRAINING + 1
    assert target[0] == pytest.approx(0.01)
    assert ml_port.train.call_args.kwargs.get("persist", True) is True
    assert result.outcome is TrainingOutcome.ACCEPTED


def test_snapshot_source_with_too_few_gated_vectors_skips_the_shipped_model(
    ml_port, repository_port, training_data
):
    gated, prices = _cycles("BTC", 10)
    training_data.get_feature_snapshots.return_value = gated
    training_data.get_price_series.return_value = {"BTC": prices}
    use_case = TrainModelUseCase(
        ml_port=ml_port,
        db_port=repository_port,
        training_data=training_data,
        snapshot_source=True,
    )

    result = use_case.run_pooled(["BTC"], record_scorecard=True)

    ml_port.train.assert_not_called()
    assert result.outcome is TrainingOutcome.SKIPPED
    assert result.n_samples == 10
    assert result["feature_set_version"] == MARKET_FEATURE_SET_VERSION
    repository_port.save_model_scorecard.assert_called_once_with(POOLED_MODEL_SYMBOL, result)


# -- ML-11: target = zwrot / σ (σ z 20 poprzednich cykli, as-of) — flaga ---------


def _zigzag_prices(start: datetime, n: int) -> list[tuple[datetime, float]]:
    """Dzienne snapshoty ceny z niezerową zmiennością cykl do cyklu."""
    return [
        (start + timedelta(days=i), 100.0 * (1.0 + (0.012 if i % 2 else -0.009)) ** i)
        for i in range(n)
    ]


def test_sigma_target_divides_the_pooled_target_by_the_as_of_sigma(
    ml_port, repository_port, training_data
):
    rows = _stamped_rows("AAPL", MIN_SAMPLES_FOR_TRAINING + 10, start_hour=0, step_hours=24)
    for i, row in enumerate(rows):
        row["target_return"] = 0.001 * (i + 1)
    repository_port.get_feature_store_data.return_value = rows
    series = _zigzag_prices(datetime(2026, 7, 1, tzinfo=UTC), 150)
    training_data.get_price_series.return_value = {"AAPL": series}
    ml_port.train.return_value = TrainingReport(TrainingOutcome.ACCEPTED, n_samples=60)
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, training_data=training_data, sigma_target=True
    )

    use_case.run_pooled(["AAPL"])

    _, target = ml_port.train.call_args.args
    first_stamp = pd.Timestamp(rows[0]["timestamp"]).to_pydatetime()
    sigma = return_sigma(series, first_stamp)
    assert sigma is not None
    assert list(target)[0] == pytest.approx(0.001 / sigma)
    assert ml_port.train.call_args.kwargs["target_unit"] is TargetUnit.SIGMA


def test_sigma_target_drops_rows_without_enough_history(
    ml_port, repository_port, training_data
):
    # Krótka historia daje niestabilną σ — wiersz wypada, a model wspólny uczciwie
    # mówi, ile próbek zostało.
    repository_port.get_feature_store_data.return_value = _stamped_rows(
        "AAPL", MIN_SAMPLES_FOR_TRAINING + 10, start_hour=0, step_hours=24
    )
    training_data.get_price_series.return_value = {}
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, training_data=training_data, sigma_target=True
    )

    result = use_case.run_pooled(["AAPL"])

    ml_port.train.assert_not_called()
    assert result.outcome is TrainingOutcome.SKIPPED
    assert result.n_samples == 0
    assert "sigma" in (result.reason or "")


def test_sigma_target_puts_the_snapshot_cards_in_sigma_units(
    ml_port, repository_port, training_data
):
    snapshots, _ = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING + 1)
    training_data.get_feature_snapshots.return_value = snapshots
    training_data.get_price_series.return_value = {
        "BTC": _zigzag_prices(T0 - timedelta(days=40), 120)
    }
    ml_port.train.return_value = TrainingReport(TrainingOutcome.EVALUATION_PASSED, n_samples=51)
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, training_data=training_data, sigma_target=True
    )

    use_case.run_snapshot_models(["BTC"])

    for call in ml_port.train.call_args_list:
        assert call.kwargs["target_unit"] is TargetUnit.SIGMA


def test_without_the_flag_training_passes_no_target_unit(
    snapshot_use_case, ml_port, training_data
):
    snapshots, prices = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING + 1)
    training_data.get_feature_snapshots.return_value = snapshots
    training_data.get_price_series.return_value = {"BTC": prices}

    snapshot_use_case.run_snapshot_models(["BTC"])

    assert all("target_unit" not in c.kwargs for c in ml_port.train.call_args_list)


# -- ML-22: opis próbki dla proweniencji treningu ---------------------------------


def test_pooled_training_describes_the_sample_to_the_model(ml_port, repository_port):
    # Karta i sidecar mają mówić, na jakich danych powstał model: ile wierszy
    # per symbol, która wersja zestawu cech i który commit kodu.
    use_case = TrainModelUseCase(
        ml_port=ml_port, db_port=repository_port, feature_set_version=2, git_sha="abc1234"
    )
    rows = {
        "AAPL": _stamped_rows("AAPL", 30, start_hour=0, step_hours=2),
        "BTC": _stamped_rows("BTC", 25, start_hour=1, step_hours=2),
    }
    repository_port.get_feature_store_data.side_effect = lambda symbol, **_: rows[symbol]
    ml_port.train.return_value = TrainingReport(TrainingOutcome.ACCEPTED, n_samples=55)

    use_case.run_pooled(["AAPL", "BTC"])

    assert ml_port.train.call_args.kwargs["dataset"] == DatasetDescription(
        n_per_symbol={"AAPL": 30, "BTC": 25}, feature_set_version=2, git_sha="abc1234"
    )


def test_pooled_training_set_is_exactly_what_the_model_trains_on(
    use_case, repository_port, ml_port
):
    # `reproduce-training` odbudowuje próbkę tą samą ścieżką co Slow Loop.
    repository_port.get_feature_store_data.side_effect = _rows_by_symbol(
        {
            "AAPL": _stamped_rows("AAPL", 30, start_hour=0, step_hours=2),
            "BTC": _stamped_rows("BTC", 30, start_hour=1, step_hours=2),
        }
    )
    ml_port.train.return_value = TrainingReport(TrainingOutcome.ACCEPTED, n_samples=60)

    training_set = use_case.pooled_training_set(["AAPL", "BTC"])
    use_case.run_pooled(["AAPL", "BTC"])

    assert isinstance(training_set, TrainingSet)
    features, target = ml_port.train.call_args.args
    assert training_set.features == features
    assert training_set.target == target
    assert training_set.timestamps == ml_port.train.call_args.kwargs["timestamps"]
    assert training_set.dataset == ml_port.train.call_args.kwargs["dataset"]


def test_pooled_training_set_below_the_minimum_is_the_skip_report(use_case, repository_port):
    repository_port.get_feature_store_data.return_value = _stamped_rows(
        "AAPL", 5, start_hour=0, step_hours=1
    )

    skipped = use_case.pooled_training_set(["AAPL"])

    assert isinstance(skipped, TrainingReport)
    assert skipped.outcome is TrainingOutcome.SKIPPED


def test_snapshot_source_describes_the_gated_sample(ml_port, repository_port, training_data):
    gated, prices = _cycles("BTC", MIN_SAMPLES_FOR_TRAINING + 1)
    training_data.get_feature_snapshots.return_value = gated
    training_data.get_price_series.return_value = {"BTC": prices}
    ml_port.train.return_value = TrainingReport(TrainingOutcome.ACCEPTED, n_samples=51)
    use_case = TrainModelUseCase(
        ml_port=ml_port,
        db_port=repository_port,
        training_data=training_data,
        snapshot_source=True,
        git_sha="abc1234",
    )

    use_case.run_pooled(["BTC"])

    assert ml_port.train.call_args.kwargs["dataset"] == DatasetDescription(
        n_per_symbol={"BTC": MIN_SAMPLES_FOR_TRAINING + 1},
        feature_set_version=MARKET_FEATURE_SET_VERSION,
        git_sha="abc1234",
    )


def test_snapshot_model_cards_describe_their_own_samples(
    snapshot_use_case, ml_port, training_data
):
    gated, gated_prices = _cycles("BTC", 30)
    quiet, quiet_prices = _cycles("ETH", 30, gated=False)
    training_data.get_feature_snapshots.return_value = gated + quiet
    training_data.get_price_series.return_value = {"BTC": gated_prices, "ETH": quiet_prices}

    snapshot_use_case.run_snapshot_models(["BTC", "ETH"])

    dataset = _train_call(ml_port, llm=False).kwargs["dataset"]
    assert dataset.n_per_symbol == {"BTC": 30, "ETH": 30}
    assert dataset.feature_set_version == MARKET_FEATURE_SET_VERSION


def test_per_symbol_evaluation_describes_its_symbol(use_case, repository_port, ml_port):
    repository_port.get_feature_store_data.return_value = _feature_store_rows(
        MIN_SAMPLES_FOR_TRAINING + 10
    )
    ml_port.train.return_value = {"status": "evaluated_passed_gate"}

    use_case.run("AAPL", refresh_view=False)

    assert ml_port.train.call_args.kwargs["dataset"] == DatasetDescription(
        n_per_symbol={"AAPL": MIN_SAMPLES_FOR_TRAINING + 10}
    )
