from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort, ReturnScaleUnavailableError
from src.domain.model_scorecard import TargetUnit, TrainingOutcome, TrainingReport
from src.domain.value_objects import Money
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)

FEATURE_NAMES = [
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
]


def _synthetic_dataset(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Generuje syntetyczny dataset, gdzie target to ZWROT 12h (nie cena bezwzgl.).

    y ≈ 0.4*price_delta + 0.06*sentiment + 0.02*trend + 0.03*agreement — małe
    wartości wokół 0 (typowy rozkład zwrotów). Sygnał na tyle silny, że płytki
    XGBoost go odtworzy i pobije baseline persystencji (predykcja: zwrot = 0).
    """
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "price_delta": rng.normal(0, 0.05, n),
            "av_sentiment_score": rng.uniform(-1, 1, n),
            "av_relevance_avg": rng.uniform(0, 1, n),
            "news_volume_24h": rng.integers(0, 12, n).astype(float),
            "high_relevance_count": rng.integers(0, 5, n).astype(float),
            "llm_trend_signal": rng.choice([-1, 0, 1], size=n).astype(float),
            "av_llm_agreement": rng.uniform(0, 1, n),
        }
    )
    target = pd.Series(
        features["price_delta"] * 0.4
        + features["av_sentiment_score"] * 0.06
        + features["llm_trend_signal"] * 0.02
        + features["av_llm_agreement"] * 0.03
    )
    return features, target


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    return str(tmp_path / "price_predictor.ubj")


@pytest.fixture
def adapter(model_path: str) -> XGBoostAdapter:
    return XGBoostAdapter(model_path=model_path)


@pytest.fixture
def trained_adapter(adapter: XGBoostAdapter) -> XGBoostAdapter:
    features, target = _synthetic_dataset()
    adapter.train(features, target)
    return adapter


class TestTrain:
    def test_returns_success_status(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)
        assert result["status"] == "trained_successfully"

    def test_includes_sample_count_in_status(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=50)
        result = adapter.train(features, target)
        assert result["n_samples"] == 50

    def test_saves_model_to_disk(self, adapter: XGBoostAdapter, model_path: str) -> None:
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0

    def test_returns_validation_metrics(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset()
        result: Mapping[str, Any] = adapter.train(features, target)

        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]

    def test_metric_keys_are_candidate_scoped_and_self_describing(
        self, adapter: XGBoostAdapter
    ) -> None:
        # Finding #22: shipowany model jest refitowany na WSZYSTKICH danych i nie
        # jest re-walidowany, więc raportowane RMSE/hit-rate dotyczą KANDYDATA na
        # holdoucie — nazwy kluczy muszą to mówić wprost, plus `metric_note`.
        features, target = _synthetic_dataset()
        result: Mapping[str, Any] = adapter.train(features, target)

        assert "candidate_holdout_rmse" in result
        assert "candidate_holdout_directional_hit_rate" in result
        # Stare, mylące nazwy nie mogą udawać jakości shipowanego modelu.
        assert "validation_rmse" not in result
        assert "validation_directional_hit_rate" not in result
        assert "refit on all data" in result["metric_note"]
        # Niezmienniki kontraktu zachowane.
        assert result["n_samples"] == len(target)
        assert result["model_path"] == adapter._model_path

    def test_skips_save_when_model_does_not_beat_baseline(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        # Zwrot zawsze dokładnie 0 (cena się nie rusza) → baseline persystencji
        # (predykcja: zwrot=0) jest bezbłędny, żaden model go nie pobije.
        features, _ = _synthetic_dataset(n=80)
        target = pd.Series([0.0] * 80)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert not Path(model_path).exists()
        assert adapter.is_trained is False

    def test_baseline_rmse_is_zero_return_persistence(
        self, adapter: XGBoostAdapter
    ) -> None:
        # Baseline = "predykcja zwrotu = 0" (persystencja: cena bez zmian).
        # baseline_rmse musi być RMSE zwrotów względem 0 (a NIE rozrzutem
        # pojedynczego skalara z train — stary, błędny baseline). Przy małej
        # próbce (n=15 < MIN_FOLD_SAMPLES) adapter spada do pojedynczego splitu,
        # więc baseline jest liczony na holdoucie tego jednego splitu.
        n = 15
        features, target = _synthetic_dataset(n=n)
        validation_size = max(10, int(n * 0.2))
        valid_target = target.iloc[n - validation_size:]
        expected_baseline = float(np.sqrt(np.mean(valid_target.to_numpy() ** 2)))

        result = adapter.train(features, target)

        assert result["n_folds"] == 1
        assert result["baseline_rmse"] == pytest.approx(expected_baseline, rel=1e-9)

    def test_reports_candidate_holdout_directional_hit_rate(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset()
        result: Mapping[str, Any] = adapter.train(features, target)

        hit_rate = cast(float, result["candidate_holdout_directional_hit_rate"])
        assert 0.0 <= hit_rate <= 1.0
        # Sygnał jest realny → kierunek trafiony częściej niż losowo.
        assert hit_rate > 0.5


class TestWalkForwardValidation:
    """Finding #23: ewaluacja walk-forward (expanding window) zamiast jednego,
    hałaśliwego splitu 20%. Decyzja o shipowaniu zależy od ŚREDNIEJ z foldów +
    raport niesie per-feature drift stats i `n_folds`."""

    def test_strong_signal_ships_across_folds(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)
        result: Mapping[str, Any] = adapter.train(features, target)

        assert result["status"] == "trained_successfully"
        # Wystarczająco duża próbka → realne foldy walk-forward.
        assert result["n_folds"] >= 3
        # Bramka oparta na ŚREDNIEJ z foldów: kandydat bije baseline średnio.
        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]

    def test_no_signal_dataset_is_skipped(self, adapter: XGBoostAdapter) -> None:
        # Czysty szum bez sygnału — model nie pobije baseline persystencji
        # średnio po foldach → skip, model nie ląduje na dysku.
        rng = np.random.default_rng(7)
        n = 200
        features = pd.DataFrame(
            {name: rng.normal(0, 1, n) for name in FEATURE_NAMES}
        )
        target = pd.Series(rng.normal(0, 0.05, n))

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert adapter.is_trained is False

    def test_zero_return_dataset_is_skipped(self, adapter: XGBoostAdapter) -> None:
        # Zwrot zawsze 0 → baseline persystencji bezbłędny, nic go nie pobije.
        features, _ = _synthetic_dataset(n=200)
        target = pd.Series([0.0] * 200)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert adapter.is_trained is False

    def test_result_reports_n_folds_above_one_for_large_sample(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)
        result: Mapping[str, Any] = adapter.train(features, target)
        assert result["n_folds"] > 1

    def test_small_sample_falls_back_to_single_split(
        self, adapter: XGBoostAdapter
    ) -> None:
        # Poniżej progu na walk-forward → pojedynczy split (zachowanie sane).
        features, target = _synthetic_dataset(n=15)
        result = adapter.train(features, target)
        assert result["n_folds"] == 1

    def test_result_includes_per_feature_distribution_stats(
        self, adapter: XGBoostAdapter
    ) -> None:
        # Drift-detection: per-feature mean/std w wyniku, żeby porównać z
        # poprzednim runem i wykryć dryf rozkładu cech wejściowych.
        features, target = _synthetic_dataset(n=120)
        result: Mapping[str, Any] = adapter.train(features, target)

        stats = cast(dict[str, Any], result["feature_stats"])
        assert set(stats.keys()) == set(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert "mean" in stats[name]
            assert "std" in stats[name]
            expected_mean = float(features[name].mean())
            assert stats[name]["mean"] == pytest.approx(expected_mean, rel=1e-9)

    def test_feature_stats_present_even_when_skipped(
        self, adapter: XGBoostAdapter
    ) -> None:
        # Dryf cech jest obserwowalny niezależnie od werdyktu bramki.
        features, _ = _synthetic_dataset(n=200)
        target = pd.Series([0.0] * 200)
        result: Mapping[str, Any] = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert set(result["feature_stats"].keys()) == set(FEATURE_NAMES)


class TestPredict:
    def test_returns_money_instance(self, trained_adapter: XGBoostAdapter) -> None:
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.02,
            "av_sentiment_score": 0.5,
            "av_relevance_avg": 0.75,
            "news_volume_24h": 4.0,
            "high_relevance_count": 2.0,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.8,
        })
        prediction = trained_adapter.predict(features, current_price=Decimal("100"))
        assert isinstance(prediction, Money)

    def test_prediction_is_finite_positive_for_typical_input(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "av_relevance_avg": 0.5,
            "news_volume_24h": 3.0,
            "av_llm_agreement": 0.5,
        })
        prediction = trained_adapter.predict(features, current_price=Decimal("100"))
        # Model przewiduje mały ZWROT → cena odtworzona wokół current_price.
        assert Decimal("80") < prediction.amount < Decimal("130")

    def test_reconstructs_absolute_price_multiplicatively_from_return(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Model zwraca ZWROT; predict rekonstruuje cenę = current_price*(1+zwrot).
        # Ten sam zwrot na innym poziomie ceny → ta sama RELATYWNA zmiana.
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.03,
            "av_sentiment_score": 0.8,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.9,
        })
        p100 = trained_adapter.predict(features, current_price=Decimal("100"))
        p200 = trained_adapter.predict(features, current_price=Decimal("200"))

        r100 = (p100.amount - Decimal("100")) / Decimal("100")
        r200 = (p200.amount - Decimal("200")) / Decimal("200")
        assert r100 == pytest.approx(r200, rel=1e-9)
        # Sygnał byczy → niezerowy dodatni zwrot (cena ≠ current_price).
        assert p100.amount != Decimal("100")

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter) -> None:
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        current_price = Decimal("100")
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.predict(features, current_price=current_price)


class TestSigmaTarget:
    """ML-11: model wspólny uczony na zwrot / σ. Plik wag niesie jednostkę targetu,
    a `predict` mnoży prognozę z powrotem przez σ symbolu z chwili cyklu."""

    @staticmethod
    def _sigma_adapter(model_path: str) -> XGBoostAdapter:
        features, target = _synthetic_dataset(n=200)
        XGBoostAdapter(model_path=model_path).train(
            features, target / 0.02, target_unit=TargetUnit.SIGMA
        )
        return XGBoostAdapter(model_path=model_path)

    def test_predict_rescales_return_by_sigma(self, model_path: str) -> None:
        adapter = self._sigma_adapter(model_path)
        features, _ = _synthetic_dataset(n=1)
        row = features.iloc[0].to_dict()
        price = Decimal("100")

        low = adapter.predict(row, current_price=price, return_scale=0.01).amount - price
        high = adapter.predict(row, current_price=price, return_scale=0.03).amount - price

        assert low != 0
        assert float(high) == pytest.approx(3 * float(low), rel=1e-6)

    def test_sigma_model_without_sigma_refuses_to_predict(self, model_path: str) -> None:
        # Z-score podany jako zwrot byłby cichą pomyłką skali — serving wraca wtedy
        # do baseline'u, zamiast serwować liczbę w złych jednostkach.
        adapter = self._sigma_adapter(model_path)
        features, _ = _synthetic_dataset(n=1)

        with pytest.raises(ReturnScaleUnavailableError):
            adapter.predict(features.iloc[0].to_dict(), current_price=Decimal("100"))
        with pytest.raises(ReturnScaleUnavailableError):
            adapter.explain(features.iloc[0].to_dict())

    def test_explain_is_rescaled_with_the_prediction(self, model_path: str) -> None:
        adapter = self._sigma_adapter(model_path)
        features, _ = _synthetic_dataset(n=1)
        row = features.iloc[0].to_dict()

        low = adapter.explain(row, return_scale=0.01)
        high = adapter.explain(row, return_scale=0.02)

        assert high["price_delta"] == pytest.approx(2 * low["price_delta"])
        assert high["_bias"] == pytest.approx(2 * low["_bias"])

    def test_return_model_ignores_the_sigma(self, trained_adapter: XGBoostAdapter) -> None:
        features, _ = _synthetic_dataset(n=1)
        row = features.iloc[0].to_dict()
        price = Decimal("100")

        assert trained_adapter.predict(row, current_price=price, return_scale=0.05) == (
            trained_adapter.predict(row, current_price=price)
        )

    def test_report_names_the_sigma_unit_only_for_sigma_targets(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        sigma = adapter.train(features, target / 0.02, target_unit=TargetUnit.SIGMA)
        plain = adapter.train(features, target)

        assert sigma["target_unit"] == "sigma"
        assert "target_unit" not in plain


class TestPredictFeatureAlignment:
    """predict() wyrównuje wejście do cech, na których model był NAPRAWDĘ
    trenowany (`feature_names_in_`). Bez tego ewolucja kontraktu cech (dodanie
    nowej cechy do ML_FEATURE_COLUMNS przy starym .ubj w repo) cicho psuła
    predykcję mismatchem nazw kolumn."""

    def test_ignores_unknown_extra_features_and_scrambled_order(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Model zna 7 cech. Wejście dorzuca nieznaną cechę + miesza kolejność.
        features: dict[str, float] = {"holding_hours": 24.0}
        for name in reversed(FEATURE_NAMES):
            features[name] = 0.5
        prediction = trained_adapter.predict(features, current_price=Decimal("100"))
        assert isinstance(prediction, Money)

    def test_raises_keyerror_when_required_feature_missing(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Brak wymaganej cechy → jawny błąd, nie ciche NaN wpychane do modelu.
        features = dict.fromkeys(FEATURE_NAMES[:-1], 0.5)
        current_price = Decimal("100")
        with pytest.raises(KeyError):
            trained_adapter.predict(features, current_price=current_price)


class TestExplain:
    """Q3 — feature attribution (SHAP-lite). `explain()` zwraca znakowane wkłady
    per-cecha z boostera (pred_contribs), żeby raport pokazał, co pchnęło
    prognozę w górę/dół. Lokalny CPU, zero nowych płatnych wywołań."""

    def _typical_features(self) -> dict[str, float]:
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.02,
            "av_sentiment_score": 0.5,
            "av_relevance_avg": 0.75,
            "news_volume_24h": 4.0,
            "high_relevance_count": 2.0,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.8,
        })
        return features

    def test_returns_contribution_per_known_feature(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        contribs = trained_adapter.explain(self._typical_features())
        # Klucz per cecha, na której model był trenowany (bias pod osobnym kluczem).
        assert set(FEATURE_NAMES).issubset(set(contribs.keys()))

    def test_contributions_are_finite_floats(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        contribs = trained_adapter.explain(self._typical_features())
        for value in contribs.values():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_sum_of_contributions_plus_bias_approximates_raw_prediction(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Niezmiennik addytywności SHAP: suma wkładów + bias ≈ surowy zwrot modelu.
        features = self._typical_features()
        contribs = trained_adapter.explain(features)
        raw_return = float(
            trained_adapter._model.predict(  # type: ignore[union-attr]
                trained_adapter._align_features(features)
            )[0]
        )
        total = sum(contribs.values())
        assert total == pytest.approx(raw_return, abs=1e-4)

    def test_exposes_bias_under_dedicated_key(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Bias/base value jest osobno (nie jako cecha) — czytelny dla raportu.
        contribs = trained_adapter.explain(self._typical_features())
        assert "_bias" in contribs
        assert "_bias" not in FEATURE_NAMES

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter) -> None:
        features = self._typical_features()
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.explain(features)

    def test_ignores_unknown_extra_features(
        self, trained_adapter: XGBoostAdapter
    ) -> None:
        # Tak jak predict(): nieznana cecha jest ignorowana (wyrównanie do modelu).
        features = self._typical_features()
        features["holding_hours"] = 24.0
        contribs = trained_adapter.explain(features)
        assert "holding_hours" not in contribs


class TestModelPersistence:
    def test_loads_existing_model_from_disk_on_init(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ) -> None:
        # Pierwszy adapter trenuje i zapisuje. Drugi czyta z dysku.
        reloaded = XGBoostAdapter(model_path=model_path)
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.02,
            "av_sentiment_score": 0.5,
            "av_relevance_avg": 0.75,
            "news_volume_24h": 4.0,
            "high_relevance_count": 2.0,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.8,
        })

        original = trained_adapter.predict(features, current_price=Decimal("100"))
        from_disk = reloaded.predict(features, current_price=Decimal("100"))

        assert original.amount == from_disk.amount

    def test_cold_start_does_not_raise(self, tmp_path: Path) -> None:
        # Brak pliku — konstruktor musi przeżyć (czekamy na pierwszy trening).
        adapter = XGBoostAdapter(model_path=str(tmp_path / "missing.ubj"))
        # Cold start: model nie jest jeszcze wczytany, adapter to wie.
        assert adapter.is_trained is False


class TestIsTrained:
    def test_false_on_cold_start(self, adapter: XGBoostAdapter) -> None:
        # Brak pliku modelu → is_trained False (Fast Loop użyje baseline)
        assert adapter.is_trained is False

    def test_true_after_training(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert adapter.is_trained is True

    def test_true_when_model_loaded_from_disk(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ) -> None:
        reloaded = XGBoostAdapter(model_path=model_path)
        assert reloaded.is_trained is True


class TestAdapterImplementsPort:
    def test_is_ml_prediction_port(self) -> None:
        assert issubclass(XGBoostAdapter, MLPredictionPort)


class TestDomainGate:
    """ML-05/ML-06: bramka „bije baseline" żyje w domenie (`src.domain.backtest`),
    a adapter tylko ją stosuje przed zapisem wag."""

    def test_noise_level_improvement_is_not_shipped(
        self, adapter: XGBoostAdapter, model_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0.0200 vs 0.0201 = 0.5% poprawy. Stara reguła (próg 1e-9) shipowała szum.
        folds = [
            {"candidate_rmse": 0.0200, "baseline_rmse": 0.0201, "hit_rate": 0.5,
             "n_valid": 30.0}
            for _ in range(4)
        ]
        monkeypatch.setattr(adapter, "_walk_forward_folds_detail", lambda *a, **k: folds)
        features, target = _synthetic_dataset(n=200)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert not Path(model_path).exists()

    def test_too_few_oos_points_are_not_shipped(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        # n=60 → 4 foldy po 12 punktów = 48 < MIN_OOS_POINTS mimo silnego sygnału.
        features, target = _synthetic_dataset(n=60)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert result["n_oos_points"] == 48
        assert not Path(model_path).exists()
        assert adapter.is_trained is False


def _hourly_stamps(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2026-09-01", periods=n, freq="h", tz="UTC"))


class TestTypedTrainingReport:
    """ARCH-09: adapter zwraca domenowy `TrainingReport` z typowanym werdyktem."""

    def test_shipped_model_is_an_accepted_report_with_fold_metrics(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        report = adapter.train(features, target)

        assert isinstance(report, TrainingReport)
        assert report.outcome is TrainingOutcome.ACCEPTED
        assert report.n_samples == 200
        assert report.n_folds is not None and report.n_folds > 1
        assert report.rmse is not None and report.baseline_rmse is not None
        assert report.rmse < report.baseline_rmse
        assert report.model_path == model_path

    def test_evaluation_reports_carry_evaluation_outcomes(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)

        passed = adapter.train(features, target, persist=False)
        failed = adapter.train(features, pd.Series([0.0] * 200), persist=False)

        assert passed.outcome is TrainingOutcome.EVALUATION_PASSED
        assert failed.outcome is TrainingOutcome.EVALUATION_FAILED
        assert passed.model_path is None


class TestFeatureRowsWithGaps:
    """ML-52: wiersze modelu cech darmowych mają braki (None) — także całe kolumny,
    np. cechy makro przy wyłączonym FRED. Brak ma być brakiem XGBoost, nie błędem."""

    def test_rows_with_missing_values_train_as_xgboost_missing(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)
        rows = [
            {**row, "macro_off": None, "sometimes": None if i % 2 else float(i)}
            for i, row in enumerate(features.to_dict("records"))
        ]

        report = adapter.train(rows, list(target), persist=False)

        assert report.outcome is TrainingOutcome.EVALUATION_PASSED
        assert report.feature_stats is not None
        assert set(report.feature_stats) == {*FEATURE_NAMES, "macro_off", "sometimes"}


class TestPooledTrainingSupport:
    """ML-13: per-symbol pętla tylko ewaluuje (bez zapisu), a model wspólny trenuje
    na foldach po czasie GLOBALNYM z purgem horyzontu etykiety."""

    def test_evaluation_mode_never_writes_weights(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        result: Mapping[str, Any] = adapter.train(features, target, persist=False)

        assert result["status"] == "evaluated_passed_gate"
        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]
        assert not Path(model_path).exists()
        assert adapter.is_trained is False

    def test_evaluation_mode_reports_a_failed_gate_without_rejecting_a_model(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, _ = _synthetic_dataset(n=200)

        result = adapter.train(features, pd.Series([0.0] * 200), persist=False)

        # Status ewaluacji, nie „odrzucony kandydat" — karta nie może sugerować,
        # że dla symbolu został jakiś osobny, sprawdzony model.
        assert result["status"] == "evaluated_failed_gate"

    def test_purged_folds_drop_training_rows_inside_the_label_horizon(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter._walk_forward_folds_detail(
            features, target, timestamps=_hourly_stamps(200)
        )

        # Godzinne wiersze, horyzont 12h → 11 ostatnich wierszy przed walidacją
        # ma etykietę po jej starcie i wypada z treningu foldu.
        assert [f["n_train"] for f in folds] == [f["train_end"] - 11 for f in folds]

    def test_timestamps_must_follow_global_time_order(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)
        stamps = _hourly_stamps(200).iloc[::-1].reset_index(drop=True)

        with pytest.raises(ValueError, match="time"):
            adapter.train(features, target, timestamps=stamps)

    def test_pooled_training_with_timestamps_ships_once(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        result = adapter.train(features, target, timestamps=_hourly_stamps(200))

        assert result["status"] == "trained_successfully"
        assert Path(model_path).exists()


class TestSignalSkillMetrics:
    """ML-20: obok RMSE fold niesie IC (korelacja rang Spearmana prognozy z
    realizacją), zwrot strategii znakowej i turnover."""

    def test_fold_detail_contains_ic(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target)

        for fold in folds:
            assert {"ic", "sign_strategy_return", "turnover"} <= fold.keys()
            assert -1.0 <= fold["ic"] <= 1.0
            assert 0.0 <= fold["turnover"] <= 1.0
        # Sygnał jest realny → porządek prognoz zgodny z realizacją.
        assert all(fold["ic"] > 0.3 for fold in folds)
        assert all(fold["sign_strategy_return"] > 0 for fold in folds)

    def test_constant_target_has_zero_ic_instead_of_nan(self, adapter: XGBoostAdapter) -> None:
        # Stała seria nie ma porządku — korelacja rang jest niezdefiniowana (NaN),
        # a NaN zatrułby średnią IC w podsumowaniu.
        features, _ = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, pd.Series([0.0] * 200))

        assert all(fold["ic"] == 0.0 for fold in folds)


class TestWalkForwardReportPurge:
    """Decyzja 4 (2026-10-18): offline backtest purguje horyzont etykiety tak jak
    bramka `train()` modelu wspólnego."""

    def test_report_with_timestamps_drops_rows_inside_the_label_horizon(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target, timestamps=_hourly_stamps(200))

        assert [f["n_train"] for f in folds] == [f["train_end"] - 11 for f in folds]

    def test_report_without_timestamps_stays_unpurged(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target)

        assert [f["n_train"] for f in folds] == [f["train_end"] for f in folds]

    def test_report_rejects_timestamps_out_of_time_order(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)
        stamps = _hourly_stamps(200).iloc[::-1].reset_index(drop=True)

        with pytest.raises(ValueError, match="time"):
            adapter.walk_forward_report(features, target, timestamps=stamps)


class TestStrongerBaselines:
    """ML-10: bramka wymaga pobicia max(zero, dryf, sam LLM) — baseline'y liczone
    wyłącznie z części treningowej foldu, bez podglądu walidacji."""

    @staticmethod
    def _llm_driven_dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
        """target = 0.01 × llm_trend_signal + szum; pozostałe cechy to szum."""
        rng = np.random.default_rng(42)
        features = pd.DataFrame({name: rng.normal(0, 1, n) for name in FEATURE_NAMES})
        features["llm_trend_signal"] = rng.choice([-1.0, 1.0], size=n)
        target = pd.Series(0.01 * features["llm_trend_signal"] + rng.normal(0, 0.005, n))
        return features, target

    def test_candidate_rejected_when_llm_only_baseline_is_better(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        # Dziś kandydat bije zero o ~45% i zostaje wdrożony, choć przeskalowany sam
        # sygnał LLM ma niższe RMSE — „hybryda" nic by nie dodała.
        features, target = self._llm_driven_dataset()

        result: Mapping[str, Any] = adapter.train(features, target)

        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]
        assert result["llm_baseline_rmse"] < result["candidate_holdout_rmse"]
        assert result["status"] == "skipped_validation_failed"
        assert "sam LLM" in result["reason"]
        assert not Path(model_path).exists()

    def test_result_reports_all_three_baselines(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)

        result: Mapping[str, Any] = adapter.train(features, target)

        assert result["status"] == "trained_successfully"
        for key in ("baseline_rmse", "drift_baseline_rmse", "llm_baseline_rmse"):
            assert result[key] > result["candidate_holdout_rmse"]

    def test_fold_baselines_come_from_the_training_part_only(
        self, adapter: XGBoostAdapter
    ) -> None:
        # n=15 → jeden split: 5 wierszy treningu (target 0.01), 10 walidacji (0.05).
        # Z podglądem walidacji oba baseline'y miałyby RMSE 0; z samego treningu
        # przewidują 0.01, więc RMSE = 0.04.
        features, _ = _synthetic_dataset(n=15)
        features["llm_trend_signal"] = 1.0
        target = pd.Series([0.01] * 5 + [0.05] * 10)

        (fold,) = adapter.walk_forward_report(features, target)

        assert fold["drift_baseline_rmse"] == pytest.approx(0.04)
        assert fold["llm_baseline_rmse"] == pytest.approx(0.04)

    def test_llm_baseline_is_omitted_without_the_llm_feature(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(
            features.drop(columns=["llm_trend_signal"]), target
        )

        assert all("llm_baseline_rmse" not in fold for fold in folds)
        assert all("drift_baseline_rmse" in fold for fold in folds)


class TestFeatureStatsBins:
    """ML-21: `feature_stats` niesie biny kwantylowe rozkładu treningu i PSI
    ostatnich 30 dni próbki względem jej starszej części (mean/std zostają)."""

    def test_stats_carry_quantile_bins_summing_to_one(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=120)

        stats = cast(dict[str, Any], adapter.train(features, target)["feature_stats"])

        for name in FEATURE_NAMES:
            assert {"mean", "std", "edges", "fractions", "psi_recent_30d"} <= set(stats[name])
            assert len(stats[name]["fractions"]) == len(stats[name]["edges"]) + 1
            assert sum(stats[name]["fractions"]) == pytest.approx(1.0)
        # Cecha dyskretna (sygnał LLM −1/0/+1) dostaje granice z wartości próby.
        assert stats["llm_trend_signal"]["edges"] == [-1.0, 0.0, 1.0]

    def test_psi_flags_a_feature_shifted_in_the_last_30_days(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)
        stamps = pd.Series(pd.date_range("2026-05-01", periods=200, freq="D", tz="UTC"))
        # Ostatnie 30 dni: rozkład price_delta przesunięty o +0.2 (dryf).
        features.loc[170:, "price_delta"] += 0.2

        stats = cast(
            dict[str, Any], adapter.train(features, target, timestamps=stamps)["feature_stats"]
        )

        assert stats["price_delta"]["psi_recent_30d"] > 0.25
        assert stats["av_relevance_avg"]["psi_recent_30d"] < 0.25

    def test_psi_is_none_without_timestamps(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=120)

        stats = cast(dict[str, Any], adapter.train(features, target)["feature_stats"])

        assert stats["price_delta"]["psi_recent_30d"] is None

class TestPairedLossTest:
    """EXP-04: foldy niosą Σd i Σd² różnicy strat (baseline − kandydat), a domena
    wymaga istotnego z — sam margines 2% na średniej 4 foldów przepuszczał szum."""

    def test_folds_carry_paired_loss_sums_for_every_baseline(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target)

        for fold in folds:
            for prefix in ("", "drift_", "llm_"):
                assert f"{prefix}sum_loss_diff" in fold
                assert fold[f"{prefix}sum_loss_diff_sq"] >= 0.0
            # Średnia d to różnica średnich strat: RMSE_baseline² − RMSE_kandydata².
            assert fold["sum_loss_diff"] / fold["n_valid"] == pytest.approx(
                fold["baseline_rmse"] ** 2 - fold["candidate_rmse"] ** 2
            )

    def test_pure_noise_is_rejected_across_seeds(self, model_path: str) -> None:
        # 80 wierszy → 4 foldy po 16 = 64 punkty OOS (≥ MIN_OOS_POINTS). Przy samym
        # marginesie część ziaren szumu (normalnego i grubogonowego) przechodziła.
        passed = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            features = pd.DataFrame({name: rng.normal(0, 1, 80) for name in FEATURE_NAMES})
            for kind, noise in (
                ("normal", rng.normal(0, 0.05, 80)),
                ("student-t", rng.standard_t(2, 80) * 0.02),
            ):
                result = XGBoostAdapter(model_path=model_path).train(
                    features, pd.Series(noise), persist=False
                )
                if result["status"] == "evaluated_passed_gate":
                    passed.append((kind, seed))

        assert passed == []


class TestWeightsHash:
    """CI-13: karta modelu niesie sha256 pliku wag, który faktycznie leży na dysku
    (ten sam plik idzie do Release `model-latest`) — bez ponownej serializacji."""

    def test_accepted_report_carries_the_sha256_of_the_written_file(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        import hashlib

        features, target = _synthetic_dataset(n=200)

        report = adapter.train(features, target)

        assert report.outcome is TrainingOutcome.ACCEPTED
        expected = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        assert report.weights_sha256 == expected
        assert report["weights_sha256"] == expected

    def test_reports_without_written_weights_carry_no_hash(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=200)

        evaluated = adapter.train(features, target, persist=False)
        rejected = adapter.train(features, pd.Series([0.0] * 200))

        assert evaluated.weights_sha256 is None
        assert rejected.weights_sha256 is None
        assert "weights_sha256" not in rejected


class TestReproducibleTraining:
    """ML-22: raport niesie odcisk zbioru, parametry z jawnym ziarnem i wersję
    biblioteki, a zapis wag dostaje sidecar `.json`; te same dane i to samo ziarno
    dają identyczny booster."""

    def test_default_params_pin_the_random_state(self) -> None:
        from src.infrastructure.adapters.xgboost_local import DEFAULT_PARAMS

        assert DEFAULT_PARAMS["random_state"] == 0

    def test_dataset_fingerprint_independent_of_input_row_order(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = _synthetic_dataset(n=60)
        rows = features.to_dict("records")

        forward = adapter.train(rows, list(target), persist=False)
        backward = adapter.train(rows[::-1], list(target)[::-1], persist=False)

        assert forward.provenance is not None and backward.provenance is not None
        assert forward.provenance.dataset_sha256 == backward.provenance.dataset_sha256
        changed = adapter.train(rows, [*list(target)[:-1], 0.5], persist=False)
        assert changed.provenance is not None
        assert changed.provenance.dataset_sha256 != forward.provenance.dataset_sha256

    def test_same_data_same_seed_gives_identical_booster_dump(self, tmp_path: Path) -> None:
        features, target = _synthetic_dataset(n=200)
        first = XGBoostAdapter(model_path=str(tmp_path / "a.ubj"))
        second = XGBoostAdapter(model_path=str(tmp_path / "b.ubj"))

        report = first.train(features, target)
        second.train(features, target)

        assert report.outcome is TrainingOutcome.ACCEPTED
        assert report.provenance is not None
        assert first.booster_dump() == second.booster_dump()
        refit = XGBoostAdapter.refit_booster_dump(
            features.to_dict("records"), list(target), params=report.provenance.params
        )
        assert refit == first.booster_dump()

    def test_report_carries_the_provenance_of_the_sample(
        self, adapter: XGBoostAdapter
    ) -> None:
        import xgboost

        from src.domain.training_provenance import DatasetDescription

        features, target = _synthetic_dataset(n=60)
        stamps = [T0 + pd.Timedelta(hours=i).to_pytimedelta() for i in range(60)]
        description = DatasetDescription(
            n_per_symbol={"AAPL": 40, "BTC": 20}, feature_set_version=4, git_sha="abc1234"
        )

        report = adapter.train(
            features, target, persist=False, timestamps=stamps, dataset=description
        )

        assert report["n_per_symbol"] == {"AAPL": 40, "BTC": 20}
        assert report["feature_set_version"] == 4
        assert report["git_sha"] == "abc1234"
        assert report["data_start"] == stamps[0].isoformat()
        assert report["data_end"] == stamps[-1].isoformat()
        assert report["library_version"] == f"xgboost {xgboost.__version__}"
        params = report["training_params"]
        assert isinstance(params, dict)
        assert params["random_state"] == 0

    def test_accepted_training_writes_a_sidecar_next_to_the_weights(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        import json

        features, target = _synthetic_dataset(n=200)

        report = adapter.train(features, target)

        sidecar = json.loads(Path(model_path).with_suffix(".json").read_text(encoding="utf-8"))
        assert report.provenance is not None
        assert sidecar == {
            "model_file": Path(model_path).name,
            "weights_sha256": report.weights_sha256,
            "target_unit": "return",
            "n_samples": 200,
            **report.provenance.as_record(),
        }

    def test_rejected_training_writes_no_sidecar(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, _ = _synthetic_dataset(n=200)

        adapter.train(features, pd.Series([0.0] * 200))

        assert not Path(model_path).with_suffix(".json").exists()

    def test_booster_dump_needs_a_trained_model(self, adapter: XGBoostAdapter) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.booster_dump()


class TestSettledPurge:
    """ML-26: z chwilami końca etykiet (`label_ends`) fold trenuje na wierszach
    z `purge_training_indices` — także w fallbacku single-split; bez nich foldy są
    jak dotąd (purge stałego horyzontu 12 h)."""

    @staticmethod
    def _stamps(n: int) -> list[datetime]:
        return [T0 + pd.Timedelta(hours=i).to_pytimedelta() for i in range(n)]

    def test_fixed_horizon_label_ends_change_nothing(self, adapter: XGBoostAdapter) -> None:
        from src.domain.backtest import LABEL_HORIZON

        features, target = _synthetic_dataset(n=200)
        stamps = self._stamps(200)

        plain = adapter.walk_forward_report(features, target, timestamps=stamps)
        explicit = adapter.walk_forward_report(
            features,
            target,
            timestamps=stamps,
            label_ends=[stamp + LABEL_HORIZON for stamp in stamps],
        )

        assert explicit == plain

    def test_late_settlements_leave_the_training_part(self, adapter: XGBoostAdapter) -> None:
        from src.domain.backtest import LABEL_HORIZON, purge_training_indices

        features, target = _synthetic_dataset(n=200)
        stamps = self._stamps(200)
        # Co dziesiąty wiersz rozlicza się dopiero po 72 h (weekend).
        ends = [
            stamp + (pd.Timedelta(hours=72).to_pytimedelta() if i % 10 == 0 else LABEL_HORIZON)
            for i, stamp in enumerate(stamps)
        ]

        folds = adapter.walk_forward_report(features, target, timestamps=stamps, label_ends=ends)

        expected = [
            float(len(purge_training_indices(stamps, ends, int(fold["train_end"]))))
            for fold in folds
        ]
        assert [fold["n_train"] for fold in folds] == expected
        assert all(fold["n_train"] < fold["train_end"] - 11 for fold in folds)

    def test_single_split_fallback_is_purged_too(self, adapter: XGBoostAdapter) -> None:
        from src.domain.backtest import LABEL_HORIZON

        features, target = _synthetic_dataset(n=30)
        stamps = self._stamps(30)
        ends = [stamp + LABEL_HORIZON for stamp in stamps]
        ends[0] = stamps[-1]

        folds = adapter.walk_forward_report(features, target, timestamps=stamps, label_ends=ends)

        assert len(folds) == 1
        # Walidacja od wiersza 20: purge 12 h zostawia 0..8, a wiersz 0 rozlicza się
        # dopiero na końcu próbki — też wypada.
        assert folds[0]["n_train"] == 8.0

    def test_training_forwards_label_ends_to_the_gate(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=200)
        stamps = self._stamps(200)
        everything_late = [stamps[-1]] * 200

        report = adapter.train(
            features, target, persist=False, timestamps=stamps, label_ends=everything_late
        )

        # Żaden fold nie ma wiersza treningu — nie ma czego oceniać.
        assert report.n_folds == 0
        assert report.outcome is TrainingOutcome.EVALUATION_FAILED

    def test_label_ends_need_timestamps(self, adapter: XGBoostAdapter) -> None:
        features, target = _synthetic_dataset(n=60)

        with pytest.raises(ValueError, match="timestamps"):
            adapter.walk_forward_report(features, target, label_ends=self._stamps(60))


class TestDirectionVariant:
    """ML-30: wariant binarny P(wzrost poza martwą strefą) — klasyfikator z bramką
    Briera względem stopy bazowej; nie dotyka wag championa (regresji)."""

    @staticmethod
    def _direction_dataset(n: int = 300) -> tuple[pd.DataFrame, list[float]]:
        """Zwrot = 0.01 × znak sygnału + szum; co piąty wiersz w martwej strefie."""
        rng = np.random.default_rng(11)
        features = pd.DataFrame({name: rng.normal(0, 1, n) for name in FEATURE_NAMES})
        features["llm_trend_signal"] = rng.choice([-1.0, 1.0], size=n)
        target = list(0.01 * features["llm_trend_signal"] + rng.normal(0, 0.004, n))
        for i in range(0, n, 5):
            target[i] = 0.0002 if i % 2 else -0.0002
        return features, target

    def test_rows_inside_dead_band_are_excluded_from_training(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = self._direction_dataset()
        inside = sum(1 for value in target if abs(value) <= 0.001)

        report = adapter.train_direction_classifier(
            features, target, dead_band=0.001, persist=False
        )

        assert inside >= 60
        assert report.n_in_dead_band == inside
        assert report.n_samples == len(target) - inside
        assert report.dead_band == 0.001

    def test_classifier_variant_outputs_probability_in_unit_interval(
        self, adapter: XGBoostAdapter
    ) -> None:
        features, target = self._direction_dataset()

        report = adapter.train_direction_classifier(features, target, dead_band=0.001)

        assert report.outcome is TrainingOutcome.ACCEPTED
        assert report.summary is not None and report.summary.beats_base_rate
        up = dict(features.iloc[1]) | {"llm_trend_signal": 1.0}
        down = dict(features.iloc[1]) | {"llm_trend_signal": -1.0}
        p_up, p_down = adapter.predict_proba(up), adapter.predict_proba(down)
        assert 0.0 < p_down < 0.5 < p_up < 1.0

    def test_direction_variant_never_touches_the_champion_weights(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, target = self._direction_dataset()

        report = adapter.train_direction_classifier(features, target, dead_band=0.001)

        assert not Path(model_path).exists()
        assert adapter.is_trained is False
        assert report.model_path == str(Path(model_path).with_name("price_predictor_direction.ubj"))
        assert Path(report.model_path).exists()

    def test_noise_classifier_is_rejected_and_not_written(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        rng = np.random.default_rng(5)
        features = pd.DataFrame({name: rng.normal(0, 1, 300) for name in FEATURE_NAMES})
        target = list(rng.normal(0, 0.01, 300))

        report = adapter.train_direction_classifier(features, target, dead_band=0.001)

        assert report.outcome is TrainingOutcome.REJECTED
        assert not Path(model_path).with_name("price_predictor_direction.ubj").exists()
        with pytest.raises(RuntimeError, match="direction"):
            adapter.predict_proba(dict(features.iloc[0]))

    def test_evaluation_mode_reports_without_writing(
        self, adapter: XGBoostAdapter, model_path: str
    ) -> None:
        features, target = self._direction_dataset()

        report = adapter.train_direction_classifier(
            features, target, dead_band=0.001, persist=False
        )

        assert report.outcome is TrainingOutcome.EVALUATION_PASSED
        assert report.model_path is None
        assert list(Path(model_path).parent.iterdir()) == []

    def test_folds_purge_like_the_regression(self, adapter: XGBoostAdapter) -> None:
        features, target = self._direction_dataset()
        stamps = [T0 + pd.Timedelta(hours=i).to_pytimedelta() for i in range(300)]

        report = adapter.train_direction_classifier(
            features, target, dead_band=0.001, persist=False, timestamps=stamps
        )

        assert report.summary is not None
        assert report.summary.n_folds == 4
        # Foldy liczą się na wierszach z etykietą: pierwszy blok to trening.
        assert report.summary.n_oos_points == report.n_samples - report.n_samples // 5


class TestChallenger:
    """ML-33: challenger nogi ML z własnego pliku wag — regresor zwrotu albo
    klasyfikator kierunku ML-30. Prognoza cienia nie dotyka championa."""

    @staticmethod
    def _row() -> dict[str, float]:
        features, _ = _synthetic_dataset(n=1)
        return {name: float(value) for name, value in features.iloc[0].to_dict().items()}

    @staticmethod
    def _regressor(path: Path, *, scale: float = 2.0, sigma: bool = False) -> str:
        features, target = _synthetic_dataset(n=200)
        unit = TargetUnit.SIGMA if sigma else TargetUnit.RETURN
        XGBoostAdapter(model_path=str(path)).train(features, target * scale, target_unit=unit)
        return str(path)

    @staticmethod
    def _classifier(champion_path: str) -> tuple[XGBoostAdapter, str]:
        features, target = TestDirectionVariant._direction_dataset()
        adapter = XGBoostAdapter(model_path=champion_path)
        report = adapter.train_direction_classifier(features, target, dead_band=0.001)
        assert report.model_path is not None
        return adapter, report.model_path

    def test_regression_challenger_forecasts_the_return_of_its_own_weights(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        path = self._regressor(tmp_path / "price_predictor_alt.ubj")
        challenger = load_challenger("alt", path, ForecastKind.RETURN)

        assert challenger is not None
        forecast = challenger.forecast(self._row())
        served = XGBoostAdapter(model_path=path).predict(self._row(), current_price=Decimal("1"))
        assert forecast.model_name == "alt"
        assert forecast.model_version == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert forecast.proba is None
        assert forecast.predicted_return == pytest.approx(float(served.amount) - 1.0)

    def test_direction_challenger_forecasts_the_ml30_probability(
        self, model_path: str
    ) -> None:
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        adapter, path = self._classifier(model_path)
        challenger = load_challenger("direction", path, ForecastKind.DIRECTION)

        assert challenger is not None
        forecast = challenger.forecast(self._row())
        assert forecast.predicted_return is None
        assert forecast.proba == pytest.approx(adapter.predict_proba(self._row()))

    def test_challengers_leave_the_champion_prediction_unchanged(
        self, trained_adapter: XGBoostAdapter, model_path: str, tmp_path: Path
    ) -> None:
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        row, price = self._row(), Decimal("100")
        before = trained_adapter.predict(row, current_price=price)
        alt = load_challenger(
            "alt", self._regressor(tmp_path / "alt.ubj", scale=-3.0), ForecastKind.RETURN
        )
        _, direction_path = self._classifier(model_path)
        direction = load_challenger("direction", direction_path, ForecastKind.DIRECTION)
        assert alt is not None and direction is not None
        alt.forecast(row)
        direction.forecast(row)

        assert trained_adapter.predict(row, current_price=price) == before
        assert XGBoostAdapter(model_path=model_path).predict(row, current_price=price) == before

    def test_missing_weights_mean_no_challenger(self, tmp_path: Path) -> None:
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        assert load_challenger("alt", str(tmp_path / "absent.ubj"), ForecastKind.RETURN) is None

    def test_weights_of_another_kind_or_a_broken_file_are_refused(
        self, model_path: str, tmp_path: Path
    ) -> None:
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        regressor = self._regressor(tmp_path / "alt.ubj")
        _, classifier = self._classifier(model_path)
        broken = tmp_path / "broken.ubj"
        broken.write_bytes(b"not a model")

        assert load_challenger("alt", regressor, ForecastKind.DIRECTION) is None
        assert load_challenger("direction", classifier, ForecastKind.RETURN) is None
        assert load_challenger("alt", str(broken), ForecastKind.RETURN) is None

    def test_a_bare_booster_of_another_objective_is_refused(self, tmp_path: Path) -> None:
        # Plik bez metadanych sklearn (goły booster) przechodzi wczytanie — rodzaj
        # rozpoznaje dopiero cel boostera.
        import xgboost as xgb

        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        features, target = _synthetic_dataset(n=60)
        path = tmp_path / "bare.ubj"
        xgb.train(
            {"objective": "reg:squarederror"},
            xgb.DMatrix(features, label=target),
            num_boost_round=2,
        ).save_model(str(path))

        assert load_challenger("bare", str(path), ForecastKind.DIRECTION) is None
        assert load_challenger("bare", str(path), ForecastKind.RETURN) is not None

    def test_a_vector_without_the_model_features_is_a_malformed_payload(
        self, tmp_path: Path
    ) -> None:
        from src.application.errors import MalformedPayload
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        challenger = load_challenger(
            "alt", self._regressor(tmp_path / "alt.ubj"), ForecastKind.RETURN
        )
        assert challenger is not None

        with pytest.raises(MalformedPayload):
            challenger.forecast({"price_delta": 0.01})

    def test_sigma_challenger_is_rescaled_and_needs_the_sigma(self, tmp_path: Path) -> None:
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        path = self._regressor(tmp_path / "sigma.ubj", scale=50.0, sigma=True)
        challenger = load_challenger("sigma", path, ForecastKind.RETURN)
        assert challenger is not None

        with pytest.raises(ReturnScaleUnavailableError):
            challenger.forecast(self._row())
        low = challenger.forecast(self._row(), return_scale=0.01).predicted_return
        high = challenger.forecast(self._row(), return_scale=0.03).predicted_return
        assert low is not None and high is not None and low != 0
        assert high == pytest.approx(3 * low)

    def test_challenger_implements_the_shadow_port(self, tmp_path: Path) -> None:
        from src.application.ports import ShadowModelPort
        from src.domain.promotion import ForecastKind
        from src.infrastructure.adapters.xgboost_local import load_challenger

        challenger = load_challenger(
            "alt", self._regressor(tmp_path / "alt.ubj"), ForecastKind.RETURN
        )

        assert isinstance(challenger, ShadowModelPort)
        assert challenger.name == "alt"
