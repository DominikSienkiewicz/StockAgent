from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

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
    def test_returns_success_status(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)
        assert result["status"] == "trained_successfully"

    def test_includes_sample_count_in_status(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=50)
        result = adapter.train(features, target)
        assert result["n_samples"] == 50

    def test_saves_model_to_disk(self, adapter: XGBoostAdapter, model_path: str):
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0

    def test_returns_validation_metrics(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)

        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]

    def test_metric_keys_are_candidate_scoped_and_self_describing(
        self, adapter: XGBoostAdapter
    ):
        # Finding #22: shipowany model jest refitowany na WSZYSTKICH danych i nie
        # jest re-walidowany, więc raportowane RMSE/hit-rate dotyczą KANDYDATA na
        # holdoucie — nazwy kluczy muszą to mówić wprost, plus `metric_note`.
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)

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
    ):
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
    ):
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
    ):
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)

        hit_rate = result["candidate_holdout_directional_hit_rate"]
        assert 0.0 <= hit_rate <= 1.0
        # Sygnał jest realny → kierunek trafiony częściej niż losowo.
        assert hit_rate > 0.5


class TestWalkForwardValidation:
    """Finding #23: ewaluacja walk-forward (expanding window) zamiast jednego,
    hałaśliwego splitu 20%. Decyzja o shipowaniu zależy od ŚREDNIEJ z foldów +
    raport niesie per-feature drift stats i `n_folds`."""

    def test_strong_signal_ships_across_folds(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)
        result = adapter.train(features, target)

        assert result["status"] == "trained_successfully"
        # Wystarczająco duża próbka → realne foldy walk-forward.
        assert result["n_folds"] >= 3
        # Bramka oparta na ŚREDNIEJ z foldów: kandydat bije baseline średnio.
        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]

    def test_no_signal_dataset_is_skipped(self, adapter: XGBoostAdapter):
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

    def test_zero_return_dataset_is_skipped(self, adapter: XGBoostAdapter):
        # Zwrot zawsze 0 → baseline persystencji bezbłędny, nic go nie pobije.
        features, _ = _synthetic_dataset(n=200)
        target = pd.Series([0.0] * 200)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert adapter.is_trained is False

    def test_result_reports_n_folds_above_one_for_large_sample(
        self, adapter: XGBoostAdapter
    ):
        features, target = _synthetic_dataset(n=200)
        result = adapter.train(features, target)
        assert result["n_folds"] > 1

    def test_small_sample_falls_back_to_single_split(
        self, adapter: XGBoostAdapter
    ):
        # Poniżej progu na walk-forward → pojedynczy split (zachowanie sane).
        features, target = _synthetic_dataset(n=15)
        result = adapter.train(features, target)
        assert result["n_folds"] == 1

    def test_result_includes_per_feature_distribution_stats(
        self, adapter: XGBoostAdapter
    ):
        # Drift-detection: per-feature mean/std w wyniku, żeby porównać z
        # poprzednim runem i wykryć dryf rozkładu cech wejściowych.
        features, target = _synthetic_dataset(n=120)
        result = adapter.train(features, target)

        stats = result["feature_stats"]
        assert set(stats.keys()) == set(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert "mean" in stats[name]
            assert "std" in stats[name]
            expected_mean = float(features[name].mean())
            assert stats[name]["mean"] == pytest.approx(expected_mean, rel=1e-9)

    def test_feature_stats_present_even_when_skipped(
        self, adapter: XGBoostAdapter
    ):
        # Dryf cech jest obserwowalny niezależnie od werdyktu bramki.
        features, _ = _synthetic_dataset(n=200)
        target = pd.Series([0.0] * 200)
        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert set(result["feature_stats"].keys()) == set(FEATURE_NAMES)


class TestPredict:
    def test_returns_money_instance(self, trained_adapter: XGBoostAdapter):
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
    ):
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
    ):
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

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter):
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.predict(features, current_price=Decimal("100"))


class TestPredictFeatureAlignment:
    """predict() wyrównuje wejście do cech, na których model był NAPRAWDĘ
    trenowany (`feature_names_in_`). Bez tego ewolucja kontraktu cech (dodanie
    nowej cechy do ML_FEATURE_COLUMNS przy starym .ubj w repo) cicho psuła
    predykcję mismatchem nazw kolumn."""

    def test_ignores_unknown_extra_features_and_scrambled_order(
        self, trained_adapter: XGBoostAdapter
    ):
        # Model zna 7 cech. Wejście dorzuca nieznaną cechę + miesza kolejność.
        features: dict[str, float] = {"holding_hours": 24.0}
        for name in reversed(FEATURE_NAMES):
            features[name] = 0.5
        prediction = trained_adapter.predict(features, current_price=Decimal("100"))
        assert isinstance(prediction, Money)

    def test_raises_keyerror_when_required_feature_missing(
        self, trained_adapter: XGBoostAdapter
    ):
        # Brak wymaganej cechy → jawny błąd, nie ciche NaN wpychane do modelu.
        features = dict.fromkeys(FEATURE_NAMES[:-1], 0.5)
        with pytest.raises(KeyError):
            trained_adapter.predict(features, current_price=Decimal("100"))


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
    ):
        contribs = trained_adapter.explain(self._typical_features())
        # Klucz per cecha, na której model był trenowany (bias pod osobnym kluczem).
        assert set(FEATURE_NAMES).issubset(set(contribs.keys()))

    def test_contributions_are_finite_floats(
        self, trained_adapter: XGBoostAdapter
    ):
        contribs = trained_adapter.explain(self._typical_features())
        for value in contribs.values():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_sum_of_contributions_plus_bias_approximates_raw_prediction(
        self, trained_adapter: XGBoostAdapter
    ):
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
    ):
        # Bias/base value jest osobno (nie jako cecha) — czytelny dla raportu.
        contribs = trained_adapter.explain(self._typical_features())
        assert "_bias" in contribs
        assert "_bias" not in FEATURE_NAMES

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter):
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.explain(self._typical_features())

    def test_ignores_unknown_extra_features(
        self, trained_adapter: XGBoostAdapter
    ):
        # Tak jak predict(): nieznana cecha jest ignorowana (wyrównanie do modelu).
        features = self._typical_features()
        features["holding_hours"] = 24.0
        contribs = trained_adapter.explain(features)
        assert "holding_hours" not in contribs


class TestModelPersistence:
    def test_loads_existing_model_from_disk_on_init(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ):
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

    def test_cold_start_does_not_raise(self, tmp_path: Path):
        # Brak pliku — konstruktor musi przeżyć (czekamy na pierwszy trening).
        adapter = XGBoostAdapter(model_path=str(tmp_path / "missing.ubj"))
        # Cold start: model nie jest jeszcze wczytany, adapter to wie.
        assert adapter.is_trained is False


class TestIsTrained:
    def test_false_on_cold_start(self, adapter: XGBoostAdapter):
        # Brak pliku modelu → is_trained False (Fast Loop użyje baseline)
        assert adapter.is_trained is False

    def test_true_after_training(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert adapter.is_trained is True

    def test_true_when_model_loaded_from_disk(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ):
        reloaded = XGBoostAdapter(model_path=model_path)
        assert reloaded.is_trained is True


class TestAdapterImplementsPort:
    def test_is_ml_prediction_port(self):
        assert issubclass(XGBoostAdapter, MLPredictionPort)
