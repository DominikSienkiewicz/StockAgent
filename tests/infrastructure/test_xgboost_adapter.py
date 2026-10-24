from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort
from src.domain.model_scorecard import TrainingOutcome, TrainingReport
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
        current_price = Decimal("100")
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.predict(features, current_price=current_price)


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
        features = self._typical_features()
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.explain(features)

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


class TestDomainGate:
    """ML-05/ML-06: bramka „bije baseline" żyje w domenie (`src.domain.backtest`),
    a adapter tylko ją stosuje przed zapisem wag."""

    def test_noise_level_improvement_is_not_shipped(
        self, adapter: XGBoostAdapter, model_path: str, monkeypatch: pytest.MonkeyPatch
    ):
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
    ):
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
    ):
        features, target = _synthetic_dataset(n=200)

        report = adapter.train(features, target)

        assert isinstance(report, TrainingReport)
        assert report.outcome is TrainingOutcome.ACCEPTED
        assert report.n_samples == 200
        assert report.n_folds is not None and report.n_folds > 1
        assert report.rmse is not None and report.baseline_rmse is not None
        assert report.rmse < report.baseline_rmse
        assert report.model_path == model_path

    def test_evaluation_reports_carry_evaluation_outcomes(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)

        passed = adapter.train(features, target, persist=False)
        failed = adapter.train(features, pd.Series([0.0] * 200), persist=False)

        assert passed.outcome is TrainingOutcome.EVALUATION_PASSED
        assert failed.outcome is TrainingOutcome.EVALUATION_FAILED
        assert passed.model_path is None


class TestPooledTrainingSupport:
    """ML-13: per-symbol pętla tylko ewaluuje (bez zapisu), a model wspólny trenuje
    na foldach po czasie GLOBALNYM z purgem horyzontu etykiety."""

    def test_evaluation_mode_never_writes_weights(
        self, adapter: XGBoostAdapter, model_path: str
    ):
        features, target = _synthetic_dataset(n=200)

        result = adapter.train(features, target, persist=False)

        assert result["status"] == "evaluated_passed_gate"
        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]
        assert not Path(model_path).exists()
        assert adapter.is_trained is False

    def test_evaluation_mode_reports_a_failed_gate_without_rejecting_a_model(
        self, adapter: XGBoostAdapter
    ):
        features, _ = _synthetic_dataset(n=200)

        result = adapter.train(features, pd.Series([0.0] * 200), persist=False)

        # Status ewaluacji, nie „odrzucony kandydat" — karta nie może sugerować,
        # że dla symbolu został jakiś osobny, sprawdzony model.
        assert result["status"] == "evaluated_failed_gate"

    def test_purged_folds_drop_training_rows_inside_the_label_horizon(
        self, adapter: XGBoostAdapter
    ):
        features, target = _synthetic_dataset(n=200)

        folds = adapter._walk_forward_folds_detail(
            features, target, timestamps=_hourly_stamps(200)
        )

        # Godzinne wiersze, horyzont 12h → 11 ostatnich wierszy przed walidacją
        # ma etykietę po jej starcie i wypada z treningu foldu.
        assert [f["n_train"] for f in folds] == [f["train_end"] - 11 for f in folds]

    def test_timestamps_must_follow_global_time_order(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)
        stamps = _hourly_stamps(200).iloc[::-1].reset_index(drop=True)

        with pytest.raises(ValueError, match="time"):
            adapter.train(features, target, timestamps=stamps)

    def test_pooled_training_with_timestamps_ships_once(
        self, adapter: XGBoostAdapter, model_path: str
    ):
        features, target = _synthetic_dataset(n=200)

        result = adapter.train(features, target, timestamps=_hourly_stamps(200))

        assert result["status"] == "trained_successfully"
        assert Path(model_path).exists()


class TestSignalSkillMetrics:
    """ML-20: obok RMSE fold niesie IC (korelacja rang Spearmana prognozy z
    realizacją), zwrot strategii znakowej i turnover."""

    def test_fold_detail_contains_ic(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target)

        for fold in folds:
            assert {"ic", "sign_strategy_return", "turnover"} <= fold.keys()
            assert -1.0 <= fold["ic"] <= 1.0
            assert 0.0 <= fold["turnover"] <= 1.0
        # Sygnał jest realny → porządek prognoz zgodny z realizacją.
        assert all(fold["ic"] > 0.3 for fold in folds)
        assert all(fold["sign_strategy_return"] > 0 for fold in folds)

    def test_constant_target_has_zero_ic_instead_of_nan(self, adapter: XGBoostAdapter):
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
    ):
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target, timestamps=_hourly_stamps(200))

        assert [f["n_train"] for f in folds] == [f["train_end"] - 11 for f in folds]

    def test_report_without_timestamps_stays_unpurged(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(features, target)

        assert [f["n_train"] for f in folds] == [f["train_end"] for f in folds]

    def test_report_rejects_timestamps_out_of_time_order(self, adapter: XGBoostAdapter):
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
    ):
        # Dziś kandydat bije zero o ~45% i zostaje wdrożony, choć przeskalowany sam
        # sygnał LLM ma niższe RMSE — „hybryda" nic by nie dodała.
        features, target = self._llm_driven_dataset()

        result = adapter.train(features, target)

        assert result["candidate_holdout_rmse"] < result["baseline_rmse"]
        assert result["llm_baseline_rmse"] < result["candidate_holdout_rmse"]
        assert result["status"] == "skipped_validation_failed"
        assert "sam LLM" in result["reason"]
        assert not Path(model_path).exists()

    def test_result_reports_all_three_baselines(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=200)

        result = adapter.train(features, target)

        assert result["status"] == "trained_successfully"
        for key in ("baseline_rmse", "drift_baseline_rmse", "llm_baseline_rmse"):
            assert result[key] > result["candidate_holdout_rmse"]

    def test_fold_baselines_come_from_the_training_part_only(
        self, adapter: XGBoostAdapter
    ):
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
    ):
        features, target = _synthetic_dataset(n=200)

        folds = adapter.walk_forward_report(
            features.drop(columns=["llm_trend_signal"]), target
        )

        assert all("llm_baseline_rmse" not in fold for fold in folds)
        assert all("drift_baseline_rmse" in fold for fold in folds)


class TestFeatureStatsBins:
    """ML-21: `feature_stats` niesie biny kwantylowe rozkładu treningu i PSI
    ostatnich 30 dni próbki względem jej starszej części (mean/std zostają)."""

    def test_stats_carry_quantile_bins_summing_to_one(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=120)

        stats = adapter.train(features, target)["feature_stats"]

        for name in FEATURE_NAMES:
            assert {"mean", "std", "edges", "fractions", "psi_recent_30d"} <= set(stats[name])
            assert len(stats[name]["fractions"]) == len(stats[name]["edges"]) + 1
            assert sum(stats[name]["fractions"]) == pytest.approx(1.0)
        # Cecha dyskretna (sygnał LLM −1/0/+1) dostaje granice z wartości próby.
        assert stats["llm_trend_signal"]["edges"] == [-1.0, 0.0, 1.0]

    def test_psi_flags_a_feature_shifted_in_the_last_30_days(
        self, adapter: XGBoostAdapter
    ):
        features, target = _synthetic_dataset(n=200)
        stamps = pd.Series(pd.date_range("2026-05-01", periods=200, freq="D", tz="UTC"))
        # Ostatnie 30 dni: rozkład price_delta przesunięty o +0.2 (dryf).
        features.loc[170:, "price_delta"] += 0.2

        stats = adapter.train(features, target, timestamps=stamps)["feature_stats"]

        assert stats["price_delta"]["psi_recent_30d"] > 0.25
        assert stats["av_relevance_avg"]["psi_recent_30d"] < 0.25

    def test_psi_is_none_without_timestamps(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=120)

        stats = adapter.train(features, target)["feature_stats"]

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
