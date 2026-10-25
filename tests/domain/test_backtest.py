from datetime import UTC, datetime, timedelta

import pytest

from src.domain.backtest import (
    LABEL_HORIZON,
    MIN_OOS_POINTS,
    MIN_REL_IMPROVEMENT,
    BacktestSummary,
    LabelOverlap,
    beats_baseline,
    count_label_overlaps,
    label_end_times,
    purge_training_indices,
    purged_train_end,
    summarize_folds,
)


class TestSummarizeFolds:
    """Agregacja per-fold metryk walk-forward w jeden werdykt backtestu.

    Czysta funkcja: średnie arytmetyczne RMSE/hit-rate, procentowa poprawa
    względem baseline'u (zwrot=0) i binarny werdykt „bije / nie bije baseline".
    """

    def test_means_are_arithmetic_over_folds(self) -> None:
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = summarize_folds(folds)

        assert summary.n_folds == 2
        assert abs(summary.mean_candidate_rmse - 0.20) < 1e-9
        assert abs(summary.mean_baseline_rmse - 0.30) < 1e-9
        assert abs(summary.mean_hit_rate - 0.70) < 1e-9

    def test_improvement_pct_relative_to_baseline(self) -> None:
        # mean_candidate=0.20, mean_baseline=0.30 → (0.30-0.20)/0.30*100 ≈ 33.33%
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = summarize_folds(folds)

        assert abs(summary.improvement_pct - (10.0 / 30.0 * 100)) < 1e-9

    def test_beats_baseline_true_when_candidate_rmse_lower(self) -> None:
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
        ]

        summary = summarize_folds(folds)

        assert summary.beats_baseline is True
        assert "bije baseline" in summary.verdict

    def test_does_not_beat_baseline_when_candidate_rmse_higher(self) -> None:
        folds = [
            {"candidate_rmse": 0.40, "baseline_rmse": 0.20, "hit_rate": 0.30},
        ]

        summary = summarize_folds(folds)

        assert summary.beats_baseline is False
        # improvement_pct ujemny: (0.20-0.40)/0.20*100 = -100%
        assert abs(summary.improvement_pct - (-100.0)) < 1e-9
        assert "nie bije baseline" in summary.verdict

    def test_improvement_pct_zero_when_baseline_zero(self) -> None:
        # Guard dzielenia przez zero: baseline=0 → improvement_pct=0.0
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.0, "hit_rate": 0.50},
        ]

        summary = summarize_folds(folds)

        assert summary.improvement_pct == pytest.approx(0.0)
        # candidate 0.10 > baseline 0.0 → nie bije
        assert summary.beats_baseline is False

    def test_verdict_mentions_improvement_percentage_when_beating(self) -> None:
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
        ]

        summary = summarize_folds(folds)

        # Werdykt zawiera liczbę procent poprawy (50%).
        assert "50" in summary.verdict

    def test_empty_folds_returns_brak_danych(self) -> None:
        summary = summarize_folds([])

        assert summary == BacktestSummary(0, 0, 0, 0, 0, False, "brak danych")
        assert summary.n_folds == 0
        assert summary.beats_baseline is False
        assert summary.verdict == "brak danych"

    def test_summary_is_frozen(self) -> None:
        summary = summarize_folds([])

        import dataclasses

        try:
            summary.n_folds = 5  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:  # pragma: no cover
            raise AssertionError("BacktestSummary should be frozen")


def _folds(
    candidate: float, baseline: float, n_valid: float, n: int = 4
) -> list[dict[str, float]]:
    return [
        {
            "candidate_rmse": candidate,
            "baseline_rmse": baseline,
            "hit_rate": 0.5,
            "n_valid": n_valid,
        }
        for _ in range(n)
    ]


def test_tiny_improvement_does_not_beat_baseline() -> None:
    # ML-05: 0.5% poprawy to szum, nie „lepszy model". Próg 2% ustalony przed
    # spojrzeniem na wyniki — obniżanie go pod wynik byłoby p-hackingiem.
    assert beats_baseline(0.0200, 0.0201) is False

    summary = summarize_folds(_folds(0.0200, 0.0201, n_valid=20))

    assert summary.beats_baseline is False
    assert "nie bije baseline" in summary.verdict


def test_improvement_above_the_relative_margin_beats_baseline() -> None:
    assert MIN_REL_IMPROVEMENT == 0.02
    assert beats_baseline(0.0190, 0.0200) is True


def test_zero_baseline_is_never_beaten() -> None:
    # Bezbłędny baseline (zwrot zawsze 0) — nawet idealny kandydat go nie „bije".
    assert beats_baseline(0.0, 0.0) is False


def test_too_few_oos_points_do_not_beat_baseline() -> None:
    # ML-06: 4 foldy × 10 punktów = 40 < 60 — za mało na werdykt, nawet przy
    # wyraźnej poprawie RMSE.
    summary = summarize_folds(_folds(0.010, 0.020, n_valid=10))

    assert summary.n_oos_points == 40
    assert summary.beats_baseline is False
    assert "za mało punktów OOS" in summary.verdict


def test_enough_oos_points_keep_the_verdict() -> None:
    assert MIN_OOS_POINTS == 60
    summary = summarize_folds(_folds(0.010, 0.020, n_valid=15))

    assert summary.n_oos_points == 60
    assert summary.beats_baseline is True


def test_folds_without_n_valid_skip_the_oos_condition() -> None:
    # Brak klucza `n_valid` = warunek pominięty (kształt foldów sprzed ML-06).
    summary = summarize_folds(
        [{"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.6}]
    )

    assert summary.n_oos_points is None
    assert summary.beats_baseline is True


def test_purge_drops_training_rows_whose_label_overlaps_validation() -> None:
    # ML-13: wiersz z chwili t ma etykietę (zwrot 12h) zrealizowaną w t+12h. W modelu
    # wspólnym wiersz innego symbolu z tej samej godziny w treningu „zna" ruch rynku,
    # który walidacja ma dopiero przewidzieć — przeciek przekrojowy.
    start = datetime(2026, 10, 1, tzinfo=UTC)
    stamps = [start + timedelta(hours=i) for i in range(48)]

    fit_end = purged_train_end(stamps, valid_start=30)

    # Wiersz 18 (etykieta dokładnie w chwili 30) zostaje, 19..29 wypadają.
    assert fit_end == 19
    assert timedelta(hours=12) == LABEL_HORIZON


def test_purge_keeps_everything_when_labels_settle_before_validation() -> None:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    stamps = [start + timedelta(days=i) for i in range(10)]

    assert purged_train_end(stamps, valid_start=6) == 6


# -- ML-26: purge po chwili końca etykiety + embargo (czysta funkcja domeny) -------

_T = datetime(2026, 10, 1, tzinfo=UTC)


def _hours(*values: int) -> list[datetime]:
    return [_T + timedelta(hours=value) for value in values]


def test_purge_drops_training_rows_whose_label_ends_after_validation_start() -> None:
    # Etykieta wiersza 3 (piątek) rozlicza się dopiero w poniedziałek — po starcie
    # walidacji — choć wiersz 4 jest późniejszy i jego etykieta już dojrzała.
    prediction_ts = _hours(0, 1, 2, 3, 4, 20, 21)
    label_end_ts = _hours(12, 13, 14, 30, 16, 32, 33)

    kept = purge_training_indices(prediction_ts, label_end_ts, valid_start=5)

    assert kept == (0, 1, 2, 4)


def test_label_ending_exactly_at_validation_start_stays() -> None:
    kept = purge_training_indices(_hours(0, 1, 8), _hours(8, 9, 20), valid_start=2)

    assert kept == (0,)


def test_embargo_widens_the_gap_before_validation() -> None:
    prediction_ts = _hours(0, 1, 2, 3, 20)
    label_end_ts = _hours(12, 13, 14, 15, 32)

    assert purge_training_indices(prediction_ts, label_end_ts, 4) == (0, 1, 2, 3)
    # Start walidacji 20:00, embargo 6 h → etykieta musi skończyć się do 14:00.
    assert purge_training_indices(
        prediction_ts, label_end_ts, 4, embargo=timedelta(hours=6)
    ) == (0, 1, 2)


def test_fixed_horizon_label_ends_reproduce_purged_train_end() -> None:
    # Bez chwil rozliczenia (etykieta = chwila + 12 h, embargo 0) purge jest dokładnie
    # tym, co walk-forward robił dotąd — prefiks `purged_train_end`.
    stamps = [_T + timedelta(minutes=37 * i) for i in range(80)]
    ends = label_end_times(stamps, [None] * len(stamps))

    for valid_start in (1, 20, 50, 79):
        assert purge_training_indices(stamps, ends, valid_start) == tuple(
            range(purged_train_end(stamps, valid_start))
        )


def test_label_end_is_the_settlement_but_never_before_the_horizon() -> None:
    # 029 wpisała historii dolną granicę `settled_at = timestamp` — taki koniec
    # etykiety podważałby purge horyzontu, więc liczy się co najmniej chwila + 12 h.
    stamps = _hours(0, 0, 0)
    settled = [_T + timedelta(hours=72), _T, None]

    assert label_end_times(stamps, settled) == _hours(72, 12, 12)


def test_label_end_times_need_aligned_rows() -> None:
    with pytest.raises(ValueError, match="align"):
        label_end_times(_hours(0, 1), [None])


def _ic_folds(ics: list[float]) -> list[dict[str, float]]:
    return [
        {"candidate_rmse": 0.01, "baseline_rmse": 0.02, "hit_rate": 0.5, "ic": ic}
        for ic in ics
    ]


def test_summary_reports_mean_ic_and_t_stat() -> None:
    # ML-20: przy słabym sygnale RMSE nagradza model ściągający prognozy do zera;
    # o decyzji kierunkowej mówi porządek (IC), raportowany obok RMSE.
    summary = summarize_folds(_ic_folds([0.1, 0.2, 0.3, 0.2]))

    assert summary.mean_ic == pytest.approx(0.2)
    assert summary.ic_std == pytest.approx((0.02 / 3) ** 0.5)
    # t = IC_mean / sd · √folds
    assert summary.ic_t_stat == pytest.approx(0.2 / (0.02 / 3) ** 0.5 * 2)


def test_ic_t_stat_needs_at_least_two_folds() -> None:
    summary = summarize_folds(_ic_folds([0.3]))

    assert summary.mean_ic == pytest.approx(0.3)
    assert summary.ic_std is None
    assert summary.ic_t_stat is None


def test_ic_fields_stay_empty_when_folds_carry_no_ic() -> None:
    summary = summarize_folds(
        [{"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.6}]
    )

    assert summary.mean_ic is None
    assert summary.ic_t_stat is None


def _baseline_folds(
    candidate: float, zero: float, drift: float, llm: float
) -> list[dict[str, float]]:
    return [
        {
            "candidate_rmse": candidate,
            "baseline_rmse": zero,
            "drift_baseline_rmse": drift,
            "llm_baseline_rmse": llm,
            "hit_rate": 0.5,
            "n_valid": 20.0,
        }
        for _ in range(4)
    ]


def test_candidate_must_also_beat_the_llm_only_baseline() -> None:
    # ML-10: XGBoost dostaje `llm_trend_signal` jako cechę — kandydat, który bije
    # zero, ale nie bije przeskalowanego samego sygnału LLM, tylko przepisuje LLM.
    summary = summarize_folds(_baseline_folds(0.0200, 0.0300, 0.0300, 0.0201))

    assert summary.beats_baseline is False
    assert "nie bije baseline" in summary.verdict
    assert "sam LLM" in summary.verdict


def test_candidate_must_also_beat_the_mean_drift_baseline() -> None:
    summary = summarize_folds(_baseline_folds(0.0200, 0.0300, 0.0202, 0.0300))

    assert summary.beats_baseline is False
    assert "dryf" in summary.verdict


def test_gate_uses_the_same_relative_margin_against_the_strongest_baseline() -> None:
    # 0.0195 vs najsilniejszy 0.0200 = 2.5% ≥ MIN_REL_IMPROVEMENT → przechodzi.
    summary = summarize_folds(_baseline_folds(0.0195, 0.0300, 0.0250, 0.0200))

    assert summary.beats_baseline is True
    assert summary.mean_drift_baseline_rmse == pytest.approx(0.0250)
    assert summary.mean_llm_baseline_rmse == pytest.approx(0.0200)
    assert "sam LLM" in summary.verdict


def test_folds_without_the_extra_baselines_keep_the_zero_return_gate() -> None:
    # Kształt foldów sprzed ML-10 (np. offline harness ze starym adapterem).
    summary = summarize_folds(_folds(0.010, 0.020, n_valid=15))

    assert summary.mean_drift_baseline_rmse is None
    assert summary.mean_llm_baseline_rmse is None
    assert summary.beats_baseline is True
    assert summary.verdict == "model bije baseline o 50.0%"


def _strategy_folds(*pairs: tuple[float, float]) -> list[dict[str, float]]:
    """Foldy z `sign_strategy_return` i `turnover` (ML-20, odwrócenie = 1)."""
    return [
        {
            "candidate_rmse": 0.01,
            "baseline_rmse": 0.02,
            "hit_rate": 0.5,
            "sign_strategy_return": gross,
            "turnover": turnover,
        }
        for gross, turnover in pairs
    ]


def test_summary_reports_mean_net_return_after_turnover_costs() -> None:
    # WIA-50: turnover foldu liczy odwrócenie jako 1, czyli połowę obróconego
    # kapitału (dwie nogi). Turnover 0.5 = jedna noga na wiersz → 10 bps.
    summary = summarize_folds(_strategy_folds((0.002, 0.5), (0.004, 0.0)), cost_bps=10)

    assert summary.turnover == pytest.approx(0.25)
    assert summary.mean_net_return == pytest.approx(((0.002 - 0.001) + 0.004) / 2)


def test_net_return_equals_the_gross_strategy_return_without_costs() -> None:
    summary = summarize_folds(_strategy_folds((0.002, 1.0), (0.004, 0.5)))

    assert summary.mean_net_return == pytest.approx(0.003)
    assert summary.turnover == pytest.approx(0.75)


def test_net_return_and_turnover_stay_empty_without_strategy_metrics() -> None:
    summary = summarize_folds(
        [{"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.6}], cost_bps=10
    )

    assert summary.mean_net_return is None
    assert summary.turnover is None


def test_costs_do_not_move_the_rmse_verdict() -> None:
    folds = _strategy_folds((0.002, 1.0))

    assert summarize_folds(folds, cost_bps=50).verdict == summarize_folds(folds).verdict


class TestCountLabelOverlaps:
    """ML-25: ile wierszy rozlicza się dopiero po następnym wierszu swojego symbolu
    — pomiar, bez którego purge stałym `LABEL_HORIZON` jest strzałem w ciemno."""

    _T0 = datetime(2026, 10, 2, 9, 30, tzinfo=UTC)

    def test_counts_rows_settled_after_the_next_row_of_their_symbol(self) -> None:
        day = timedelta(days=1)
        rows = [
            # Piątek rozliczony w poniedziałek dopiero po poniedziałkowym wierszu.
            ("AAPL", self._T0, self._T0 + 3 * day + timedelta(hours=1)),
            ("AAPL", self._T0 + 3 * day, self._T0 + 4 * day - timedelta(hours=1)),
            ("AAPL", self._T0 + 4 * day, None),
            # Inny symbol: jego „następny wiersz" to wiersz tego samego symbolu.
            ("BTC", self._T0 + timedelta(hours=1), self._T0 + timedelta(hours=20)),
            ("BTC", self._T0 + day, self._T0 + 2 * day),
        ]

        assert count_label_overlaps(rows) == LabelOverlap(measured=3, overlapping=1)

    def test_order_of_rows_does_not_matter(self) -> None:
        rows = [
            ("AAPL", self._T0 + timedelta(hours=24), None),
            ("AAPL", self._T0, self._T0 + timedelta(hours=30)),
        ]

        assert count_label_overlaps(rows) == LabelOverlap(measured=1, overlapping=1)

    def test_rows_without_settled_at_or_next_row_are_not_measured(self) -> None:
        rows = [("AAPL", self._T0, None), ("MSFT", self._T0, self._T0 + timedelta(hours=26))]

        assert count_label_overlaps(rows) == LabelOverlap(measured=0, overlapping=0)


# EXP-04: test sparowany „DM-lite”. Średnia RMSE z 4 foldów bez testu istotności
# przepuszcza szum, gdy różnica strat na pojedynczych punktach ma duży rozrzut.
_MEAN_LOSS_DIFF = 0.0200**2 - 0.0194**2


def _paired_folds(
    spread: float, *, prefix: str = "", extra: dict[str, float] | None = None
) -> list[dict[str, float]]:
    """4 foldy po 30 punktów OOS, kandydat 3% lepszy od „zwrot 0” (margines 2%
    przechodzi); sumy d i d² dają średnią różnicę strat i odchylenie `spread`."""
    return [
        {
            "candidate_rmse": 0.0194,
            "baseline_rmse": 0.0200,
            "hit_rate": 0.5,
            "n_valid": 30.0,
            f"{prefix}sum_loss_diff": 30 * _MEAN_LOSS_DIFF,
            f"{prefix}sum_loss_diff_sq": 30 * (_MEAN_LOSS_DIFF**2 + spread**2),
            **(extra or {}),
        }
        for _ in range(4)
    ]


def test_margin_without_a_significant_paired_difference_does_not_pass() -> None:
    from src.domain.backtest import PAIRED_Z_CRIT

    summary = summarize_folds(_paired_folds(spread=1e-3))

    assert summary.improvement_pct > MIN_REL_IMPROVEMENT * 100
    assert summary.paired_z is not None and summary.paired_z < PAIRED_Z_CRIT
    assert summary.beats_baseline is False
    assert "istotnie" in summary.verdict


def test_significant_paired_difference_keeps_the_pass() -> None:
    from src.domain.backtest import PAIRED_Z_CRIT

    summary = summarize_folds(_paired_folds(spread=1e-5))

    assert summary.paired_z is not None and summary.paired_z > PAIRED_Z_CRIT
    assert summary.beats_baseline is True


def test_folds_without_paired_sums_keep_the_margin_rule() -> None:
    folds = [
        {"candidate_rmse": 0.0194, "baseline_rmse": 0.0200, "hit_rate": 0.5, "n_valid": 30.0}
        for _ in range(4)
    ]

    summary = summarize_folds(folds)

    assert summary.paired_z is None
    assert summary.beats_baseline is True


def test_paired_test_runs_against_the_binding_baseline() -> None:
    # Dryf (0.0199) jest najsilniejszy; względem niego różnica strat to szum,
    # choć względem „zwrot 0” byłaby istotna.
    drift_mean = 0.0199**2 - 0.0194**2
    extra = {
        "drift_baseline_rmse": 0.0199,
        "drift_sum_loss_diff": 30 * drift_mean,
        "drift_sum_loss_diff_sq": 30 * (drift_mean**2 + 1e-6),
    }

    summary = summarize_folds(_paired_folds(spread=1e-5, extra=extra))

    assert summary.beats_baseline is False
    assert "dryf średniej" in summary.verdict


def test_paired_loss_z_is_the_mean_difference_over_its_standard_error() -> None:
    from math import sqrt

    from src.domain.backtest import paired_loss_z

    folds = _paired_folds(spread=1e-4)
    n = 120
    sd = sqrt(n * 1e-4**2 / (n - 1))

    assert paired_loss_z(folds) == pytest.approx(_MEAN_LOSS_DIFF / (sd / sqrt(n)))


def test_paired_loss_z_is_undefined_without_sums_or_points() -> None:
    from src.domain.backtest import paired_loss_z

    assert paired_loss_z([{"n_valid": 30.0}]) is None
    assert paired_loss_z([{"sum_loss_diff": 1.0, "sum_loss_diff_sq": 1.0}]) is None
