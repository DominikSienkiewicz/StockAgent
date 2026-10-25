# src/domain/backtest.py
from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.trading_costs import trading_cost

# Werdykt walk-forward backtestu. Czysta domena — agreguje per-fold metryki
# (RMSE kandydata vs baseline persystencji „zwrot=0", trafność kierunku) w jeden
# obraz: czy lokalny model bije naiwny baseline i o ile. Bez I/O, bez pandas.
# To JEDYNE miejsce reguły „bije baseline" — adapter XGBoost stosuje ten sam
# werdykt przed zapisem wag (ML-05), zamiast trzymać własny próg.

# ML-05: kandydat musi obniżyć RMSE o co najmniej 2% WZGLĘDNIE. Próg ustalony
# przed spojrzeniem na wyniki — obniżanie go pod wynik to p-hacking.
MIN_REL_IMPROVEMENT = 0.02

# ML-06: werdykt wymaga co najmniej tylu punktów out-of-sample łącznie ze
# wszystkich foldów. Przy 4 foldach expanding-window oznacza to n ≥ 75 próbek.
MIN_OOS_POINTS = 60

# ML-10: kandydat musi pobić NAJSILNIEJSZY z trzech baseline'ów z tym samym
# marginesem względnym: persystencję („zwrot = 0"), dryf (średni target części
# treningowej foldu) i „sam LLM" (`llm_trend_signal` × średni |target| części
# treningowej). XGBoost dostaje `llm_trend_signal` jako cechę — jeśli nie bije
# przeskalowanego samego sygnału, noga ML tylko kosztownie przepisuje LLM.
ZERO_BASELINE_LABEL = "zwrot 0"
DRIFT_BASELINE_LABEL = "dryf średniej"
LLM_BASELINE_LABEL = "sam LLM"

# EXP-04: test sparowany „DM-lite” (Diebold–Mariano bez korekty HAC). Na każdym
# punkcie OOS d = strata² baseline'u − strata² kandydata; foldy niosą Σd i Σd²,
# a kandydat musi mieć z = średnia(d) / SE > PAIRED_Z_CRIT. 2.0 ≈ jednostronne
# α 0,023 dla JEDNEGO testu — wagi shipuje jeden model wspólny. Bonferroni dla
# 45 symboli (~2.8) dotyczyłby ocen per symbol, które wag nie zapisują.
# Bez korekty autokorelacji i korelacji między symbolami z jest zawyżone, więc
# test to warunek konieczny, nie dowód umiejętności.
PAIRED_Z_CRIT = 2.0

# Prefiks kluczy Σd / Σd² w foldzie per baseline (ten sam, który rozstrzyga margines).
_PAIRED_PREFIX = {
    ZERO_BASELINE_LABEL: "",
    DRIFT_BASELINE_LABEL: "drift_",
    LLM_BASELINE_LABEL: "llm_",
}

# Horyzont etykiety: target to zwrot 12h, więc wiersz z chwili t „zna" rynek do
# t + 12h. Wiersz treningowy, którego etykieta dojrzewa po starcie walidacji,
# przecieka do foldu (ML-13).
LABEL_HORIZON = timedelta(hours=12)


@dataclass(frozen=True)
class BacktestSummary:
    """Zagregowany wynik walk-forward backtestu nad N foldami.

    `mean_*` to średnie arytmetyczne metryk per-fold. `improvement_pct` to
    procentowa redukcja RMSE względem baseline'u (dodatnia = model lepszy).
    `beats_baseline` jest True gdy średnie RMSE kandydata jest niższe od
    baseline'u o co najmniej `MIN_REL_IMPROVEMENT` i foldy dały co najmniej
    `MIN_OOS_POINTS` punktów OOS. `n_oos_points` to suma `n_valid` z foldów
    (None, gdy foldy nie niosą tego klucza). `verdict` to polska etykieta do raportu.

    ML-20: `mean_ic` to średni IC (korelacja rang prognozy z realizacją) po
    foldach, `ic_std` jego odchylenie (próbkowe), `ic_t_stat = mean_ic / ic_std ·
    √n_folds`. None, gdy foldy nie niosą `ic`, a sd/t-stat także przy jednym
    foldzie albo zerowym rozrzucie. Przy 4 foldach t-stat jest mało informatywny —
    czytaj go razem z `n_folds`.

    ML-10: `mean_drift_baseline_rmse` / `mean_llm_baseline_rmse` to średnie RMSE
    baseline'ów „dryf średniej" i „sam LLM" (None, gdy foldy ich nie niosą).
    `beats_baseline` wymaga pobicia najsilniejszego z dostępnych baseline'ów;
    `improvement_pct` zostaje liczone względem „zwrot = 0" (jak w karcie modelu).

    WIA-50: `turnover` to średni obrót foldów (skala ML-20: odwrócenie = 1,
    wejście/wyjście = 0.5), a `mean_net_return` to średni zwrot strategii
    „pozycja = znak prognozy" na wiersz po koszcie obrotu. None, gdy foldy nie
    niosą `sign_strategy_return` i `turnover`. Koszt nie rusza werdyktu RMSE.
    """

    n_folds: int
    mean_candidate_rmse: float
    mean_baseline_rmse: float
    mean_hit_rate: float
    improvement_pct: float
    beats_baseline: bool
    verdict: str
    n_oos_points: int | None = None
    mean_ic: float | None = None
    ic_std: float | None = None
    ic_t_stat: float | None = None
    mean_drift_baseline_rmse: float | None = None
    mean_llm_baseline_rmse: float | None = None
    mean_net_return: float | None = None
    turnover: float | None = None
    paired_z: float | None = None


def paired_loss_z(folds: Sequence[Mapping[str, float]], prefix: str = "") -> float | None:
    """EXP-04: statystyka z sparowanej różnicy strat kwadratowych po wszystkich
    punktach OOS foldów (klucze `{prefix}sum_loss_diff`, `{prefix}sum_loss_diff_sq`,
    `n_valid`). None, gdy foldy nie niosą sum albo jest mniej niż 2 punkty;
    zerowy rozrzut daje ±inf (albo 0 przy zerowej średniej)."""
    sum_key, sq_key = f"{prefix}sum_loss_diff", f"{prefix}sum_loss_diff_sq"
    if not folds or not all(
        sum_key in fold and sq_key in fold and "n_valid" in fold for fold in folds
    ):
        return None
    n = sum(fold["n_valid"] for fold in folds)
    if n < 2:
        return None
    mean = sum(fold[sum_key] for fold in folds) / n
    variance = (sum(fold[sq_key] for fold in folds) - n * mean**2) / (n - 1)
    if variance <= 0:
        return math.copysign(math.inf, mean) if mean else 0.0
    return mean / math.sqrt(variance / n)


def beats_baseline(candidate_rmse: float, baseline_rmse: float) -> bool:
    """Czy kandydat bije baseline persystencji o co najmniej `MIN_REL_IMPROVEMENT`.

    Bezbłędny baseline (`baseline_rmse == 0`, zwrot zawsze 0) jest nie do pobicia —
    także przez kandydata z RMSE 0, który niczego się nie nauczył.
    """
    if baseline_rmse <= 0:
        return False
    return (baseline_rmse - candidate_rmse) / baseline_rmse >= MIN_REL_IMPROVEMENT


def purged_train_end(
    timestamps: Sequence[datetime], valid_start: int, horizon: timedelta = LABEL_HORIZON
) -> int:
    """Koniec (wyłącznie) treningu foldu walidowanego od wiersza `valid_start`.

    `timestamps` są posortowane rosnąco — w modelu wspólnym po czasie GLOBALNYM
    (wszystkie symbole razem), w ewaluacji per symbol i w backteście po czasie
    symbolu. Z treningu wypadają wiersze, których etykieta dojrzewa po starcie
    walidacji: w modelu wspólnym AAPL z 10:00 w treningu i MSFT z 10:01 w walidacji
    dzielą ten sam ruch rynku — to przeciek przekrojowy (purge). Etykieta dojrzała
    dokładnie w chwili startu walidacji jest dozwolona.
    """
    cutoff = timestamps[valid_start] - horizon
    return bisect_right(timestamps, cutoff, 0, valid_start)


# ML-26: embargo — dodatkowa przerwa między końcem etykiety wiersza treningu
# a startem walidacji. Walk-forward trenuje wyłącznie na przeszłości foldu, więc
# przeciek przyszłość→trening (powód embarga u López de Prado) tu nie występuje;
# przeciek etykiet zamyka sam purge po końcu etykiety. Stała jest jawna, żeby
# zmiana szerokości była decyzją w jednym miejscu, nie strojeniem pod wynik.
PURGE_EMBARGO = timedelta(0)


def label_end_times(
    prediction_ts: Sequence[datetime],
    settled_at: Sequence[datetime | None],
    horizon: timedelta = LABEL_HORIZON,
) -> list[datetime]:
    """ML-26: chwila końca etykiety każdego wiersza — rozliczenie (`settled_at`),
    ale nigdy wcześniej niż chwila + `horizon`.

    Brak rozliczenia = stały horyzont (jak dotąd). Migracja 029 wpisała historii
    dolną granicę `settled_at = timestamp`; taki koniec skróciłby etykietę poniżej
    horyzontu i osłabił purge, więc bierzemy późniejszą z dwóch chwil."""
    if len(prediction_ts) != len(settled_at):
        raise ValueError("settled_at must align with the prediction timestamps")
    return [
        stamp + horizon if settled is None else max(settled, stamp + horizon)
        for stamp, settled in zip(prediction_ts, settled_at, strict=True)
    ]


def purge_training_indices(
    prediction_ts: Sequence[datetime],
    label_end_ts: Sequence[datetime],
    valid_start: int,
    embargo: timedelta = PURGE_EMBARGO,
) -> tuple[int, ...]:
    """ML-26: indeksy wierszy treningu foldu walidowanego od wiersza `valid_start`.

    Trening to wiersze przed `valid_start`, których etykieta kończy się najpóźniej
    `embargo` przed startem walidacji (`prediction_ts[valid_start]`); koniec
    dokładnie w tej chwili jest dozwolony. Wynik bywa nieciągły: późne rozliczenie
    (weekend) wyrzuca wiersz, a późniejszy wiersz z dojrzałą etykietą zostaje.
    `prediction_ts` posortowane rosnąco (czas globalny w modelu wspólnym). Ze
    stałym horyzontem (`label_end_times` bez rozliczeń) i zerowym embargiem to
    dokładnie prefiks `purged_train_end`."""
    cutoff = prediction_ts[valid_start] - embargo
    return tuple(index for index in range(valid_start) if label_end_ts[index] <= cutoff)


@dataclass(frozen=True)
class LabelOverlap:
    """ML-25: pomiar nakładania etykiet przed decyzją o purge'u po `settled_at`.

    `measured` — wiersze z chwilą rozliczenia i z następnym wierszem tego samego
    symbolu; `overlapping` — te z nich, których etykieta dojrzała dopiero PO
    chwili następnego wiersza (`settled_at` > jego `timestamp`)."""

    measured: int
    overlapping: int


def count_label_overlaps(
    rows: Iterable[tuple[str, datetime, datetime | None]],
) -> LabelOverlap:
    """Liczy wiersze `(symbol, timestamp, settled_at)` rozliczone po następnym
    wierszu swojego symbolu. Kolejność wejścia nie ma znaczenia; wiersz bez
    `settled_at` albo ostatni w symbolu nie jest mierzony."""
    by_symbol: dict[str, list[tuple[datetime, datetime | None]]] = {}
    for symbol, stamp, settled_at in rows:
        by_symbol.setdefault(symbol, []).append((stamp, settled_at))
    measured = overlapping = 0
    for series in by_symbol.values():
        series.sort(key=lambda item: item[0])
        for (_, settled_at), (next_stamp, _) in zip(series, series[1:], strict=False):
            if settled_at is None:
                continue
            measured += 1
            if settled_at > next_stamp:
                overlapping += 1
    return LabelOverlap(measured=measured, overlapping=overlapping)


def _oos_points(folds: Sequence[Mapping[str, float]]) -> int | None:
    if not all("n_valid" in fold for fold in folds):
        return None
    return int(sum(fold["n_valid"] for fold in folds))


def _mean_of(folds: Sequence[Mapping[str, float]], key: str) -> float | None:
    """Średnia `key` po foldach albo None, gdy choć jeden fold go nie niesie."""
    if not all(key in fold for fold in folds):
        return None
    return statistics.fmean(fold[key] for fold in folds)


def strongest_baseline(
    zero_rmse: float, drift_rmse: float | None, llm_rmse: float | None
) -> tuple[str, float]:
    """(etykieta, RMSE) najtrudniejszego do pobicia baseline'u — najniższe RMSE.

    Brakujący baseline (None) nie startuje; przy remisie wygrywa wcześniejszy
    na liście (zero, dryf, LLM), więc bez dodatkowych baseline'ów to „zwrot 0".
    """
    options = [(ZERO_BASELINE_LABEL, zero_rmse)]
    if drift_rmse is not None:
        options.append((DRIFT_BASELINE_LABEL, drift_rmse))
    if llm_rmse is not None:
        options.append((LLM_BASELINE_LABEL, llm_rmse))
    return min(options, key=lambda option: option[1])


def _verdict(
    mean_candidate: float,
    binding_label: str,
    binding_rmse: float,
    *,
    passed: bool,
    improvement_pct: float,
) -> str:
    """Polska etykieta werdyktu względem baseline'u, który rozstrzygnął bramkę."""
    if binding_label == ZERO_BASELINE_LABEL:
        if passed:
            return f"model bije baseline o {improvement_pct:.1f}%"
        if mean_candidate < binding_rmse:
            return (
                f"model nie bije baseline (poprawa {improvement_pct:.1f}% "
                f"< marginesu {MIN_REL_IMPROVEMENT:.0%})"
            )
        return "model nie bije baseline"
    binding_pct = (binding_rmse - mean_candidate) / binding_rmse * 100 if binding_rmse else 0.0
    if passed:
        return f"model bije najsilniejszy baseline „{binding_label}” o {binding_pct:.1f}%"
    if mean_candidate < binding_rmse:
        return (
            f"model nie bije baseline'u „{binding_label}” (poprawa {binding_pct:.1f}% "
            f"< marginesu {MIN_REL_IMPROVEMENT:.0%})"
        )
    return f"model nie bije baseline'u „{binding_label}”"


def _ic_stats(
    folds: Sequence[Mapping[str, float]],
) -> tuple[float | None, float | None, float | None]:
    """(średni IC, sd, t-stat) po foldach; None tam, gdzie miara niezdefiniowana."""
    if not all("ic" in fold for fold in folds):
        return None, None, None
    ics = [fold["ic"] for fold in folds]
    mean_ic = statistics.fmean(ics)
    if len(ics) < 2:
        return mean_ic, None, None
    ic_std = statistics.stdev(ics)
    if ic_std == 0:
        return mean_ic, ic_std, None
    return mean_ic, ic_std, mean_ic / ic_std * math.sqrt(len(ics))


def _net_strategy_stats(
    folds: Sequence[Mapping[str, float]], cost_bps: float
) -> tuple[float | None, float | None]:
    """(średni zwrot netto strategii znakowej, średni turnover) albo (None, None).

    Turnover foldu (ML-20) liczy odwrócenie jako 1, czyli połowę obróconego
    kapitału, więc koszt wiersza to `trading_cost(2 × turnover, cost_bps)`.
    """
    keys = ("sign_strategy_return", "turnover")
    if not all(key in fold for fold in folds for key in keys):
        return None, None
    net = statistics.fmean(
        fold["sign_strategy_return"] - trading_cost(2.0 * fold["turnover"], cost_bps)
        for fold in folds
    )
    return net, statistics.fmean(fold["turnover"] for fold in folds)


def summarize_folds(
    folds: Sequence[Mapping[str, float]], *, cost_bps: float = 0.0
) -> BacktestSummary:
    """Agreguje per-fold metryki walk-forward w `BacktestSummary`.

    Każdy fold to mapa z kluczami `candidate_rmse`, `baseline_rmse`, `hit_rate`
    i opcjonalnie `n_valid`, `drift_baseline_rmse`, `llm_baseline_rmse` (kształt
    z `MLPredictionPort.walk_forward_report`).
    Średnie są arytmetyczne; `improvement_pct = (baseline - candidate) / baseline
    * 100` (0.0 gdy baseline == 0, by uniknąć dzielenia przez zero). Werdykt
    stosuje `beats_baseline` wobec najsilniejszego baseline'u (ML-10) i — gdy
    foldy niosą `n_valid` — `MIN_OOS_POINTS`. `cost_bps` (WIA-50, koszt
    jednostronny) wchodzi tylko do `mean_net_return`.
    Pusta sekwencja zwraca sentinel „brak danych".
    """
    if not folds:
        return BacktestSummary(0, 0, 0, 0, 0, False, "brak danych")

    n = len(folds)
    mean_candidate = sum(f["candidate_rmse"] for f in folds) / n
    mean_baseline = sum(f["baseline_rmse"] for f in folds) / n
    mean_hit_rate = sum(f["hit_rate"] for f in folds) / n

    if mean_baseline == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = (mean_baseline - mean_candidate) / mean_baseline * 100

    n_oos_points = _oos_points(folds)
    mean_ic, ic_std, ic_t_stat = _ic_stats(folds)
    mean_drift = _mean_of(folds, "drift_baseline_rmse")
    mean_llm = _mean_of(folds, "llm_baseline_rmse")
    binding_label, binding_rmse = strongest_baseline(mean_baseline, mean_drift, mean_llm)
    passed = beats_baseline(mean_candidate, binding_rmse)
    mean_net_return, turnover = _net_strategy_stats(folds, cost_bps)
    paired_z = paired_loss_z(folds, _PAIRED_PREFIX[binding_label])

    if n_oos_points is not None and n_oos_points < MIN_OOS_POINTS:
        passed = False
        verdict = f"za mało punktów OOS ({n_oos_points} < {MIN_OOS_POINTS})"
    elif passed and paired_z is not None and paired_z <= PAIRED_Z_CRIT:
        passed = False
        verdict = (
            f"model nie bije baseline'u „{binding_label}” istotnie "
            f"(test sparowany z={paired_z:.1f} ≤ {PAIRED_Z_CRIT:.1f})"
        )
    else:
        verdict = _verdict(
            mean_candidate,
            binding_label,
            binding_rmse,
            passed=passed,
            improvement_pct=improvement_pct,
        )

    return BacktestSummary(
        n_folds=n,
        mean_candidate_rmse=mean_candidate,
        mean_baseline_rmse=mean_baseline,
        mean_hit_rate=mean_hit_rate,
        improvement_pct=improvement_pct,
        beats_baseline=passed,
        verdict=verdict,
        n_oos_points=n_oos_points,
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
        mean_drift_baseline_rmse=mean_drift,
        mean_llm_baseline_rmse=mean_llm,
        mean_net_return=mean_net_return,
        turnover=turnover,
        paired_z=paired_z,
    )
