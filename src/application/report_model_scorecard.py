"""Render sekcji "Karta kondycji modelu" w raporcie e-mail (roadmap #12).

Repo shipuje model tylko, gdy kandydat bije naiwny baseline persystencji
("zwrot=0"), ale ta bramka jest dla użytkownika NIEWIDZIALNA. Ta sekcja zdejmuje
z niej welon: pokazuje procentową poprawę względem baseline'u, świeżość modelu
(badge STALE), rozkład przebiegu treningu per-symbol oraz — gdy kandydat został
odrzucony — uczciwie przyznaje, że jedziemy na starym, sprawdzonym modelu. Gdy
żadnego modelu nie ma (cold-start), mówi to wprost: noga ML to baseline.

Czysta logika prezentacji — zero I/O, zero płatnych wywołań. Cała reguła biznesu
(poprawa RMSE, próg świeżości, agregacja outcome'ów) żyje w domenie
(`src.domain.model_scorecard`); ten moduł tylko formatuje jej wynik.

Sekcja samosupresująca: brak danych (`summary is None`) → `render_*` zwraca "".

Uczciwe copy jest wymogiem, nie ozdobą: sekcja "trust" bez zastrzeżenia
`METRIC_NOTE` sama byłaby nieuczciwa (metryki dotyczą kandydata na foldach
walk-forward, a finalny model jest refitowany na całości). Dlatego notę
IMPORTUJEMY z domeny i renderujemy dosłownie — nie przepisujemy jej treści.
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.application.data_coverage import coverage_line_pl
from src.application.report_formatting import number as _number
from src.application.report_formatting import pct as _pct
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.backtest import (
    DRIFT_BASELINE_LABEL,
    LLM_BASELINE_LABEL,
    ZERO_BASELINE_LABEL,
    strongest_baseline,
)
from src.domain.coverage import CoverageReport
from src.domain.drift import PSI_DRIFT_THRESHOLD, FeatureDrift, drifted_features
from src.domain.model_scorecard import (
    FREE_SNAPSHOT_MODEL_SYMBOL,
    LLM_SNAPSHOT_MODEL_SYMBOL,
    METRIC_NOTE,
    POOLED_MODEL_SYMBOL,
    SNAPSHOT_MODEL_SYMBOLS,
    EvaluationSummary,
    RunSummary,
    TrainingOutcome,
    classify_status,
    improvement_pct,
    is_stale,
    latest_run_indices_by_id,
    summarize_evaluations,
    summarize_run,
)

_HEADING = "🩺 Karta kondycji modelu"

# ML-52: modele z `feature_snapshots` w kolejności linii i ich etykiety w mailu.
_SNAPSHOT_MODEL_LABELS: Mapping[str, str] = {
    FREE_SNAPSHOT_MODEL_SYMBOL: "cechy darmowe",
    LLM_SNAPSHOT_MODEL_SYMBOL: "cechy LLM (cykle gated)",
}
_SNAPSHOT_OUTCOME_PL = {
    TrainingOutcome.ACCEPTED: "bije baseline",
    TrainingOutcome.EVALUATION_PASSED: "bije baseline",
    TrainingOutcome.REJECTED: "nie bije baseline'u",
    TrainingOutcome.EVALUATION_FAILED: "nie bije baseline'u",
    TrainingOutcome.SKIPPED: "pominięty (za mało próbek)",
}

# Kolory poprawki względem baseline'u — spójne z resztą raportu.
_COLOR_GOOD = "#16a34a"
_COLOR_POOR = "#dc2626"
_COLOR_MUTED = "#6b7280"

# Badge STALE — bursztynowy, jak ostrzeżenia w innych sekcjach.
_STALE_TEXT_COLOR = "#b45309"
_STALE_BG = "#fef3c7"


def _fmt_date(moment: datetime) -> str:
    """Data treningu bez czasu — szczegół zegara jest tu szumem."""
    return moment.strftime("%Y-%m-%d")


def _improvement(candidate_rmse: float, baseline_rmse: float) -> tuple[float, str]:
    """Zwraca (procent poprawy, kolor) dla RMSE kandydata vs baseline.

    Dodatni procent = kandydat lepszy (zielony), ujemny = gorszy (czerwony).
    Nie chowamy ujemnej wartości — brutalna prawda, nie udawany sukces.
    """
    pct = improvement_pct(candidate_rmse, baseline_rmse)
    return pct, (_COLOR_GOOD if pct >= 0 else _COLOR_POOR)


_COLD_START_NOTICE = (
    "Brak wytrenowanego modelu — noga ML jedzie na baseline'ie „cena bez zmian” (cold-start)."
)


def _provenance(metrics_symbol: str | None, n_samples: int | None) -> str:
    """Podpis metryk: którego kandydata dotyczą i na ilu próbkach (ML-03)."""
    bits: list[str] = []
    if metrics_symbol:
        bits.append(metrics_symbol)
    if n_samples is not None:
        bits.append(f"n={n_samples}")
    return ", ".join(bits)


def _baselines_line(
    baseline_rmse: float | None,
    drift_baseline_rmse: float | None,
    llm_baseline_rmse: float | None,
) -> str:
    """ML-10: RMSE trzech baseline'ów bramki z najsilniejszym na końcu, albo ""
    gdy wiersz karty ich nie ma (sprzed migracji 028 / zapis bez flagi)."""
    if baseline_rmse is None or (drift_baseline_rmse is None and llm_baseline_rmse is None):
        return ""
    labelled = [
        (label, value)
        for label, value in (
            (ZERO_BASELINE_LABEL, baseline_rmse),
            (DRIFT_BASELINE_LABEL, drift_baseline_rmse),
            (LLM_BASELINE_LABEL, llm_baseline_rmse),
        )
        if value is not None
    ]
    strongest, _ = strongest_baseline(baseline_rmse, drift_baseline_rmse, llm_baseline_rmse)
    values = " · ".join(f"{label} {_number(value, decimals=4)}" for label, value in labelled)
    return f"Baseline'y bramki (RMSE): {values} — najsilniejszy: {strongest}"


def _drift_line(drifts: tuple[FeatureDrift, ...]) -> str:
    """ML-21: ostrzeżenie o cechach z istotnym dryfem albo "" gdy go nie ma.

    PSI liczy trening: ostatnie 30 dni próbki względem jej starszej części (bez
    `feature_snapshots` rozkład samego servingu jest nieznany)."""
    if not drifts:
        return ""
    listed = " · ".join(f"{d.feature} {_number(d.psi, decimals=2)}" for d in drifts)
    threshold = _number(PSI_DRIFT_THRESHOLD, decimals=2)
    return f"Dryf cech (PSI > {threshold}, ostatnie 30 dni próbki vs starsza część): {listed}"


def _verdict_label(evaluations: EvaluationSummary | None) -> str:
    """Etykieta werdyktu: z podziałem na model wspólny albo zbiorczy przebieg (legacy)."""
    if evaluations is None:
        return "Przebieg"
    return f"Przebieg — model wspólny ({POOLED_MODEL_SYMBOL})"


def _evaluations_line(evaluations: EvaluationSummary | None) -> str:
    """Linia ewaluacji per symbol albo "" gdy przebieg ich nie ma."""
    if evaluations is None or evaluations.total == 0:
        return ""
    return f"Ewaluacje per symbol (bez zapisu wag): {evaluations.summary_line}"


def _model_notice(summary: RunSummary, *, model_loaded: bool) -> str:
    """Jawny komunikat o stanie modelu albo "" gdy nie ma czego zgłaszać.

    Brak załadowanego modelu — cold-start, niezależnie od wyniku przebiegu
    (odrzucony kandydat nie zostawia żadnego "sprawdzonego" modelu).
    Gdy WSZYSTKO odrzucono (accepted == 0) — jedziemy na sprawdzonym modelu.
    Gdy odrzuty są częściowe — część symboli zostaje na sprawdzonym modelu.
    """
    if not model_loaded:
        if summary.rejected == 0:
            return _COLD_START_NOTICE
        return f"Nowy model odrzucony. {_COLD_START_NOTICE}"
    if summary.rejected == 0:
        return ""
    if summary.accepted == 0:
        return "Nowy model odrzucony — jedziemy na sprawdzonym modelu z poprzedniego treningu."
    return (
        f"Część kandydatów odrzucona ({summary.rejected}) — te symbole zostają "
        "na sprawdzonym modelu."
    )


def render_model_scorecard_html(
    summary: RunSummary | None,
    *,
    trained_at: datetime,
    now: datetime,
    candidate_rmse: float | None = None,
    baseline_rmse: float | None = None,
    directional_hit_rate: float | None = None,
    folds: int | None = None,
    model_loaded: bool = True,
    metrics_symbol: str | None = None,
    n_samples: int | None = None,
    evaluations: EvaluationSummary | None = None,
    drift_baseline_rmse: float | None = None,
    llm_baseline_rmse: float | None = None,
    coverage: CoverageReport | None = None,
    drifted_features: tuple[FeatureDrift, ...] = (),
    snapshot_models: tuple[SnapshotModelCard, ...] = (),
) -> str:
    """Renderuje sekcję HTML karty kondycji modelu lub "" gdy brak danych.

    `now` jest wstrzykiwane — renderer nie woła zegara. Metryki kandydata są
    opcjonalne (przebieg z samymi odrzutami/pominięciami może ich nie mieć);
    `metrics_symbol` i `n_samples` podpisują, czyje to metryki. Z `evaluations`
    `summary` to werdykt modelu wspólnego, a ewaluacje per symbol idą osobną linią.
    `coverage` (DAT-16) dokłada zdanie o pokryciu danych z symbolami bez rozliczeń.
    """
    if summary is None:
        return ""

    stale_badge = ""
    if is_stale(trained_at, now):
        stale_badge = (
            f"<span style='display: inline-block; margin-left: 8px; padding: 1px 6px; "
            f"background: {_STALE_BG}; color: {_STALE_TEXT_COLOR}; border-radius: 3px; "
            f"font-size: 11px; font-weight: 700;'>STALE</span>"
        )

    rows: list[str] = [
        "<div style='font-size: 13px; margin-bottom: 4px;'>"
        + f"Ostatni trening: <strong>{_html.escape(_fmt_date(trained_at))}</strong>"
        + f"{stale_badge}</div>",
        "<div style='font-size: 13px; color: #374151; margin-bottom: 4px;'>"
        + f"{_html.escape(_verdict_label(evaluations))}: "
        + f"{_html.escape(summary.summary_line)}</div>",
    ]
    evaluations_line = _evaluations_line(evaluations)
    if evaluations_line:
        rows.append(
            "<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            + f"{_html.escape(evaluations_line)}</div>"
        )
    snapshot_line = _snapshot_models_line(snapshot_models)
    if snapshot_line:
        rows.append(
            "<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            + f"{_html.escape(snapshot_line)}</div>"
        )

    notice = _model_notice(summary, model_loaded=model_loaded)
    if notice:
        rows.append(
            f"<div style='font-size: 12px; color: {_COLOR_POOR}; font-weight: 600; "
            f"margin-bottom: 4px;'>⚠️ {_html.escape(notice)}</div>"
        )

    if candidate_rmse is not None and baseline_rmse is not None:
        pct, color = _improvement(candidate_rmse, baseline_rmse)
        provenance = _provenance(metrics_symbol, n_samples)
        provenance_html = f" — {_html.escape(provenance)}" if provenance else ""
        rows.append(
            f"<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            f"RMSE kandydata <strong>{_number(candidate_rmse, decimals=4)}</strong> vs baseline "
            f"<strong>{_number(baseline_rmse, decimals=4)}</strong> "
            f"(<span style='color: {color}; font-weight: 700;'>"
            f"{_number(pct, decimals=1, signed=True)}%</span>)"
            f"{provenance_html}</div>"
        )

    baselines = _baselines_line(baseline_rmse, drift_baseline_rmse, llm_baseline_rmse)
    if baselines:
        rows.append(
            "<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            + f"{_html.escape(baselines)}</div>"
        )

    metric_bits: list[str] = []
    if directional_hit_rate is not None:
        metric_bits.append(
            f"trafność kierunkowa <strong>{_pct(directional_hit_rate, decimals=0)}</strong>"
        )
    if folds is not None:
        metric_bits.append(f"foldy walk-forward <strong>{folds}</strong>")
    if metric_bits:
        rows.append(
            f"<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            f"{' · '.join(metric_bits)}</div>"
        )

    coverage_line = "" if coverage is None else coverage_line_pl(coverage, list_symbols=True)
    if coverage_line:
        rows.append(
            "<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            + f"{_html.escape(coverage_line)}</div>"
        )

    drift = _drift_line(drifted_features)
    if drift:
        rows.append(
            f"<div style='font-size: 12px; color: {_STALE_TEXT_COLOR}; font-weight: 600; "
            f"margin-bottom: 4px;'>⚠️ {_html.escape(drift)}</div>"
        )

    rows.append(
        f"<div style='font-size: 11px; color: {_COLOR_MUTED}; margin-top: 6px; "
        f"font-style: italic;'>{_html.escape(METRIC_NOTE)}</div>"
    )

    return (
        f"<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>{_HEADING}</h2>"
        "<div style='padding: 10px 14px; background: #f8fafc; "
        "border-left: 3px solid #6366f1; border-radius: 4px; margin-bottom: 20px;'>"
        + "".join(rows)
        + "</div>"
    )


def render_model_scorecard_text(
    summary: RunSummary | None,
    *,
    trained_at: datetime,
    now: datetime,
    candidate_rmse: float | None = None,
    baseline_rmse: float | None = None,
    directional_hit_rate: float | None = None,
    folds: int | None = None,
    model_loaded: bool = True,
    metrics_symbol: str | None = None,
    n_samples: int | None = None,
    evaluations: EvaluationSummary | None = None,
    drift_baseline_rmse: float | None = None,
    llm_baseline_rmse: float | None = None,
    drifted_features: tuple[FeatureDrift, ...] = (),
    snapshot_models: tuple[SnapshotModelCard, ...] = (),
) -> str:
    """Renderuje sekcję plain-text karty kondycji modelu lub "" gdy brak danych.

    `drift_baseline_rmse` / `llm_baseline_rmse` (ML-10) dokładają linię z RMSE
    wszystkich trzech baseline'ów bramki."""
    if summary is None:
        return ""

    stale_suffix = "  [STALE]" if is_stale(trained_at, now) else ""
    lines: list[str] = [
        "KARTA KONDYCJI MODELU",
        "-" * 64,
        f"  Ostatni trening: {_fmt_date(trained_at)}{stale_suffix}",
        f"  {_verdict_label(evaluations)}: {summary.summary_line}",
    ]
    evaluations_line = _evaluations_line(evaluations)
    if evaluations_line:
        lines.append(f"  {evaluations_line}")
    snapshot_line = _snapshot_models_line(snapshot_models)
    if snapshot_line:
        lines.append(f"  {snapshot_line}")

    notice = _model_notice(summary, model_loaded=model_loaded)
    if notice:
        lines.append(f"  ⚠ {notice}")

    if candidate_rmse is not None and baseline_rmse is not None:
        pct, _ = _improvement(candidate_rmse, baseline_rmse)
        provenance = _provenance(metrics_symbol, n_samples)
        lines.append(
            f"  RMSE kandydata {_number(candidate_rmse, decimals=4)} vs baseline "
            f"{_number(baseline_rmse, decimals=4)} ({_number(pct, decimals=1, signed=True)}%)"
            + (f" — {provenance}" if provenance else "")
        )

    baselines = _baselines_line(baseline_rmse, drift_baseline_rmse, llm_baseline_rmse)
    if baselines:
        lines.append(f"  {baselines}")

    metric_bits: list[str] = []
    if directional_hit_rate is not None:
        metric_bits.append(f"trafność kierunkowa {_pct(directional_hit_rate, decimals=0)}")
    if folds is not None:
        metric_bits.append(f"foldy walk-forward {folds}")
    if metric_bits:
        lines.append(f"  {'  ·  '.join(metric_bits)}")

    drift = _drift_line(drifted_features)
    if drift:
        lines.append(f"  ⚠ {drift}")

    lines.append(f"  Uwaga: {METRIC_NOTE}")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class SnapshotModelCard:
    """ML-52: najnowsza karta modelu ewaluowanego na wektorach `feature_snapshots`.

    `symbol` to `FREE_SNAPSHOT_MODEL_SYMBOL` albo `LLM_SNAPSHOT_MODEL_SYMBOL`;
    metryki None, gdy ewaluacja się nie odbyła (za mało próbek)."""

    symbol: str
    outcome: TrainingOutcome
    candidate_rmse: float | None = None
    baseline_rmse: float | None = None
    n_samples: int | None = None


def snapshot_model_cards(rows: list[dict[str, Any]]) -> tuple[SnapshotModelCard, ...]:
    """ML-52: najnowsza karta każdego modelu z `feature_snapshots` w oknie odczytu,
    w kolejności `_SNAPSHOT_MODEL_LABELS` (cechy darmowe, potem LLM). Wiersz bez daty
    dającej się sparsować pomijamy; brak kart → pusta krotka."""
    newest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        symbol = row.get("symbol")
        stamp = _parse_timestamp(row.get("timestamp"))
        if symbol not in SNAPSHOT_MODEL_SYMBOLS or stamp is None:
            continue
        key = str(symbol)
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, row)
    cards: list[SnapshotModelCard] = []
    for symbol in _SNAPSHOT_MODEL_LABELS:
        if symbol not in newest:
            continue
        _, row = newest[symbol]
        cards.append(
            SnapshotModelCard(
                symbol=symbol,
                outcome=classify_status(str(row.get("status", ""))),
                candidate_rmse=row.get("candidate_holdout_rmse"),
                baseline_rmse=row.get("baseline_rmse"),
                n_samples=row.get("n_samples"),
            )
        )
    return tuple(cards)


def _snapshot_card_text(card: SnapshotModelCard) -> str:
    """„cechy darmowe — bije baseline, RMSE 0,0210 vs 0,0240 (+12,5%), n=812”."""
    bits = [_SNAPSHOT_OUTCOME_PL[card.outcome]]
    if card.candidate_rmse is not None and card.baseline_rmse is not None:
        pct = improvement_pct(card.candidate_rmse, card.baseline_rmse)
        bits.append(
            f"RMSE {_number(card.candidate_rmse, decimals=4)} vs "
            f"{_number(card.baseline_rmse, decimals=4)} ({_number(pct, decimals=1, signed=True)}%)"
        )
    if card.n_samples is not None:
        bits.append(f"n={card.n_samples}")
    return f"{_SNAPSHOT_MODEL_LABELS[card.symbol]} — {', '.join(bits)}"


def _snapshot_models_line(cards: tuple[SnapshotModelCard, ...]) -> str:
    """ML-52: jedna linia z kartami modeli z `feature_snapshots` albo "" bez kart."""
    if not cards:
        return ""
    listed = " · ".join(_snapshot_card_text(card) for card in cards)
    return f"Modele z feature_snapshots (ewaluacja, bez zapisu wag): {listed}"


@dataclass(frozen=True)
class ModelScorecardView:
    """Widok sekcji: zagregowany przebieg + metryki ostatniego kandydata.

    `metrics_symbol` / `metrics_n_samples` mówią, z którego wiersza pochodzą
    metryki (np. `ALL` — model wspólny) i na ilu próbkach go oceniono.
    `evaluations` jest ustawione, gdy przebieg ma wiersz modelu wspólnego: wtedy
    `summary` to jego werdykt, a ewaluacje per symbol są liczone osobno. None —
    przebieg sprzed ML-13, `summary` agreguje wszystkie wiersze jak dawniej.
    `drift_baseline_rmse` / `llm_baseline_rmse` (ML-10, migracja 028) pochodzą
    z tego samego wiersza co metryki kandydata; None = wiersz ich nie ma.
    `drifted_features` (ML-21, migracja 052) to cechy z PSI ostatnich 30 dni
    powyżej progu z `feature_stats` tego wiersza; pusta krotka = brak dryfu
    albo wiersz bez statystyk."""

    summary: RunSummary
    trained_at: datetime
    candidate_rmse: float | None = None
    baseline_rmse: float | None = None
    directional_hit_rate: float | None = None
    folds: int | None = None
    metrics_symbol: str | None = None
    metrics_n_samples: int | None = None
    evaluations: EvaluationSummary | None = None
    drift_baseline_rmse: float | None = None
    llm_baseline_rmse: float | None = None
    # DAT-16: pokrycie danych okna track recordu; None = zdanie się chowa.
    coverage: CoverageReport | None = None
    drifted_features: tuple[FeatureDrift, ...] = ()
    # ML-52: karty modeli z `feature_snapshots` (ALL_FREE, ALL_LLM) — jedna linia
    # pod werdyktem; pusta krotka = linia się chowa (dokłada je ReportAssembler).
    snapshot_models: tuple[SnapshotModelCard, ...] = ()


def _feature_drifts(row: Mapping[str, Any] | None) -> tuple[FeatureDrift, ...]:
    """ML-21: cechy z istotnym dryfem z `feature_stats` wiersza karty. Kolumna
    sprzed migracji 052, zapis bez flagi albo zły kształt JSON → brak dryfu."""
    stats = row.get("feature_stats") if row else None
    if not isinstance(stats, Mapping):
        return ()
    psi_by_feature: dict[str, float | None] = {}
    for name, entry in stats.items():
        psi = entry.get("psi_recent_30d") if isinstance(entry, Mapping) else None
        if isinstance(psi, int | float) and not isinstance(psi, bool):
            psi_by_feature[str(name)] = float(psi)
        else:
            psi_by_feature[str(name)] = None
    return drifted_features(psi_by_feature)


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _newest_metrics_row(
    run: list[tuple[datetime, dict[str, Any]]],
) -> dict[str, Any] | None:
    """Najnowszy wdrożony kandydat z metrykami, a gdy go brak — najnowszy wiersz
    z metrykami (odrzucony kandydat też ma swoje RMSE, tylko podpisane)."""
    with_metrics = [
        (stamp, row) for stamp, row in run if row.get("candidate_holdout_rmse") is not None
    ]
    accepted = [
        pair for pair in with_metrics if pair[1].get("status") == "trained_successfully"
    ]
    for candidates in (accepted, with_metrics):
        if candidates:
            return max(candidates, key=lambda pair: pair[0])[1]
    return None


def build_scorecard_view(rows: list[dict[str, Any]]) -> ModelScorecardView | None:
    """Składa widok z surowych wierszy `model_scorecards`.

    Agreguje wyłącznie NAJNOWSZY przebieg (`latest_run_indices_by_id`): wiersze
    z `run_id` (ML-24) grupuje identyfikator, historię bez niego przerwa czasowa
    (ML-03) — okno odczytu niesie kilka przebiegów, które dawniej się sklejały.
    Werdykt główny to wiersz(e) modelu wspólnego `ALL`, a ewaluacje per symbol
    idą do `evaluations` (decyzja 3); bez wiersza `ALL` werdykt agreguje cały
    przebieg jak przed ML-13. Metryki bierze z najnowszego wdrożonego kandydata
    tego przebiegu (albo z najnowszego wiersza z metrykami) i podpisuje je
    symbolem oraz liczbą próbek; przy wierszu `ALL` wyłącznie z niego — RMSE
    ewaluacji per symbol nie może udawać metryki modelu wspólnego.
    Brak wierszy albo brak dającej się sparsować daty → None (sekcja się chowa).
    Karty modeli z `feature_snapshots` (ML-52, `SNAPSHOT_MODEL_SYMBOLS`) to osobne
    modele ewaluacyjne, więc ta karta ich nie liczy.
    """
    rows = [r for r in rows if r.get("symbol") not in SNAPSHOT_MODEL_SYMBOLS]
    stamps = [_parse_timestamp(r.get("timestamp")) for r in rows]
    run_ids = [str(r["run_id"]) if r.get("run_id") else None for r in rows]
    run: list[tuple[datetime, dict[str, Any]]] = []
    for idx in latest_run_indices_by_id(stamps, run_ids):
        stamp = stamps[idx]
        if stamp is not None:
            run.append((stamp, rows[idx]))
    if not run:
        return None
    statuses = [(row.get("symbol"), str(row.get("status", ""))) for _, row in run]
    pooled = [status for symbol, status in statuses if symbol == POOLED_MODEL_SYMBOL]
    evaluations: EvaluationSummary | None = None
    if pooled:
        summary = summarize_run(pooled)
        evaluations = summarize_evaluations(
            status for symbol, status in statuses if symbol != POOLED_MODEL_SYMBOL
        )
    else:
        summary = summarize_run(status for _, status in statuses)
    metrics = _newest_metrics_row(
        [(stamp, row) for stamp, row in run if row.get("symbol") == POOLED_MODEL_SYMBOL]
        if pooled
        else run
    )
    return ModelScorecardView(
        summary=summary,
        trained_at=max(stamp for stamp, _ in run),
        candidate_rmse=metrics.get("candidate_holdout_rmse") if metrics else None,
        baseline_rmse=metrics.get("baseline_rmse") if metrics else None,
        directional_hit_rate=(
            metrics.get("candidate_holdout_directional_hit_rate") if metrics else None
        ),
        folds=metrics.get("n_folds") if metrics else None,
        metrics_symbol=str(metrics["symbol"]) if metrics and metrics.get("symbol") else None,
        metrics_n_samples=metrics.get("n_samples") if metrics else None,
        evaluations=evaluations,
        drift_baseline_rmse=metrics.get("drift_baseline_rmse") if metrics else None,
        llm_baseline_rmse=metrics.get("llm_baseline_rmse") if metrics else None,
        drifted_features=_feature_drifts(metrics),
    )


def _section_html(data: SectionInput) -> str:
    """#12: karta kondycji modelu. Brak danych → "" (sekcja sama się chowa)."""
    model_scorecard = data.context.model_scorecard
    if model_scorecard is None:
        return ""
    return render_model_scorecard_html(
        model_scorecard.summary,
        trained_at=model_scorecard.trained_at,
        now=data.started_at,
        model_loaded=data.context.ml_model_loaded,
        candidate_rmse=model_scorecard.candidate_rmse,
        baseline_rmse=model_scorecard.baseline_rmse,
        directional_hit_rate=model_scorecard.directional_hit_rate,
        folds=model_scorecard.folds,
        metrics_symbol=model_scorecard.metrics_symbol,
        n_samples=model_scorecard.metrics_n_samples,
        evaluations=model_scorecard.evaluations,
        drift_baseline_rmse=model_scorecard.drift_baseline_rmse,
        llm_baseline_rmse=model_scorecard.llm_baseline_rmse,
        coverage=model_scorecard.coverage,
        drifted_features=model_scorecard.drifted_features,
        snapshot_models=model_scorecard.snapshot_models,
    )


def _section_text(data: SectionInput) -> list[str]:
    """MAIL-09: karta kondycji modelu w wersji tekstowej (te same dane co HTML)."""
    model_scorecard = data.context.model_scorecard
    if model_scorecard is None:
        return []
    text = render_model_scorecard_text(
        model_scorecard.summary,
        trained_at=model_scorecard.trained_at,
        now=data.started_at,
        model_loaded=data.context.ml_model_loaded,
        candidate_rmse=model_scorecard.candidate_rmse,
        baseline_rmse=model_scorecard.baseline_rmse,
        directional_hit_rate=model_scorecard.directional_hit_rate,
        folds=model_scorecard.folds,
        metrics_symbol=model_scorecard.metrics_symbol,
        n_samples=model_scorecard.metrics_n_samples,
        evaluations=model_scorecard.evaluations,
        drift_baseline_rmse=model_scorecard.drift_baseline_rmse,
        llm_baseline_rmse=model_scorecard.llm_baseline_rmse,
        snapshot_models=model_scorecard.snapshot_models,
    )
    return text_block(text)


# #12: tuż za Track Recordem i Historią trafności — ta sama oś zaufania.
MODEL_SCORECARD_SECTION = ReportSection(
    key="model_scorecard",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"model_scorecard", "ml_model_loaded"}),
)
