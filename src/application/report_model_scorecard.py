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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.domain.model_scorecard import (
    METRIC_NOTE,
    RunSummary,
    improvement_pct,
    is_stale,
    latest_run_indices,
    summarize_run,
)

_HEADING = "🩺 Karta kondycji modelu"

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
) -> str:
    """Renderuje sekcję HTML karty kondycji modelu lub "" gdy brak danych.

    `now` jest wstrzykiwane — renderer nie woła zegara. Metryki kandydata są
    opcjonalne (przebieg z samymi odrzutami/pominięciami może ich nie mieć);
    `metrics_symbol` i `n_samples` podpisują, czyje to metryki.
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
        + f"Przebieg: {_html.escape(summary.summary_line)}</div>",
    ]

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
            f"RMSE kandydata <strong>{candidate_rmse:.4f}</strong> vs baseline "
            f"<strong>{baseline_rmse:.4f}</strong> "
            f"(<span style='color: {color}; font-weight: 700;'>{pct:+.1f}%</span>)"
            f"{provenance_html}</div>"
        )

    metric_bits: list[str] = []
    if directional_hit_rate is not None:
        metric_bits.append(
            f"trafność kierunkowa <strong>{directional_hit_rate * 100:.0f}%</strong>"
        )
    if folds is not None:
        metric_bits.append(f"foldy walk-forward <strong>{folds}</strong>")
    if metric_bits:
        rows.append(
            f"<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            f"{' · '.join(metric_bits)}</div>"
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
) -> str:
    """Renderuje sekcję plain-text karty kondycji modelu lub "" gdy brak danych."""
    if summary is None:
        return ""

    stale_suffix = "  [STALE]" if is_stale(trained_at, now) else ""
    lines: list[str] = [
        "KARTA KONDYCJI MODELU",
        "-" * 64,
        f"  Ostatni trening: {_fmt_date(trained_at)}{stale_suffix}",
        f"  Przebieg: {summary.summary_line}",
    ]

    notice = _model_notice(summary, model_loaded=model_loaded)
    if notice:
        lines.append(f"  ⚠ {notice}")

    if candidate_rmse is not None and baseline_rmse is not None:
        pct, _ = _improvement(candidate_rmse, baseline_rmse)
        provenance = _provenance(metrics_symbol, n_samples)
        lines.append(
            f"  RMSE kandydata {candidate_rmse:.4f} vs baseline "
            f"{baseline_rmse:.4f} ({pct:+.1f}%)"
            + (f" — {provenance}" if provenance else "")
        )

    metric_bits: list[str] = []
    if directional_hit_rate is not None:
        metric_bits.append(f"trafność kierunkowa {directional_hit_rate * 100:.0f}%")
    if folds is not None:
        metric_bits.append(f"foldy walk-forward {folds}")
    if metric_bits:
        lines.append(f"  {'  ·  '.join(metric_bits)}")

    lines.append(f"  Uwaga: {METRIC_NOTE}")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class ModelScorecardView:
    """Widok sekcji: zagregowany przebieg + metryki ostatniego kandydata.

    `metrics_symbol` / `metrics_n_samples` mówią, z którego wiersza pochodzą
    metryki (np. `ALL` — model wspólny) i na ilu próbkach go oceniono."""

    summary: RunSummary
    trained_at: datetime
    candidate_rmse: float | None = None
    baseline_rmse: float | None = None
    directional_hit_rate: float | None = None
    folds: int | None = None
    metrics_symbol: str | None = None
    metrics_n_samples: int | None = None


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

    Agreguje wyłącznie NAJNOWSZY przebieg (`latest_run_indices`, ML-03) — okno
    odczytu niesie kilka tygodniowych przebiegów, które dawniej się sklejały.
    Metryki bierze z najnowszego wdrożonego kandydata tego przebiegu (albo z
    najnowszego wiersza z metrykami) i podpisuje je symbolem oraz liczbą próbek.
    Brak wierszy albo brak dającej się sparsować daty → None (sekcja się chowa).
    """
    stamps = [_parse_timestamp(r.get("timestamp")) for r in rows]
    run: list[tuple[datetime, dict[str, Any]]] = []
    for idx in latest_run_indices(stamps):
        stamp = stamps[idx]
        if stamp is not None:
            run.append((stamp, rows[idx]))
    if not run:
        return None
    summary = summarize_run([str(row.get("status", "")) for _, row in run])
    metrics = _newest_metrics_row(run)
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
    )
