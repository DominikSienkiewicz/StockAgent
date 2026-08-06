"""Render sekcji "Karta kondycji modelu" w raporcie e-mail (roadmap #12).

Repo shipuje model tylko, gdy kandydat bije naiwny baseline persystencji
("zwrot=0"), ale ta bramka jest dla użytkownika NIEWIDZIALNA. Ta sekcja zdejmuje
z niej welon: pokazuje procentową poprawę względem baseline'u, świeżość modelu
(badge STALE), rozkład przebiegu treningu per-symbol oraz — gdy kandydat został
odrzucony — uczciwie przyznaje, że jedziemy na starym, sprawdzonym modelu.

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


def _reject_notice(summary: RunSummary) -> str:
    """Jawny komunikat o odrzuconych kandydatach albo "" gdy nic nie odrzucono.

    Gdy WSZYSTKO odrzucono (accepted == 0) — jedziemy na sprawdzonym modelu.
    Gdy odrzuty są częściowe — część symboli zostaje na sprawdzonym modelu.
    """
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
) -> str:
    """Renderuje sekcję HTML karty kondycji modelu lub "" gdy brak danych.

    `now` jest wstrzykiwane — renderer nie woła zegara. Metryki kandydata są
    opcjonalne (przebieg z samymi odrzutami/pominięciami może ich nie mieć).
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

    reject_notice = _reject_notice(summary)
    if reject_notice:
        rows.append(
            f"<div style='font-size: 12px; color: {_COLOR_POOR}; font-weight: 600; "
            f"margin-bottom: 4px;'>⚠️ {_html.escape(reject_notice)}</div>"
        )

    if candidate_rmse is not None and baseline_rmse is not None:
        pct, color = _improvement(candidate_rmse, baseline_rmse)
        rows.append(
            f"<div style='font-size: 12px; color: #374151; margin-bottom: 4px;'>"
            f"RMSE kandydata <strong>{candidate_rmse:.4f}</strong> vs baseline "
            f"<strong>{baseline_rmse:.4f}</strong> "
            f"(<span style='color: {color}; font-weight: 700;'>{pct:+.1f}%</span>)"
            f"</div>"
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

    reject_notice = _reject_notice(summary)
    if reject_notice:
        lines.append(f"  ⚠ {reject_notice}")

    if candidate_rmse is not None and baseline_rmse is not None:
        pct, _ = _improvement(candidate_rmse, baseline_rmse)
        lines.append(
            f"  RMSE kandydata {candidate_rmse:.4f} vs baseline "
            f"{baseline_rmse:.4f} ({pct:+.1f}%)"
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
    """Widok sekcji: zagregowany przebieg + metryki ostatniego kandydata."""

    summary: RunSummary
    trained_at: datetime
    candidate_rmse: float | None = None
    baseline_rmse: float | None = None
    directional_hit_rate: float | None = None
    folds: int | None = None


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def build_scorecard_view(rows: list[dict[str, Any]]) -> ModelScorecardView | None:
    """Składa widok z surowych wierszy `model_scorecards` (najnowsze pierwsze).

    Agreguje CAŁY przebieg (trening leci per-symbol, ~43 wiersze na run), a
    metryki bierze z najnowszego wiersza, który je niesie. Odrzucony kandydat
    nie musi mieć metryk — sekcja i tak renderuje rozbicie przebiegu.
    Brak wierszy albo brak dającej się sparsować daty → None (sekcja się chowa).
    """
    if not rows:
        return None
    stamps = [ts for ts in (_parse_timestamp(r.get("timestamp")) for r in rows) if ts]
    if not stamps:
        return None
    summary = summarize_run([str(r.get("status", "")) for r in rows])
    metrics = next(
        (r for r in rows if r.get("candidate_holdout_rmse") is not None), None
    )
    return ModelScorecardView(
        summary=summary,
        trained_at=max(stamps),
        candidate_rmse=metrics.get("candidate_holdout_rmse") if metrics else None,
        baseline_rmse=metrics.get("baseline_rmse") if metrics else None,
        directional_hit_rate=(
            metrics.get("candidate_holdout_directional_hit_rate") if metrics else None
        ),
        folds=metrics.get("n_folds") if metrics else None,
    )
