"""NIEZ-33 — publiczna strona statusu i JSON odznaki „last digest” (czyste funkcje).

Stan systemu znał dotąd tylko właściciel z UI Actions, a brak maila wygląda jak spokojny
dzień. Strona mówi z historii przebiegów (`cycle_runs`, NIEZ-29): czy Fast Loop działa,
jest świadomie wstrzymany, czy spóźniony (ta sama reguła co deadman, `job_state`), kiedy
ruszył ostatnio i czy mail wyszedł, kiedy ostatnio wyszedł digest, ile płatnych wywołań
poszło w 7 i 30 dni i jaki odsetek cykli był darmowy. „Ostatni przebieg 60 dni temu” jest
uczciwe, choć niekorzystne (decyzja 1).

Inwarianty jak scorecard: ZERO symboli, adresów i kwot — `CycleRun` niesie same liczby,
koszt to liczba płatnych wywołań, nie dolary. CSS bez reguł z małpą (anty-PII: brak „@”).
Odznaka to endpoint JSON shields.io (`schemaVersion`, `label`, `message`, `color`).
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.application.ports import CycleRun
from src.domain.liveness import JobState, LivenessExpectation, job_state

# Ile dni historii czyta strona: ostatni przebieg starszy niż okno = „brak przebiegów”.
STATUS_HISTORY_DAYS = 90
BADGE_JSON_NAME = "badge.json"
_FAST_LOOP = "fast"
_COST_WINDOWS_DAYS = (7, 30)

_LOOP_LABELS = {
    "fast": "Fast Loop (dzienny cykl predykcji)",
    "slow": "Slow Loop (tygodniowy trening)",
    "watch": "Shock Watch (czujka ruchów)",
}
_STATE_LABELS = {
    JobState.RUNNING: "działa",
    JobState.PAUSED: "wstrzymany",
    JobState.OVERDUE: "brak przebiegu w terminie (wstrzymany albo awaria)",
}
_DELIVERY_LABELS = {True: "doręczony", False: "niedoręczony", None: "bez kanału e-mail"}

# Kolor odznaki od wieku ostatniego digestu (dni, włącznie) — starszy = czerwony.
_BADGE_COLOURS = ((1, "brightgreen"), (3, "green"), (7, "yellow"))
_BADGE_STALE = "red"
_BADGE_NEUTRAL = "lightgrey"


@dataclass(frozen=True)
class PublicStatus:
    """Agregaty strony statusu w chwili `generated_at`.

    `fast_state` — stan Fast Loopa z reguły deadmana; `last_runs` — ostatni przebieg
    każdej pętli z historią (kolejność pętli stała); `last_digest_at` — start ostatniego
    przebiegu, którego mail wyszedł; liczniki cykli i darmowych cykli dotyczą Fast Loopa,
    płatne wywołania — wszystkich pętli."""

    generated_at: datetime
    history_days: int
    fast_state: JobState
    last_runs: tuple[CycleRun, ...]
    last_digest_at: datetime | None
    paid_calls_7d: int
    paid_calls_30d: int
    cycles_30d: int
    free_cycles_30d: int


def build_public_status(
    runs: Sequence[CycleRun],
    *,
    now: datetime,
    fast_loop: LivenessExpectation,
    history_days: int = STATUS_HISTORY_DAYS,
) -> PublicStatus:
    """Agregaty strony z przebiegów okna `history_days` (starsze są pomijane)."""
    recent = [run for run in runs if run.started_at >= now - timedelta(days=history_days)]
    latest: dict[str, CycleRun] = {}
    for run in sorted(recent, key=lambda run: run.started_at):
        latest[run.loop] = run
    fast_runs = [run for run in recent if run.loop == _FAST_LOOP]
    last_fast = latest.get(_FAST_LOOP)
    delivered = [run.started_at for run in fast_runs if run.report_delivered is True]
    paid_7d, paid_30d = (
        sum(_paid(run) for run in recent if run.started_at >= now - timedelta(days=days))
        for days in _COST_WINDOWS_DAYS
    )
    cycles_30d = [run for run in fast_runs if run.started_at >= now - timedelta(days=30)]
    return PublicStatus(
        generated_at=now,
        history_days=history_days,
        fast_state=job_state(
            fast_loop, None if last_fast is None else last_fast.started_at, now
        ),
        last_runs=tuple(latest[loop] for loop in _ordered_loops(latest)),
        last_digest_at=max(delivered, default=None),
        paid_calls_7d=paid_7d,
        paid_calls_30d=paid_30d,
        cycles_30d=len(cycles_30d),
        free_cycles_30d=sum(1 for run in cycles_30d if _paid(run) == 0),
    )


def _paid(run: CycleRun) -> int:
    return sum(run.paid_calls.values())


def _ordered_loops(latest: dict[str, CycleRun]) -> list[str]:
    known = [loop for loop in _LOOP_LABELS if loop in latest]
    return known + sorted(loop for loop in latest if loop not in _LOOP_LABELS)


def _days_ago(moment: datetime, now: datetime) -> int:
    return max(0, (now.date() - moment.date()).days)


def _ago_pl(moment: datetime, now: datetime) -> str:
    days = _days_ago(moment, now)
    if days == 0:
        return "dziś"
    return "1 dzień temu" if days == 1 else f"{days} dni temu"


def render_status_page_html(status: PublicStatus) -> str:
    """Samodzielna strona HTML statusu (polskie copy, inline CSS, wszystko escapowane)."""
    sections = [
        _render_fast_loop(status),
        _render_other_loops(status),
        _render_costs(status),
    ]
    generated = status.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    return (
        "<!DOCTYPE html><html lang='pl'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Status StockAgenta</title>"
        f"<style>{_PAGE_CSS}</style></head><body><main class='wrap'>"
        "<h1>Status StockAgenta</h1>"
        "<p class='lede'>Stan pętli z historii przebiegów — także wtedy, gdy wynik jest "
        "niekorzystny. Same liczby: bez symboli, adresów i kwot.</p>"
        f"{''.join(sections)}"
        f"<footer class='foot'>Wygenerowano: {_html.escape(generated)}</footer>"
        "</main></body></html>"
    )


def _paragraph(text: str) -> str:
    return f"<p>{_html.escape(text)}</p>"


def _run_line(run: CycleRun, now: datetime) -> str:
    started = run.started_at.strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"Ostatni przebieg: {started} ({_ago_pl(run.started_at, now)}) — zapisane "
        f"predykcje {run.saved} z {run.eligible}, błędy {run.failures}, mail: "
        f"{_DELIVERY_LABELS[run.report_delivered]}."
    )


def _render_fast_loop(status: PublicStatus) -> str:
    now = status.generated_at
    fast = next((run for run in status.last_runs if run.loop == _FAST_LOOP), None)
    lines = [
        f"<h2>{_html.escape(_LOOP_LABELS[_FAST_LOOP])}: "
        f"<strong>{_html.escape(_STATE_LABELS[status.fast_state])}</strong></h2>"
    ]
    if fast is None:
        lines.append(_paragraph(f"brak przebiegów w ostatnich {status.history_days} dniach."))
    else:
        lines.append(_paragraph(_run_line(fast, now)))
    if status.last_digest_at is None:
        digest = f"Ostatni wysłany digest: brak w ostatnich {status.history_days} dniach."
    else:
        digest = (
            f"Ostatni wysłany digest: {status.last_digest_at:%Y-%m-%d} "
            f"({_ago_pl(status.last_digest_at, now)})."
        )
    lines.append(_paragraph(digest))
    return f"<section class='block'>{''.join(lines)}</section>"


def _render_other_loops(status: PublicStatus) -> str:
    others = [run for run in status.last_runs if run.loop != _FAST_LOOP]
    if not others:
        return ""
    items = "".join(
        f"<li>{_html.escape(_LOOP_LABELS.get(run.loop, run.loop))}: "
        f"{_html.escape(_run_line(run, status.generated_at))}</li>"
        for run in others
    )
    return f"<section class='block'><h2>Pozostałe pętle</h2><ul>{items}</ul></section>"


def _render_costs(status: PublicStatus) -> str:
    calls = (
        f"Płatne wywołania (LLM, dane, embeddingi) — 7 dni: {status.paid_calls_7d}, "
        f"30 dni: {status.paid_calls_30d}."
    )
    if status.cycles_30d == 0:
        free = "Cykle bez płatnych wywołań: brak cykli w ostatnich 30 dniach."
    else:
        share = round(100 * status.free_cycles_30d / status.cycles_30d)
        free = (
            "Cykle bez płatnych wywołań (30 dni, bramka zmienności nie wpuściła ruchu): "
            f"{status.free_cycles_30d} z {status.cycles_30d} ({share}%)."
        )
    return (
        "<section class='block'><h2>Koszt</h2>"
        f"{_paragraph(calls)}{_paragraph(free)}</section>"
    )


def render_status_badge(status: PublicStatus) -> dict[str, Any]:
    """Endpoint JSON shields.io: „last digest: N days ago”.

    Kolor od wieku ostatniego digestu; wstrzymana pętla i brak digestu w oknie są
    szare — to stan, nie awaria."""
    if status.last_digest_at is None:
        message, colour = f"none in {status.history_days} days", _BADGE_NEUTRAL
    else:
        days = _days_ago(status.last_digest_at, status.generated_at)
        message = "today" if days == 0 else "1 day ago" if days == 1 else f"{days} days ago"
        colour = next((c for limit, c in _BADGE_COLOURS if days <= limit), _BADGE_STALE)
    if status.fast_state is JobState.PAUSED:
        colour = _BADGE_NEUTRAL
    return {"schemaVersion": 1, "label": "last digest", "message": message, "color": colour}


_PAGE_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "color:#0f172a;background:#f8fafc;margin:0;padding:24px;line-height:1.5}"
    ".wrap{max-width:720px;margin:0 auto}"
    "h1{font-size:26px;margin:0 0 6px}"
    ".lede{color:#475569;font-size:15px;margin:0 0 18px}"
    ".block{background:#fff;border:1px solid #e2e8f0;border-radius:10px;"
    "padding:16px;margin:16px 0}"
    ".block h2{font-size:17px;margin:0 0 8px}"
    ".block p,.block li{font-size:14px;color:#334155;margin:0 0 8px}"
    ".foot{color:#94a3b8;font-size:11px;text-align:center;margin:20px 0 0}"
)
