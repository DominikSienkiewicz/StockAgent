"""WIA-43 — raport CLI wkładu krańcowego person rady (leave-one-out).

Liczy domena (`council_ablation.leave_one_out`) z surowych głosów RPC
`persona_vote_outcomes` (migracja 045); tu tylko tekst. Składanie `Settings`
i repozytorium robi `main_tools.py council-ablation` (ARCH-39). Read-only, zero
płatnych wywołań:
    uv run python main_tools.py council-ablation --days 90
"""

from __future__ import annotations

from src.domain.council_ablation import CouncilAblation, PersonaMarginal
from src.domain.significance import MIN_BOOTSTRAP_CYCLES

_CANDIDATE_TAG = "kandydat do wycięcia"


def _pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def _interval(interval: tuple[float, float] | None, render: str) -> str:
    if interval is None:
        return "bez CI"
    low, high = interval
    if render == "pp":
        return f"[{_pp(low)}; {_pp(high)}]"
    return f"[{low:+.3f}; {high:+.3f}]"


def _line(marginal: PersonaMarginal) -> str:
    tag = f"  ← {_CANDIDATE_TAG}" if marginal.removal_candidate else ""
    return (
        f"  {marginal.investor_name:<14} n={marginal.predictions:<4} "
        f"odwraca {marginal.flip_rate * 100:4.1f}%  "
        f"Δ trafności {_pp(marginal.accuracy_delta)} "
        f"{_interval(marginal.accuracy_interval, 'pp')}  "
        f"Δ Brier {marginal.brier_delta:+.3f} {_interval(marginal.brier_interval, 'brier')}"
        f"{tag}"
    )


def format_council_ablation(report: CouncilAblation, *, days: int) -> str:
    """Tabela wkładu krańcowego; Δ = wynik rady BEZ persony − wynik pełnej rady
    (Δ trafności < 0: persona pomaga; Δ Brier > 0: persona poprawia kalibrację)."""
    header = f"Wkład krańcowy person rady (leave-one-out, ostatnie {days} dni)"
    if report.predictions == 0 or report.accuracy is None or report.brier is None:
        return f"{header}\n  Brak rozliczonych predykcji z radą ≥ 2 głosy do oceny."
    lines = [
        header,
        f"  Predykcje z radą (≥ 2 głosy):   n={report.predictions} · "
        f"trafność rady {report.accuracy * 100:.1f}% · Brier siły konsensusu {report.brier:.3f}",
        "  Δ = rada BEZ persony − pełna rada; CI blokowe 95% po dniach UTC "
        f"(k < {MIN_BOOTSTRAP_CYCLES} cykli → bez CI)",
        *(_line(marginal) for marginal in report.personas),
        "  Uwaga: leave-one-out pomija interakcje — dwie redundantne persony wychodzą "
        "obie na zero, bo każda zastępuje drugą (zob. redundancja rady w recapie "
        "tygodniowym, WIA-45). Wytnij jedną i zmierz od nowa.",
    ]
    return "\n".join(lines)
