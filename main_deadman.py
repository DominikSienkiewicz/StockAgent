"""Deadman entry point — mail, gdy job, który powinien chodzić, zamilkł (NIEZ-11).

Brak maila wyglądał dotąd identycznie jak spokojny tydzień, a Slow Loop mógł paść
po cichu. Cron raz dziennie (`deadman_daily.yml`) czyta ostatnie ślady życia jobów
w bazie i porównuje je z oczekiwaniami z config.toml (`liveness_*_max_gap_h`;
0 = job świadomie wstrzymany, np. Fast Loop wg decyzji 2). Spóźnienie → jeden mail
przez Resend na przebieg, więc przy dziennym cronie najwyżej jeden na dobę.

Ślady życia: najnowszy `price_snapshots.timestamp` (Fast Loop) i najnowszy wiersz
`model_scorecards` (Slow Loop zapisuje go per symbol także przy pominięciu).

FinOps: **zero płatnych wywołań** — odczyt Supabase i jeden mail. Gdy śladu nie da
się odczytać, przebieg kończy się exit 1, a krok alertu workflowu mailuje o porażce.

Użycie:
    uv run python main_deadman.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime

from src.config import Settings
from src.domain.liveness import LivenessExpectation, overdue_jobs

logger = logging.getLogger(__name__)


def _render(overdue: tuple[str, ...], last_seen: dict[str, datetime | None]) -> tuple[str, str]:
    """Temat i treść maila deadmana. Po polsku."""
    subject = f"⚠️ StockAgent: brak znaku życia — {', '.join(overdue)}"
    lines = []
    for job in overdue:
        stamp = last_seen.get(job)
        lines.append(f"{job}: ostatni ślad {stamp.isoformat() if stamp else 'brak'}")
    body = "Deadman nie widzi świeżego śladu życia jobów:\n" + "\n".join(lines)
    return subject, body


def main(settings: Settings | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = settings or Settings()  # type: ignore[call-arg]

    # Import lokalny — fabryki z DI Containera Fast Loopa; tylko darmowe adaptery.
    from main_agent import build_notifier, build_repository

    repository = build_repository(settings)
    fast = LivenessExpectation("Fast Loop", settings.liveness_fast_loop_max_gap_h)
    slow = LivenessExpectation("Slow Loop", settings.liveness_slow_loop_max_gap_h)
    probes: list[tuple[LivenessExpectation, Callable[[], datetime | None]]] = [
        (fast, repository.get_latest_snapshot_at),
        (slow, repository.get_last_scorecard_at),
    ]
    expectations = [expectation for expectation, _ in probes]
    last_seen: dict[str, datetime | None] = {}
    try:
        for expectation, read in probes:
            if expectation.max_gap_hours > 0:
                last_seen[expectation.job] = read()
    except Exception:
        logger.exception("Deadman: nie da się odczytać śladów życia — exit 1")
        return 1

    overdue = overdue_jobs(expectations, last_seen, datetime.now(UTC))
    if not overdue:
        logger.info("Deadman: wszystkie oczekiwane joby dały znak życia")
        return 0

    subject, body = _render(overdue, last_seen)
    try:
        build_notifier(settings).send_report(subject=subject, html_body=body, plain_text=body)
    except Exception:
        logger.exception("Deadman: wysyłka alertu padła — exit 1")
        return 1
    logger.warning("Deadman: wysłano alert — %s", ", ".join(overdue))
    return 0


if __name__ == "__main__":
    sys.exit(main())
