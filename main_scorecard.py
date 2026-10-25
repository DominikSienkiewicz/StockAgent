"""Publiczny scorecard entry point — strona z agregatami na GitHub Pages (WIA-52).

Scorecard to jedyny artefakt, który może zbudować zaufanie obcych, a Fast Loop stoi
(decyzja 1). Ten entry-point renderuje go z danych, które już są w bazie: zamknięte
predykcje, próbki kalibracji i darmowe snapshoty IVV pod benchmark. Strona jest ta
sama co w cyklu Fast Loopa (`main_agent.public_scorecard_use_case`, po kosztach
paper-PnL z config.toml), a obok niej leży jej maszynowy bliźniak `scorecard.json`
(WIA-81, schemat `docs/schemas/scorecard.schema.json`). Składa wyłącznie
repozytorium (bez `DB_WRITES_ENABLED`, więc tylko odczyt) i publisher scorecardu
z DI Containera Fast Loopa.

Workflow `scorecard_pages.yml` wysyła potem na Pages WYŁĄCZNIE katalog
`dirname(public_scorecard_path)`, nigdy `public/` z prywatnym digestem. Atestacji
(JSONL) ta droga nie publikuje: deploy Pages podmienia całą witrynę.

`--days N` nadpisuje `track_record_days` (okno odczytu i okno na stronie): przy
wstrzymanym Fast Loopie domyślne 30 dni daje puste okno, a puste okno nie daje
artefaktu — Pages zostaje przy ostatnio opublikowanej stronie.

NIEZ-33: przy `public_status_page_enabled` ten sam przebieg zapisuje stronę statusu
(`public_status_path`, podkatalog scorecardu) i `badge.json` z historii `cycle_runs` —
wychodzi tym samym artefaktem Pages, więc tylko razem ze stroną scorecardu.

Kody wyjścia: 0 — strona zapisana albo świadomie bez artefaktu (flaga off, puste
okno); 1 — błąd odczytu bazy albo zapisu strony, `scorecard.json` lub strony statusu
(krok alertu workflowu mailuje).

FinOps: **zero płatnych wywołań** — odczyt Supabase i zapis plików strony.

Użycie:
    uv run python main_scorecard.py [--days N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from src.application.errors import PortError
from src.application.ports import RepositoryPort
from src.config import Settings
from src.infrastructure.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _positive_days(value: str) -> int:
    days = int(value)
    if days < 1:
        raise argparse.ArgumentTypeError(f"--days must be >= 1 (got {days})")
    return days


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_scorecard.py",
        description="Render the public scorecard (aggregates only) for GitHub Pages.",
    )
    parser.add_argument(
        "--days",
        type=_positive_days,
        default=None,
        help="Okno zamkniętych predykcji w dniach (default: track_record_days z config.toml).",
    )
    return parser


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or Settings.from_env()
    configure_logging("main_scorecard", settings.secret_values())
    if not settings.public_scorecard_enabled:
        logger.info(
            "Public scorecard wyłączony (public_scorecard_enabled = false) — brak artefaktu"
        )
        return 0
    if args.days is not None:
        settings = settings.model_copy(update={"track_record_days": args.days})
    return _publish(settings)


def _publish(settings: Settings) -> int:
    # Import lokalny — fabryki z DI Containera Fast Loopa; tylko darmowe adaptery.
    from main_agent import (
        _paper_costs,
        build_public_scorecard_publisher,
        build_repository,
        public_scorecard_use_case,
    )

    repository = build_repository(settings)
    use_case = public_scorecard_use_case(
        settings, repository, build_public_scorecard_publisher(settings)
    )
    try:
        published = use_case.publish(_paper_costs(settings))
    except Exception:
        logger.exception("Public scorecard: odczyt bazy padł — exit 1")
        return 1
    if published is None:
        logger.warning(
            "Public scorecard: brak zamkniętych predykcji w oknie %d dni — brak artefaktu",
            settings.track_record_days,
        )
        return 0
    if not published:
        logger.error(
            "Public scorecard: zapis strony albo scorecard.json obok %s padł — exit 1",
            settings.public_scorecard_path,
        )
        return 1
    logger.info("Public scorecard: okno %d dni zapisane", settings.track_record_days)
    return _publish_status(settings, repository)


def _publish_status(settings: Settings, repository: RepositoryPort) -> int:
    """NIEZ-33: strona statusu w katalogu scorecardu — ten sam artefakt Pages."""
    from main_agent import _utc_now, public_status_enabled, status_page_use_case

    if not public_status_enabled(settings):
        return 0
    try:
        written = status_page_use_case(settings, repository, _utc_now).publish()
    except PortError:
        logger.exception("Status page: odczyt cycle_runs padł — exit 1")
        return 1
    if not written:
        logger.error(
            "Status page: zapis %s albo badge.json padł — exit 1", settings.public_status_path
        )
        return 1
    logger.info("Status page: zapisana obok scorecardu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
