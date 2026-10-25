"""NIEZ-30: bramka schematu na starcie przebiegu.

Flaga zależna od migracji, włączona przed jej zaaplikowaniem, kończy się PGRST204 i
gubi cały zapis predykcji. Entry-point pyta bazę o najwyższą zaaplikowaną migrację
(`schema_version()`, migracja 071) i dostaje tu listę flag do wyłączenia na czas
przebiegu razem z alertem do bannera maila.

Decyzja właściciela (2026-10-24), fail-open: gdy wersji nie da się ustalić (brak
funkcji, baza nie odpowiada, rejestr migracji pusty), flagi ZOSTAJĄ. Wynik niesie
wtedy `schema_version = None`, a ostrzeżenie idzie do logu; alertu w mailu nie ma,
bo to brak informacji, nie dowód rozjazdu. Zamiana wyniku na `Settings` i podsumowanie
przebiegu należą do entry-pointu — warstwa application nie zna konfiguracji.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.application.errors import PortError
from src.domain.feature_requirements import FLAG_REQUIRED_MIGRATION, flags_missing_migration
from src.domain.quota import QuotaAlert, QuotaSeverity

logger = logging.getLogger(__name__)

SCHEMA_ALERT_SOURCE = "Schemat bazy"


class SchemaVersionSource(Protocol):
    """Port, który umie podać numer najnowszej zaaplikowanej migracji."""

    def schema_version(self) -> int | None: ...


@dataclass(frozen=True)
class SchemaGateOutcome:
    """`schema_version` None = nie wiadomo (flagi zostają); `alert` tylko przy wyłączeniu."""

    schema_version: int | None
    disabled: frozenset[str]
    alert: QuotaAlert | None


def evaluate_schema_gate(
    source: SchemaVersionSource, enabled: Mapping[str, bool], *, now: datetime
) -> SchemaGateOutcome:
    """Które z włączonych flag `enabled` czekają na migrację, której baza jeszcze nie ma.

    Błąd portu (brak funkcji `schema_version`, awaria bazy) i odpowiedź, która nie jest
    liczbą całkowitą, oznaczają „nie wiem”: flagi zostają, w logu jeden WARNING."""
    try:
        version = source.schema_version()
    except PortError as exc:
        logger.warning(
            "Schema probe unavailable (%s) — migration-bound flags left on", type(exc).__name__
        )
        return _unknown()
    if isinstance(version, bool) or not isinstance(version, int):
        logger.warning(
            "Schema version unknown (registry missing or empty) — migration-bound flags left on"
        )
        return _unknown()

    disabled = flags_missing_migration(enabled, schema_version=version)
    if not disabled:
        return SchemaGateOutcome(schema_version=version, disabled=frozenset(), alert=None)
    needed = ", ".join(
        f"{flag} (needs {FLAG_REQUIRED_MIGRATION[flag]:03d})" for flag in sorted(disabled)
    )
    logger.warning("Database is at migration %d — switched off for this run: %s", version, needed)
    return SchemaGateOutcome(
        schema_version=version, disabled=disabled, alert=_alert(version, disabled, now)
    )


def _unknown() -> SchemaGateOutcome:
    return SchemaGateOutcome(schema_version=None, disabled=frozenset(), alert=None)


def _alert(version: int, disabled: frozenset[str], now: datetime) -> QuotaAlert:
    names = ", ".join(sorted(disabled))
    return QuotaAlert(
        source=SCHEMA_ALERT_SOURCE,
        severity=QuotaSeverity.WARNING,
        message=(
            f"Baza ma migracje do {version:03d}, więc na ten przebieg wyłączono flagi "
            f"czekające na nowsze: {names}."
        ),
        action=(
            "Uruchom workflow „DB Migrate (manual)” (najpierw dry_run); kolejny przebieg "
            "włączy flagi sam."
        ),
        occurred_at=now,
    )
