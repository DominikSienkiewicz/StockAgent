"""MAIL-35: polityka cichego dnia — pełny, zwarty albo żaden mail.

Pełny mail (40–70 KB) w dzień bez sygnału uczy, że „ten mail zwykle nic nie mówi”.
Polityka decyduje, co wysłać, gdy cykl nie ma nic ponad stan:

- ALWAYS — zawsze pełny mail (zachowanie sprzed MAIL-35);
- COMPACT_WHEN_QUIET — w cichy dzień zwarty mail (największy ruch, próg, stan
  portfela i pełna stopka źródeł); decyzja właściciela 2026-10-23;
- SKIP_WHEN_QUIET — w cichy dzień bez maila. Ma sens dopiero z tygodniowym
  heartbeatem (MAIL-28): bez niego brak maila wygląda jak awaria.

Cichy dzień jest wąski: pusty lead, zero rozliczeń, zero szoków z watcha od
ostatniego maila, żaden symbol za bramką volatility (ani przeanalizowany, ani
odcięty budżetem), zero błędów i nie pierwszy cykl (onboarding). Nieznana liczba
rozliczeń (odczyt padł) nie jest ciszą — pełny mail ostrzega o awarii odczytu.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from src.domain.digest_lead import LeadItem


class DigestPolicy(StrEnum):
    """Co zrobić z mailem w cichy dzień (wartości = napisy z config.toml)."""

    ALWAYS = "always"
    COMPACT_WHEN_QUIET = "compact_when_quiet"
    SKIP_WHEN_QUIET = "skip_when_quiet"


class DigestForm(StrEnum):
    """Postać maila tego cyklu."""

    FULL = "full"
    COMPACT = "compact"
    SKIP = "skip"


def is_quiet_day(
    lead_items: Sequence[LeadItem],
    resolved: int | None,
    shocks: int,
    *,
    gated: int = 0,
    errors: int = 0,
    onboarding: bool = False,
) -> bool:
    """Czy cykl nie ma nic, co pełny mail powiedziałby ponad stan (patrz moduł)."""
    return (
        not lead_items
        and resolved == 0
        and shocks == 0
        and gated == 0
        and errors == 0
        and not onboarding
    )


def should_send(
    lead_items: Sequence[LeadItem],
    resolved: int | None,
    shocks: int,
    policy: DigestPolicy,
    *,
    gated: int = 0,
    errors: int = 0,
    onboarding: bool = False,
) -> DigestForm:
    """Postać maila z polityki: dzień, który nie jest cichy, zawsze dostaje pełny mail.

    `resolved` — rozliczenia w oknie maila (None = odczyt padł), `shocks` — alerty
    szoku z watcha od ostatniego maila, `gated` — symbole za bramką volatility
    (zapisane i odcięte budżetem), `errors` — symbole z błędem, `onboarding` —
    pierwszy cykl z sekcją powitalną."""
    if policy is DigestPolicy.ALWAYS or not is_quiet_day(
        lead_items, resolved, shocks, gated=gated, errors=errors, onboarding=onboarding
    ):
        return DigestForm.FULL
    if policy is DigestPolicy.COMPACT_WHEN_QUIET:
        return DigestForm.COMPACT
    return DigestForm.SKIP
