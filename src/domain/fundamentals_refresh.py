"""Rotacja spółek w tygodniowym refreshu fundamentów Alpha Vantage (DAT-19).

Darmowy klucz AV daje 25 requestów na dobę, a fundamenty spółki to dwa requesty
(OVERVIEW + EARNINGS). Refresh szedł po liście `SYMBOLS` od początku, więc przy
jednym kluczu zawsze kończył się na tych samych pierwszych 12 spółkach, a reszta
nigdy nie dostawała świeżych danych. Partia rotuje: najpierw spółki bez wpisu
w cache, potem najdawniej odświeżone, remis alfabetycznie. Czysta logika.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

AV_DAILY_REQUESTS_PER_KEY = 25
AV_REQUESTS_PER_COMPANY = 2


def refresh_budget(n_keys: int) -> int:
    """Ile spółek zmieści się w dziennym limicie `n_keys` kluczy AV:
    `floor(25 × n_keys / 2)`. Brak kluczy → 0."""
    return max(0, AV_DAILY_REQUESTS_PER_KEY * n_keys // AV_REQUESTS_PER_COMPANY)


def select_refresh_batch(fetched_at: Mapping[str, datetime | None], budget: int) -> list[str]:
    """Spółki do odświeżenia w tym przebiegu, najwyżej `budget`.

    `fetched_at` mapuje symbol na chwilę ostatniego zapisu w cache (None = brak
    wpisu). Kolejność: bez wpisu (alfabetycznie), potem rosnąco po `fetched_at`
    (remis alfabetycznie)."""
    if budget <= 0:
        return []
    missing = sorted(symbol for symbol, stamp in fetched_at.items() if stamp is None)
    known = sorted(
        (stamp, symbol) for symbol, stamp in fetched_at.items() if stamp is not None
    )
    return [*missing, *(symbol for _, symbol in known)][:budget]
