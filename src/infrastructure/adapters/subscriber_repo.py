"""Adapter Supabase dla tabeli `subscribers` (U2 — per-recipient watchlist).

Czysty odczyt konfiguracji odbiorców digestu: kto dostaje raport i do jakiego
podzbioru symboli ma być on zawężony. NIE wprowadza żadnych nowych płatnych
wywołań — analiza cyklu i bramka volatility pozostają nietknięte; ta tabela
napędza WYŁĄCZNIE re-slicing gotowego raportu per odbiorca.

`symbols` w bazie może mieć kilka kształtów (różne źródła wpisów: SQL seed,
import CSV, panel admina) — parsujemy je odpornie:
  - lista (text[] / JSONB array)  → frozenset elementów
  - CSV string "AAPL, VOO"        → frozenset po podziale i trim
  - None / "" / []                → pusty frozenset = "wszystko" (globalny)

Czytanie jest graceful: dowolny błąd zapytania (brak tabeli przy
niezaaplikowanej migracji, błąd sieci) → []. Wtedy main_agent robi fallback do
pojedynczego `DIGEST_TO_EMAIL`, zamiast wywalać cały cykl.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from supabase import create_client

from src.application.ports import SubscriberRepositoryPort
from src.domain.subscriber import Subscriber

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIBERS_TABLE = "subscribers"


def _parse_symbols(raw: Any) -> frozenset[str]:
    """Mapuje surową wartość kolumny `symbols` na frozenset symboli.

    Akceptuje listę (text[]/JSONB), CSV string albo wartość pustą (None/""/[]).
    Każdy element jest trimowany; puste tokeny są pomijane. Pusty wynik =
    "wszystko" (subskrybent globalny) — interpretację robi domena (`watches` /
    `symbols_for`), adapter tylko normalizuje kształt.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        tokens = raw.split(",")
    elif isinstance(raw, list | tuple):
        tokens = [str(t) for t in raw]
    else:
        # Nieoczekiwany typ (np. dict) — traktujemy jak brak watchlisty.
        return frozenset()
    return frozenset(tok.strip() for tok in tokens if tok and str(tok).strip())


class SupabaseSubscriberRepository(SubscriberRepositoryPort):
    """Adapter odczytu tabeli `subscribers` (PostgreSQL via PostgREST).

    Implementuje kontrakt `SubscriberRepositoryPort` (`list_subscribers`).
    """

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = DEFAULT_SUBSCRIBERS_TABLE,
    ) -> None:
        self._client = create_client(url, key)
        self._table = table_name

    def list_subscribers(self) -> list[Subscriber]:
        """Zwraca wszystkich odbiorców digestu z tabeli `subscribers`.

        Graceful: błąd zapytania → []. Wiersze bez `email` są pomijane (nie ma
        gdzie wysłać). Kolumna `symbols` parsowana odpornie (lista / CSV / puste).
        """
        try:
            response = self._client.table(self._table).select("*").execute()
        except Exception:
            logger.warning(
                "list_subscribers: query on table %r failed — "
                "falling back to no subscribers (apply the subscribers "
                "migration to enable per-recipient watchlists).",
                self._table,
                exc_info=True,
            )
            return []

        rows = cast(list[dict[str, Any]], response.data or [])
        out: list[Subscriber] = []
        for row in rows:
            email = row.get("email")
            if not isinstance(email, str) or not email.strip():
                continue
            out.append(
                Subscriber(
                    email=email.strip(),
                    symbols=_parse_symbols(row.get("symbols")),
                )
            )
        return out
