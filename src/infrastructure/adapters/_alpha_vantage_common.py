"""Wspólne reguły odpowiedzi Alpha Vantage dla klienta newsów i fundamentów (DAT-06).

AV sygnalizuje limit odpowiedzią 200 z polem-markerem zamiast danych:

- `Information` — limit dzienny (25 zapytań na klucz) albo komunikat planu,
  więc wołający od razu rotuje klucz;
- `Note` — limit częstotliwości (minutowy): wołający odczekuje
  `NOTE_BACKOFF_SECONDS` i ponawia TYM SAMYM kluczem, a rotuje dopiero przy
  drugim `Note`. Natychmiastowa rotacja spaliłaby dzienny budżet rezerwowego
  klucza na limicie, który mija po chwili.

Treści markera nie klasyfikujemy po frazach: bez przechwyconych prawdziwych
payloadów każda reguła tekstowa byłaby zgadywaniem.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.infrastructure.adapters._http import redact_secrets

INFORMATION_MARKER = "Information"
NOTE_MARKER = "Note"
NOTE_BACKOFF_SECONDS = 2.0
_MARKER_LOG_CHARS = 160


def is_soft_limit(payload: Mapping[str, Any], data_keys: Iterable[str]) -> bool:
    """Marker `Information`/`Note` bez żadnego z `data_keys`.

    Obecny klucz danych znaczy prawdziwą odpowiedź — AV nie miesza danych
    z notką o limicie, więc notka obok danych nie jest soft-limitem."""
    if INFORMATION_MARKER not in payload and NOTE_MARKER not in payload:
        return False
    return not any(key in payload for key in data_keys)


def is_frequency_limit(payload: Mapping[str, Any]) -> bool:
    """`Note` bez `Information` — limit minutowy, po którym warto chwilę odczekać."""
    return NOTE_MARKER in payload and INFORMATION_MARKER not in payload


def soft_limit_text(payload: Mapping[str, Any], api_keys: Iterable[str]) -> str:
    """Treść markera do logu, bez kluczy: AV potrafi wkleić klucz do komunikatu.

    Redakcja idzie przed przycięciem — cięcie w środku klucza zostawiłoby jego
    prefiks, którego dosłowna redakcja już nie rozpozna."""
    raw = payload.get(INFORMATION_MARKER) or payload.get(NOTE_MARKER) or ""
    return redact_secrets(str(raw), api_keys)[:_MARKER_LOG_CHARS]
