"""Tłumaczenie wyjątków dostawców na taksonomię portów (ARCH-33).

`as_port_error(exc, kind)` zwraca kopię `exc`, która jest JEDNOCZEŚNIE klasą
taksonomii (`kind`) i dotychczasowym typem dostawcy. Przejście na taksonomię jest
przyrostowe: wywołujący, którzy łapią `requests.RequestException`,
`anthropic.APIStatusError` albo PostgREST `APIError`, oraz testy na tych typach
działają bez zmian, a nowy kod łapie `ProviderUnavailable`, `QuotaExhausted` itd.

Kopia zachowuje komunikat (już zredagowany przez `safe_get` /
`raise_for_status_redacted`), atrybuty (`response`, `status_code`, `code`…)
i traceback. Nazwa klasy to `<Taksonomia>[<TypDostawcy>]`, np.
`ProviderUnavailable[HTTPError]` — log i step summary (NIEZ-22) pokazują obie.
"""

from __future__ import annotations

import threading
from typing import cast

from src.application.errors import PortError, ProviderUnavailable, QuotaExhausted

# Jedna klasa na parę (taksonomia, typ dostawcy) — `except` i `isinstance` widzą
# ten sam typ przy każdym tłumaczeniu, także z wątków równoległych symboli.
_bridges: dict[tuple[type[PortError], type[Exception]], type[Exception]] = {}
_bridges_lock = threading.Lock()


def _bridge_class(kind: type[PortError], original: type[Exception]) -> type[Exception]:
    with _bridges_lock:
        bridge = _bridges.get((kind, original))
        if bridge is None:
            name = f"{kind.__name__}[{original.__name__}]"
            bridge = type(name, (kind, original), {"__module__": __name__, "__qualname__": name})
            _bridges[(kind, original)] = bridge
        return bridge


def as_port_error[E: Exception](exc: E, kind: type[PortError]) -> E:
    """`exc` przetłumaczony na `kind` bez utraty typu dostawcy. Wyjątek, który już
    jest `kind`, wraca bez zmian."""
    if isinstance(exc, kind):
        return exc
    bridge = _bridge_class(kind, type(exc))
    # Bez `__init__`: konstruktory dostawców mają różne sygnatury, a stan wyjątku
    # (args + atrybuty instancji) da się skopiować wprost.
    translated = bridge.__new__(bridge, *exc.args)
    translated.args = exc.args
    translated.__dict__.update(exc.__dict__)
    return cast(E, translated.with_traceback(exc.__traceback__))


_RATE_LIMITED = 429
_SERVER_ERROR = 500


def sdk_error_kind(exc: Exception) -> type[PortError]:
    """Klasa taksonomii błędu SDK dostawcy LLM (Anthropic, OpenAI): brak odpowiedzi
    (połączenie, timeout — bez `status_code`) i 5xx (w tym 529 Overloaded) to
    niedostępny dostawca, 429 to kwota, reszta 4xx (zły prompt, klucz, model) to
    ogólny błąd portu."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return ProviderUnavailable
    if status == _RATE_LIMITED:
        return QuotaExhausted
    return ProviderUnavailable if status >= _SERVER_ERROR else PortError
