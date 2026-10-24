"""Domena Subscriber — per-recipient watchlist (U2).

Czysty value object reprezentujący odbiorcę dziennego digestu wraz ze zbiorem
symboli, które go interesują. Analiza cyklu (i bramka volatility) biegnie RAZ na
unii wszystkich symboli — Subscriber służy WYŁĄCZNIE do re-slicingu gotowego
raportu per odbiorca, więc nie wprowadza żadnych nowych płatnych wywołań.

Reguła: pusty zbiór symboli = "wszystko". Subskrybent bez zadeklarowanej
watchlisty dostaje pełny union symboli cyklu (zachowanie zgodne z dotychczasowym
pojedynczym `DIGEST_TO_EMAIL`). Zbiór niepusty zawęża raport do przecięcia z
symbolami cyklu — `symbols_for` nigdy nie "wymyśla" symboli spoza unii.

BEZ-01: portfel właściciela (`DIGEST_TO_EMAIL`) idzie wyłącznie do niego —
`is_owner` jest fail-closed: brak adresu właściciela = nikt nim nie jest.

Wyłącznie stdlib + inne moduły domeny — zero importów zewnętrznych.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Subscriber:
    """Odbiorca digestu + jego watchlista.

    `symbols` pusty = "wszystko" (subskrybent globalny). Frozen + frozenset →
    VO jest hashowalny i bezpieczny jako element zbioru / klucz słownika.
    """

    # BEZ-11: repo i logi GHA są publiczne — adres nie trafia do `repr` (logi,
    # tracebacki); do logów idzie `masked_email`.
    email: str = field(repr=False)
    symbols: frozenset[str]

    def watches(self, symbol: str) -> bool:
        """Czy `symbol` należy do watchlisty subskrybenta.

        Pusty zbiór = "wszystko" → zawsze True. W przeciwnym razie zwykła
        przynależność do zbioru. Używane do filtrowania `SymbolResult`-ów
        przy re-slicingu raportu per odbiorca.
        """
        if not self.symbols:
            return True
        return symbol in self.symbols


def symbols_for(subscriber: Subscriber, all_symbols: frozenset[str]) -> frozenset[str]:
    """Symbole z bieżącego cyklu, które należy pokazać `subscriber`-owi.

    Pusty zbiór subskrybenta = "wszystko" → zwraca pełny `all_symbols`. W
    przeciwnym razie przecięcie watchlisty z symbolami cyklu — wynik jest ZAWSZE
    podzbiorem `all_symbols` (re-slicing, nie rozszerzanie). Pusty wynik =
    subskrybent nie ma w tym cyklu nic do pokazania (use case może go pominąć).
    """
    if not subscriber.symbols:
        return all_symbols
    return subscriber.symbols & all_symbols


def _normalized(email: str) -> str:
    return email.strip().casefold()


def is_owner(subscriber: Subscriber, owner_email: str | None) -> bool:
    """Czy `subscriber` to właściciel portfela (adres `DIGEST_TO_EMAIL`).

    Porównanie bez wielkości liter i białych znaków na brzegach. Fail-closed:
    pusty albo brakujący adres właściciela nie dopasowuje nikogo, więc portfel
    nie trafia do żadnego subskrybenta.
    """
    if owner_email is None or not _normalized(owner_email):
        return False
    return _normalized(subscriber.email) == _normalized(owner_email)


def masked_email(email: str) -> str:
    """Adres do logów bez pełnej skrzynki: `f***@example.com`. Bez `@` → `***`."""
    local, at, domain = email.strip().partition("@")
    if not at or not local:
        return "***"
    return f"{local[0]}***@{domain}"
