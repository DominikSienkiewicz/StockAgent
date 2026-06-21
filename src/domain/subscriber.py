"""Domena Subscriber — per-recipient watchlist (U2).

Czysty value object reprezentujący odbiorcę dziennego digestu wraz ze zbiorem
symboli, które go interesują. Analiza cyklu (i bramka volatility) biegnie RAZ na
unii wszystkich symboli — Subscriber służy WYŁĄCZNIE do re-slicingu gotowego
raportu per odbiorca, więc nie wprowadza żadnych nowych płatnych wywołań.

Reguła: pusty zbiór symboli = "wszystko". Subskrybent bez zadeklarowanej
watchlisty dostaje pełny union symboli cyklu (zachowanie zgodne z dotychczasowym
pojedynczym `DIGEST_TO_EMAIL`). Zbiór niepusty zawęża raport do przecięcia z
symbolami cyklu — `symbols_for` nigdy nie "wymyśla" symboli spoza unii.

Wyłącznie stdlib + inne moduły domeny — zero importów zewnętrznych.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Subscriber:
    """Odbiorca digestu + jego watchlista.

    `symbols` pusty = "wszystko" (subskrybent globalny). Frozen + frozenset →
    VO jest hashowalny i bezpieczny jako element zbioru / klucz słownika.
    """

    email: str
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
