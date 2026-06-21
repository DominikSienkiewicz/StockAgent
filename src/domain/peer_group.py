"""Domena grup peerów — rozumowanie o zarażaniu (contagion) między symbolami.

Zamiast traktować każdy ticker w izolacji, agent przed predykcją wstrzykuje
werdykty SKORELOWANYCH spółek z TEGO SAMEGO cyklu. Grupy peerów są deklarowane
w konfiguracji (np. {"semis": ["NVDA","AMD","TSM"], "mega_tech": [...]}).

"Same-cycle" = tylko peery JUŻ przetworzone wcześniej w tym cyklu (best-effort).
To czyste re-użycie werdyktów już policzonych — ZERO dodatkowych płatnych wywołań.

Czysta logika, ZERO importów zewnętrznych — tylko stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def peers_of(symbol: str, groups: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Wszystkie INNE symbole dzielące jakąkolwiek grupę z `symbol`.

    Symbol może należeć do wielu grup → wynik to UNIA peerów ze wszystkich
    grup, do których należy. Dedup zachowuje pierwsze wystąpienie, kolejność
    jest stabilna (grupy w kolejności wstawienia mappingu, członkowie w
    kolejności deklaracji). Sam `symbol` jest zawsze wykluczony. Symbol bez
    żadnej grupy → `()`.
    """
    peers: list[str] = []
    seen: set[str] = set()
    for members in groups.values():
        if symbol not in members:
            continue
        for member in members:
            if member == symbol or member in seen:
                continue
            seen.add(member)
            peers.append(member)
    return tuple(peers)


def build_peer_context(
    symbol: str,
    groups: Mapping[str, Sequence[str]],
    processed: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Pary (peer_symbol, stance) dla peerów `symbol` JUŻ policzonych w cyklu.

    `processed` = {przetworzony-w-tym-cyklu-symbol: krótki opis stanowiska
    (np. trend albo rekomendacja rady)}. Zwracamy stanowiska tylko tych peerów
    `symbol`, które są już w `processed` — peery jeszcze nieliczone i sam
    `symbol` są pomijane. Kolejność stabilna, zgodna z `peers_of`. Pusto, gdy
    żaden peer nie był jeszcze przetworzony.
    """
    return tuple(
        (peer, processed[peer])
        for peer in peers_of(symbol, groups)
        if peer in processed
    )
