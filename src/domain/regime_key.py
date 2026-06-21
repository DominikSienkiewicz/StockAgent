"""Domena: kanoniczny klucz reżimu do tagowania pamięci wektorowej (RAG).

Cross-Asset Vector Memory: każda predykcja zapisuje, w jakim reżimie rynku
powstała (etykieta z `domain/regime.py`). Przy similarity search wolimy analogi
z TEGO SAMEGO reżimu — bessowy analog mało mówi o ruchu w hossie. Ten helper
kanonizuje etykietę reżimu spójnie przy ZAPISIE i przy ODCZYCIE, żeby filtr w
repozytorium porównywał jabłka z jabłkami."""

from __future__ import annotations

# Klucz dla braku/nieznanego reżimu — embeddingi sprzed wprowadzenia tagu mają
# regime NULL i mapują się tutaj; filtr traktuje je jako „pasujące do wszystkiego".
UNKNOWN_REGIME = "UNKNOWN"

# Górny limit długości klucza (kolumna w bazie jest krótkim tekstem).
_MAX_KEY_LEN = 32


def canonical_regime_key(regime_label: str | None) -> str:
    """Normalizuje dowolną etykietę reżimu do zwięzłego, porównywalnego klucza.

    Pusty / None → `UNKNOWN_REGIME`. W przeciwnym razie: trim, upper-case,
    spacje → podkreślenia, przycięcie do `_MAX_KEY_LEN`. Idempotentny:
    `canonical_regime_key(canonical_regime_key(x)) == canonical_regime_key(x)`."""
    if regime_label is None:
        return UNKNOWN_REGIME
    stripped = regime_label.strip()
    if not stripped:
        return UNKNOWN_REGIME
    return stripped.upper().replace(" ", "_")[:_MAX_KEY_LEN]


def regimes_match(a: str | None, b: str | None) -> bool:
    """Czy dwa reżimy są zgodne po kanonizacji. UNKNOWN pasuje do wszystkiego
    (nie chcemy odsiewać analogów bez tagu, póki baza nie zostanie wypełniona)."""
    ka = canonical_regime_key(a)
    kb = canonical_regime_key(b)
    if ka == UNKNOWN_REGIME or kb == UNKNOWN_REGIME:
        return True
    return ka == kb
