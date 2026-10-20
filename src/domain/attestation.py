"""Commit-reveal — kryptograficznie weryfikowalny track record (czysta domena).

W momencie predykcji publikujemy SHA-256 *commitment* z jej treści plus losowej
soli. Przy zamknięciu predykcji ujawniamy (reveal) plaintext + sól, więc każdy
sceptyk może samodzielnie policzyć hash i potwierdzić, że commitment odpowiada
dokładnie tej predykcji — a nie wersji dopasowanej po fakcie do ruchu ceny.

UCZCIWOŚĆ CLAIMU (świadomie ograniczona):
    Niezależny timestamp dałaby historia commitów GitHuba — ale TYLKO wtedy, gdy
    commitmenty faktycznie są do repo publikowane. Dziś żaden workflow ich nie
    publikuje (flaga `attestation_enabled` jest wyłączona, publikacja to osobny
    etap). Nawet po publikacji daty commitów Gita SĄ FAŁSZOWALNE
    (`GIT_COMMITTER_DATE` można ustawić dowolnie). To NIE jest niepodrabialny
    znacznik czasu. Realny gwarant to *tamper-evidence przy branch
    protection*: gdy gałąź jest chroniona i append-only, cofnięcie lub
    podmiana wcześniej opublikowanego commitmentu zostawia widoczny ślad. Nie
    obiecujemy w tym module niczego więcej — mocniejszy dowód czasu (np.
    OpenTimestamps) to opcjonalny, osobny etap.

NAJWIĘKSZE RYZYKO — kanonizacja zamrożona od dnia 1:
    Commitmenty publikowane dziś weryfikuje się miesiącami później. Każda zmiana
    sposobu kanonizacji (kolejność kluczy, separatory, serializacja `Decimal`,
    obecność `schema_version`) rozjechałaby weryfikację CAŁEJ historii bez
    możliwości naprawy wstecz. Dlatego: (1) `schema_version` jest wstrzykiwane do
    każdego commitmentu od pierwszego zapisu, (2) golden testy zamrażają konkretne
    pary payload -> hash. Kontrakt kanonizacji jest częścią danych, nie
    szczegółem implementacyjnym — nie zmieniaj go bez podniesienia SCHEMA_VERSION.

Zero I/O, zero importów zewnętrznych — tylko stdlib (`hashlib`, `json`,
`secrets`) i domena (`Money` dla skali cen). Publikacja i odczyt kwitów należą
do portu/adaptera w warstwach wyższych; tutaj żyje wyłącznie deterministyczna
krypto-logika.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from src.domain.value_objects import Money

# Wersja kontraktu kanonizacji. Wchodzi do KAŻDEGO commitmentu, żeby przyszły
# weryfikator wiedział, którą regułą serializacji odtworzyć hash. Podnieś TYLKO
# przy świadomej, nieodwracalnej zmianie kanonizacji — i zostaw stare golden testy.
#
# v1 — payload predykcji z cenami w pełnej precyzji Pythona (`str(Decimal)`).
#      Zamrożona: weryfikuje commitmenty zapisane przed v2, dlatego zostaje
#      domyślną wersją `canonicalize` / `commit` / `verify` (golden testy v1).
# v2 — ceny w skali kolumny DECIMAL(14,4) (`prediction_commitment_payload`)
#      plus pola, które scorecard eksponuje: pewność, horyzont, werdykt rady,
#      wersja przepisu i hash protokołu ewaluacji.
SCHEMA_VERSION_V1 = 1
SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION_V1, SCHEMA_VERSION})

# Zarezerwowany klucz — moduł jest właścicielem wersji schematu i nie pozwala
# payloadowi jej nadpisać (inaczej dwa różne payloady mogłyby udawać tę samą wersję).
_SCHEMA_KEY = "schema_version"


def _json_default(value: Any) -> str:
    """Fallback serializacji dla typów, których `json` nie zna natywnie.

    `Decimal` idzie przez `str()` — NIE przez float. `float(Decimal("1.10"))`
    traci końcowe zero i wprowadza błąd binarny (0.1 nie ma skończonej
    reprezentacji), co złamałoby round-trip reveal. `str(Decimal(...))` zachowuje
    dokładny literał, więc "1.10" i "1.1" pozostają rozróżnialne — to celowe:
    liczba miejsc po przecinku bywa znacząca w kontekście finansowym.
    """
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Typ nieserializowalny w attestation payload: {type(value).__name__}")


def canonicalize(
    payload: Mapping[str, Any], *, schema_version: int = SCHEMA_VERSION_V1
) -> str:
    """Deterministyczna, kanoniczna serializacja payloadu do JSON.

    Reguła (ZAMROŻONA — patrz docstring modułu):
        `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
    plus wstrzyknięty `schema_version`. `sort_keys` czyni wynik niezależnym od
    kolejności kluczy wejścia; brak białych znaków (`separators`) usuwa
    niejednoznaczność formatowania; `ensure_ascii=False` zachowuje polskie znaki
    dosłownie zamiast escape'ować je do `\\uXXXX`.

    `schema_version` domyślnie v1 — zamrożony kontrakt opublikowanych
    commitmentów. Nowe commitmenty predykcji podają `SCHEMA_VERSION` jawnie;
    wersja spoza znanych → `ValueError`.

    Payload NIE może zawierać własnego klucza `schema_version` — to klucz
    zarezerwowany przez moduł (`ValueError`).
    """
    if _SCHEMA_KEY in payload:
        raise ValueError(
            f"Klucz '{_SCHEMA_KEY}' jest zarezerwowany i wstrzykiwany przez moduł "
            "— nie przekazuj go w payloadzie."
        )
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Nieznany {_SCHEMA_KEY}: {schema_version}")
    versioned: dict[str, Any] = {_SCHEMA_KEY: schema_version, **payload}
    return json.dumps(
        versioned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def commit(
    payload: Mapping[str, Any], salt: str, *, schema_version: int = SCHEMA_VERSION_V1
) -> str:
    """Zwraca SHA-256 (hex) commitment payloadu związanego z solą.

    Hashujemy `canonicalize(payload) + salt`. Sól czyni commitment odpornym na
    atak słownikowy: bez znajomości soli nie da się zgadnąć treści małego,
    przewidywalnego payloadu przez policzenie hashy wszystkich kandydatów.
    """
    return commit_canonical(canonicalize(payload, schema_version=schema_version), salt)


def commit_canonical(canonical: str, salt: str) -> str:
    """SHA-256 (hex) z gotowego kanonicznego plaintextu (`canonicalize`) i soli.

    Reveal v2 publikuje plaintext dosłownie — weryfikator liczy hash z bajtów,
    bez ponownej kanonizacji i bez typów kolumn bazy."""
    return hashlib.sha256((canonical + salt).encode("utf-8")).hexdigest()


def verify(
    payload: Mapping[str, Any],
    salt: str,
    commitment: str,
    *,
    schema_version: int = SCHEMA_VERSION_V1,
) -> bool:
    """Sprawdza, czy `payload` + `salt` odtwarzają dany `commitment`.

    Porównanie hexa jest niewrażliwe na wielkość liter (commitment odczytany
    z JSONL/GitHuba może być zapisany wielkimi literami). Używamy
    `hmac.compare_digest`, by uniknąć wycieku informacji przez czas porównania.
    """
    return verify_canonical(
        canonicalize(payload, schema_version=schema_version), salt, commitment
    )


def verify_canonical(canonical: str, salt: str, commitment: str) -> bool:
    """`verify` dla dosłownego plaintextu z revealu v2 (bez ponownej kanonizacji)."""
    return hmac.compare_digest(commit_canonical(canonical, salt), commitment.lower())


def prediction_commitment_payload(
    *,
    symbol: str,
    predicted_trend: str | None,
    predicted_target_price: Decimal | None,
    price_at_prediction: Decimal,
) -> dict[str, Any]:
    """Payload commitmentu predykcji w schemacie v2 (bez `schema_version`).

    Ceny jako tekst kolumny DECIMAL(14,4) — `Money.for_storage()`, cztery
    miejsca, połówki od zera jak NUMERIC w Postgresie. Hash powstaje więc
    z wartości, którą baza przechowa, a nie z pełnej precyzji Pythona (G08).
    Brak ceny docelowej → null."""
    return {
        "symbol": symbol,
        "predicted_trend": predicted_trend,
        "predicted_target_price": _storage_text(predicted_target_price),
        "price_at_prediction": _storage_text(price_at_prediction),
    }


def _storage_text(price: Decimal | None) -> str | None:
    return None if price is None else str(Money(price).for_storage().amount)


def new_salt() -> str:
    """Generuje świeżą, kryptograficznie losową sól (hex, 32 znaki = 128 bitów).

    `secrets.token_hex(16)` — 16 bajtów entropii. Każda predykcja dostaje własną
    sól; ujawniamy ją dopiero w fazie reveal.
    """
    return secrets.token_hex(16)
