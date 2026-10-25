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
import math
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from src.domain.prediction import PREDICTION_HORIZON_HOURS
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


# WIA-01: status revealu commitmentu nadpisanego w tej samej godzinie UTC. Wchodzi
# do publicznego ledgera revealów — weryfikator odróżnia po nim nadpisanie od rozliczenia.
REVEAL_STATUS_SUPERSEDED = "superseded"


@dataclass(frozen=True)
class StoredCommitment:
    """WIA-01 — commitment, który leży już w wierszu (symbol, godzina UTC).

    `payload` to dosłowny plaintext z kolumny `commitment_payload` (043);
    None w wierszu sprzed migracji."""

    commitment: str
    salt: str
    payload: str | None

    def is_superseded_by(self, canonical: str) -> bool:
        """Czy zapis `canonical` nadpisze INNĄ predykcję niż ta zahashowana.

        Porównanie hashem z tą samą solą, nie tekstem payloadu: działa też dla
        wiersza bez plaintextu, a retry tej samej predykcji daje ten sam hash."""
        return not verify_canonical(canonical, self.salt, self.commitment)


def prediction_commitment_payload(
    *,
    symbol: str,
    predicted_trend: str | None,
    predicted_target_price: Decimal | None,
    price_at_prediction: Decimal,
    confidence_score: float | None = None,
    horizon_hours: int = PREDICTION_HORIZON_HOURS,
    council_recommendation: str | None = None,
    council_consensus_strength: float | None = None,
    recipe_version: str | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Payload commitmentu predykcji w schemacie v2 (bez `schema_version`).

    Ceny jako tekst kolumny DECIMAL(14,4) — `Money.for_storage()`, cztery
    miejsca, połówki od zera jak NUMERIC w Postgresie. Hash powstaje więc
    z wartości, którą baza przechowa, a nie z pełnej precyzji Pythona (G08).

    WIA-41: wiąże też to, co scorecard eksponuje najmocniej — pewność
    (kalibracja), horyzont, werdykt rady (`council`, null bez rady), wersję
    przepisu i hash protokołu ewaluacji (WIA-42). Ułamki idą jako string
    Decimal po quantize do czterech miejsc, nigdy jako float: repr floata
    i jego błąd binarny nie mogą decydować o hashu. Brak wartości → null."""
    council = (
        None
        if council_recommendation is None
        else {
            "final_recommendation": council_recommendation,
            "consensus_strength": _unit_text(council_consensus_strength),
        }
    )
    return {
        "symbol": symbol,
        "predicted_trend": predicted_trend,
        "predicted_target_price": _storage_text(predicted_target_price),
        "price_at_prediction": _storage_text(price_at_prediction),
        "confidence_score": _unit_text(confidence_score),
        "horizon_hours": horizon_hours,
        "council": council,
        "recipe_version": recipe_version,
        "protocol_sha256": protocol_sha256,
    }


def commitment_ledger_record(
    *,
    symbol: str,
    commitment: str,
    prediction_id: str,
    committed_at: datetime,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Linia JSONL ledgera commitmentów (WIA-25), publikowana PO zapisie predykcji.

    `prediction_id` paruje linię z wierszem `prediction_logs`, `committed_at` (UTC,
    ISO 8601) mówi, kiedy hash trafił do ledgera, a `schema_version` — którą regułą
    kanonizacji go policzono, żeby weryfikator nie zgadywał. Hash zostaje tym, co
    paruje linię z revealem (`commitment`), więc linie sprzed WIA-25 (tylko symbol
    i hash) weryfikuje się dalej tak samo.

    Pola linii NIE wchodzą do hasha: dopisanie ich nie zmienia kanonizacji i nie
    podbija `SCHEMA_VERSION`. Czas bez strefy to błąd wywołującego (`ValueError`) —
    „lokalny" czas runnera w publicznym ledgerze byłby dwuznaczny."""
    if committed_at.tzinfo is None:
        raise ValueError("committed_at wymaga strefy czasowej (UTC)")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Nieznany {_SCHEMA_KEY}: {schema_version}")
    return {
        "symbol": symbol,
        "commitment": commitment,
        "prediction_id": prediction_id,
        "committed_at": committed_at.astimezone(UTC).isoformat(timespec="seconds"),
        _SCHEMA_KEY: schema_version,
    }


def _storage_text(price: Decimal | None) -> str | None:
    return None if price is None else str(Money(price).for_storage().amount)


# Skala ułamków [0, 1] w payloadzie v2 (pewność, siła konsensusu rady).
_UNIT_SCALE = Decimal("0.0001")


def _unit_text(value: float | None) -> str | None:
    """Ułamek jako string Decimal z czterema miejscami (połówki od zera).

    `str(float)` daje najkrótszy repr, więc 0.85 → "0.8500", a nie binarne
    rozwinięcie. Wartość nieskończona (spoza kontraktu) → null."""
    if value is None or not math.isfinite(value):
        return None
    return str(Decimal(str(value)).quantize(_UNIT_SCALE, rounding=ROUND_HALF_UP))


def new_salt() -> str:
    """Generuje świeżą, kryptograficznie losową sól (hex, 32 znaki = 128 bitów).

    `secrets.token_hex(16)` — 16 bajtów entropii. Każda predykcja dostaje własną
    sól; ujawniamy ją dopiero w fazie reveal.
    """
    return secrets.token_hex(16)


# --- WIA-60: drzewo Merkle'a cyklu ----------------------------------------------------
#
# Zamiast N osobnych linii commitmentów cykl publikuje JEDNĄ linię z rootem drzewa,
# którego liśćmi są commitmenty predykcji cyklu (te same hashe z solą co dotąd).
# Root to jedno zdarzenie do zakotwiczenia (git, OTS, Rekor), a `n_leaves` zamyka zbiór:
# po publikacji nie da się dopisać predykcji ani przemilczeć przegranej. Reveal niesie
# ścieżkę dowodu (inclusion proof), więc da się go sprawdzić względem samego rootu.
#
# Konstrukcja ZAMROŻONA jak kanonizacja (golden testy) — RFC 9162, §2.1:
#   MTH({})   = SHA-256()
#   MTH({d})  = SHA-256(0x00 ‖ d)                — liść, d = 32 bajty commitmentu
#   MTH(D[n]) = SHA-256(0x01 ‖ MTH(D[:k]) ‖ MTH(D[k:])), k = największa potęga 2 < n
# Prefiksy 0x00/0x01 rozdzielają liście od węzłów (bez nich węzeł dałoby się podać
# jako liść). Liście cyklu są posortowane i bez powtórzeń: ten sam zbiór predykcji
# daje ten sam root niezależnie od kolejności zapisu przy `symbol_concurrency > 1`.
MERKLE_TREE = "rfc9162-sha256"

# Status revealu predykcji, której cykl nie opublikował rootu (job przerwany przed
# końcem cyklu albo push ledgera padł): commitment był tylko w bazie, więc reveal
# niczego nie dowodzi — i mówi to wprost zamiast udawać zwykły reveal.
REVEAL_STATUS_INCOMPLETE_CYCLE = "incomplete_cycle"

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def _hex_bytes(value: str) -> bytes:
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"Merkle: oczekiwany hex, jest {value!r}") from exc


def _leaf_hash(leaf: str) -> bytes:
    return hashlib.sha256(_LEAF_PREFIX + _hex_bytes(leaf)).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def _split(size: int) -> int:
    """Największa potęga dwójki mniejsza od `size` (size ≥ 2)."""
    return 1 << ((size - 1).bit_length() - 1)


def _subtree_hash(leaves: Sequence[str]) -> bytes:
    if len(leaves) == 1:
        return _leaf_hash(leaves[0])
    k = _split(len(leaves))
    return _node_hash(_subtree_hash(leaves[:k]), _subtree_hash(leaves[k:]))


def merkle_root(leaves: Sequence[str]) -> str:
    """Root (hex) drzewa z liści `leaves` w podanej kolejności; pusty cykl = SHA-256()."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    return _subtree_hash(leaves).hex()


def inclusion_proof(leaves: Sequence[str], index: int) -> tuple[str, ...]:
    """Ścieżka dowodu liścia `index` (RFC 9162 §2.1.3.1), od liścia w stronę rootu."""
    if not 0 <= index < len(leaves):
        raise IndexError(f"Merkle: liść {index} poza drzewem o {len(leaves)} liściach")
    path: list[bytes] = []
    while len(leaves) > 1:
        k = _split(len(leaves))
        if index < k:
            path.append(_subtree_hash(leaves[k:]))
            leaves = leaves[:k]
        else:
            path.append(_subtree_hash(leaves[:k]))
            leaves, index = leaves[k:], index - k
    return tuple(node.hex() for node in reversed(path))


def verify_inclusion(
    leaf: str, index: int, tree_size: int, proof: Sequence[str], root: str
) -> bool:
    """Czy `proof` dowodzi, że `leaf` jest liściem `index` drzewa o `tree_size` liściach
    z rootem `root` (algorytm RFC 9162 §2.1.3.2). Tekst, który nie jest hexem = False."""
    if not 0 <= index < tree_size:
        return False
    try:
        computed = _leaf_hash(leaf)
        path = [_hex_bytes(node) for node in proof]
    except ValueError:
        return False
    fn, sn = index, tree_size - 1
    for node in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            computed = _node_hash(node, computed)
            while not fn & 1 and fn != 0:
                fn, sn = fn >> 1, sn >> 1
        else:
            computed = _node_hash(computed, node)
        fn, sn = fn >> 1, sn >> 1
    return sn == 0 and hmac.compare_digest(computed.hex(), root.lower())


# WIA-73: artefakt wpisu `hashedrekord` w logu przejrzystości (Sigstore Rekor) to 32 bajty
# rootu cyklu, nie jego tekst hex: wpis wiąże dokładnie root z linii ledgera, a weryfikator
# liczy ten sam SHA-256. ZAMROŻONE jak kanonizacja — zmiana reguły odcina weryfikację
# opublikowanych wpisów.
_ROOT_BYTES = 32


def root_artifact(root: str) -> bytes:
    """32 bajty rootu cyklu — artefakt podpisywany i logowany w Rekorze."""
    artifact = _hex_bytes(root)
    if len(artifact) != _ROOT_BYTES:
        raise ValueError(f"Root cyklu ma {len(artifact)} bajtów zamiast {_ROOT_BYTES}")
    return artifact


def root_artifact_sha256(root: str) -> str:
    """SHA-256 (hex) artefaktu rootu — pole `spec.data.hash.value` wpisu `hashedrekord`."""
    return hashlib.sha256(root_artifact(root)).hexdigest()


def cycle_leaves(commitments: Iterable[str]) -> tuple[str, ...]:
    """Liście cyklu: commitmenty małymi literami, posortowane, bez powtórzeń (retry
    zapisu tej samej predykcji daje ten sam hash)."""
    return tuple(sorted({commitment.lower() for commitment in commitments}))


def cycle_root_record(
    *, cycle_id: str, committed_at: datetime, commitments: Iterable[str]
) -> dict[str, Any]:
    """WIA-60 — jedna linia ledgera na cykl, także dla pustego zbioru predykcji.

    `leaves` jest w linii jawnie: liście to commitmenty z solą, publikowane dotąd po
    jednym w linii, więc nic nie ujawniają — a weryfikator odtwarza z nich root
    i `n_leaves`. `schema_version` mówi, którą kanonizacją policzono liście (pola linii
    nie wchodzą do żadnego hasha), `tree` — konstrukcję drzewa."""
    if committed_at.tzinfo is None:
        raise ValueError("committed_at wymaga strefy czasowej (UTC)")
    leaves = cycle_leaves(commitments)
    return {
        "cycle_id": cycle_id,
        "committed_at": committed_at.astimezone(UTC).isoformat(timespec="seconds"),
        "root": merkle_root(leaves),
        "n_leaves": len(leaves),
        "leaves": list(leaves),
        _SCHEMA_KEY: SCHEMA_VERSION,
        "tree": MERKLE_TREE,
    }


@dataclass(frozen=True)
class CycleInclusion:
    """Dowód, że commitment jest liściem drzewa cyklu `cycle_id` z rootem `root`."""

    cycle_id: str
    root: str
    leaf_index: int
    tree_size: int
    proof: tuple[str, ...]

    def verifies(self, commitment: str) -> bool:
        return verify_inclusion(
            commitment.lower(), self.leaf_index, self.tree_size, self.proof, self.root
        )

    def as_record(self) -> dict[str, Any]:
        """Pole `inclusion` revealu."""
        return {
            "cycle_id": self.cycle_id,
            "root": self.root,
            "leaf_index": self.leaf_index,
            "tree_size": self.tree_size,
            "proof": list(self.proof),
        }


@dataclass(frozen=True)
class CycleRoot:
    """Linia rootu cyklu odczytana z ledgera (z deklarowanym `n_leaves`)."""

    cycle_id: str
    root: str
    leaves: tuple[str, ...]
    n_leaves: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CycleRoot | None:
        """Linia rootu albo None, gdy to inna (np. pojedyncza linia sprzed WIA-60) lub
        źle uformowana linia. Brak `n_leaves` = liczba liści z linii."""
        cycle_id, root, leaves = record.get("cycle_id"), record.get("root"), record.get("leaves")
        if not (isinstance(cycle_id, str) and isinstance(root, str) and isinstance(leaves, list)):
            return None
        if not all(isinstance(leaf, str) for leaf in leaves):
            return None
        n_leaves = record.get("n_leaves", len(leaves))
        if not isinstance(n_leaves, int):
            return None
        return cls(
            cycle_id=cycle_id,
            root=root.lower(),
            leaves=tuple(leaf.lower() for leaf in leaves),
            n_leaves=n_leaves,
        )

    @property
    def consistent(self) -> bool:
        """Czy root i `n_leaves` zgadzają się z liśćmi linii (inaczej linia kłamie)."""
        try:
            computed = merkle_root(self.leaves)
        except ValueError:
            return False
        return self.n_leaves == len(self.leaves) and hmac.compare_digest(computed, self.root)

    def inclusion_of(self, commitment: str) -> CycleInclusion | None:
        try:
            index = self.leaves.index(commitment.lower())
        except ValueError:
            return None
        return CycleInclusion(
            cycle_id=self.cycle_id,
            root=self.root,
            leaf_index=index,
            tree_size=len(self.leaves),
            proof=inclusion_proof(self.leaves, index),
        )


@dataclass(frozen=True)
class CommitmentLedger:
    """Opublikowane commitmenty: pojedyncze linie (sprzed WIA-60) i rooty cykli."""

    single: frozenset[str]
    cycles: tuple[CycleRoot, ...]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> CommitmentLedger:
        single: set[str] = set()
        cycles: list[CycleRoot] = []
        for record in records:
            cycle = CycleRoot.from_record(record)
            if cycle is not None:
                cycles.append(cycle)
            elif isinstance(commitment := record.get("commitment"), str) and commitment:
                single.add(commitment.lower())
        return cls(single=frozenset(single), cycles=tuple(cycles))

    @property
    def published(self) -> frozenset[str]:
        return self.single | {leaf for cycle in self.cycles for leaf in cycle.leaves}

    def is_published(self, commitment: str) -> bool:
        return commitment.lower() in self.published

    def inclusion_of(self, commitment: str) -> CycleInclusion | None:
        """Dowód z pierwszego cyklu, którego liściem jest commitment (None = brak)."""
        for cycle in self.cycles:
            inclusion = cycle.inclusion_of(commitment)
            if inclusion is not None:
                return inclusion
        return None
