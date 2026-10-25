"""WIA-26: weryfikacja ledgera commit-reveal przez osobę z zewnątrz (czyste funkcje).

Czyta zawartość `commitments.jsonl` i `reveals.jsonl` i dla każdego revealu sprawdza:

1. czy jego commitment jest w ledgerze commitmentów (reveal bez opublikowanego
   hasha niczego nie dowodzi),
2. czy SHA-256(`payload` + `salt`) odtwarza ten hash (`src.domain.attestation`),
3. czy kolumny revealu (symbol, trend, ceny) zgadzają się z zahashowanym plaintextem
   — inaczej wiersz mógłby po fakcie mówić co innego niż commitment.

Reveal sprzed migracji 043 nie ma `payload`; wtedy payload v1 odtwarza się z kolumn.
Linie commitmentów sprzed WIA-25 (tylko symbol i hash) czyta się tak samo — pole
łączące je z revealem to `commitment`.

WIA-60: od rootów Merkle'a linia commitmentów to jedna linia na cykl (`cycle_id`, `root`,
`n_leaves`, `leaves`). Weryfikator odtwarza root z liści (root albo `n_leaves` niezgodne
z liśćmi = porażka ledgera), liczy liście jako opublikowane commitmenty, a reveal ze ścieżką
dowodu (`inclusion`) sprawdza względem rootu, który ledger faktycznie opublikował dla tego
cyklu. Reveal ze statusem `incomplete_cycle` przyznaje, że root jego cyklu nigdy nie wyszedł:
jest jawny, ale niczego nie dowodzi, więc werdykt pozostaje FAIL z osobną kategorią.

WIA-73: linia rootu z polem `rekor` (wpis Sigstore Rekor) musi go potwierdzać —
`verify_rekor.rekor_problems` sprawdza powiązanie z rootem i podpis, inclusion proof, czas
wpisu i tożsamość workflowu. Linia bez wpisu nie jest porażką (flaga mogła być wyłączona);
liczba zakotwiczonych rootów trafia do raportu.

Commitment bez revealu jest „jeszcze nieujawniony", nie porażką: predykcja może się nie
rozliczyć. Liczba takich wpisów trafia do raportu, żeby długo nieujawniane commitmenty
były widoczne. Czytanie plików i kod wyjścia należą do `main_tools.py verify-attestation`
(ARCH-39); ten moduł nie robi I/O.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from src.domain.attestation import (
    REVEAL_STATUS_INCOMPLETE_CYCLE,
    SCHEMA_VERSION_V1,
    CycleRoot,
    verify,
    verify_canonical,
    verify_inclusion,
)
from src.tools.verify_rekor import (
    FAST_LOOP_IDENTITY,
    REKOR_FIELD,
    WorkflowIdentity,
    rekor_problems,
)

# Te same ścieżki co `attestation_commitments_path` / `attestation_reveals_path` w config.toml
# (pilnuje tego test); narzędzie nie ładuje `Settings`, bo weryfikator nie ma mieć kluczy.
DEFAULT_COMMITMENTS_PATH = "public/attestation/commitments.jsonl"
DEFAULT_REVEALS_PATH = "public/attestation/reveals.jsonl"

STATUS_VALID = "valid"
STATUS_TAMPERED = "tampered"
STATUS_UNPUBLISHED = "no published commitment"
STATUS_UNVERIFIABLE = "unverifiable"
STATUS_INCOMPLETE_CYCLE = "incomplete cycle"

# Kolumny revealu, które payload v2 zawiera dosłownie (tekst) albo jako cenę (4 miejsca).
_TEXT_FIELDS = ("symbol", "predicted_trend")
_PRICE_FIELDS = ("predicted_target_price", "price_at_prediction")
_LEGACY_NOTE = "legacy reveal without payload, rebuilt from its columns (schema 1)"


class _Outcome(Protocol):
    """Fabryka wyniku dla jednego revealu (commitment i symbol są już wpisane)."""

    def __call__(self, status: str, detail: str = "") -> RevealCheck: ...


@dataclass(frozen=True)
class RevealCheck:
    """Wynik sprawdzenia jednego revealu."""

    commitment: str
    symbol: str
    status: str
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.status == STATUS_VALID


@dataclass(frozen=True)
class LedgerReport:
    """Wynik weryfikacji całego ledgera."""

    checks: tuple[RevealCheck, ...]
    commitments: int
    pending: int
    problems: tuple[str, ...] = ()
    cycles: int = 0
    rekor_entries: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and all(check.valid for check in self.checks)


def parse_jsonl(text: str, source: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Rekordy (obiekty JSON) z tekstu JSONL i miejsca uszkodzonych linii (`źródło:nr`).

    Puste linie są pomijane; linia, która nie jest obiektem JSON, to uszkodzenie."""
    records: list[dict[str, Any]] = []
    malformed: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            malformed.append(f"{source}:{number}")
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            malformed.append(f"{source}:{number}")
    return records, malformed


def verify_ledger(
    commitments: Iterable[Mapping[str, Any]],
    reveals: Iterable[Mapping[str, Any]],
    *,
    malformed: Sequence[str] = (),
    rekor_identity: WorkflowIdentity = FAST_LOOP_IDENTITY,
) -> LedgerReport:
    """Sprawdza każdy reveal względem opublikowanych commitmentów.

    `malformed` to miejsca uszkodzonych linii z `parse_jsonl` — każda jest porażką,
    bo ledger, którego nie da się w pełni przeczytać, niczego nie dowodzi. Linia rootu
    z wpisem Rekora (WIA-73) musi go potwierdzać dla workflowu `rekor_identity`."""
    problems = [f"unreadable line {place}" for place in malformed]
    published: set[str] = set()
    roots: dict[str, str] = {}
    rekor_entries = 0
    for record in commitments:
        cycle = CycleRoot.from_record(record)
        if cycle is not None:
            problems.extend(_cycle_problems(cycle, roots))
            published.update(cycle.leaves)
            if REKOR_FIELD in record:
                rekor_entries += 1
                problems.extend(rekor_problems(record, cycle.root, rekor_identity))
            continue
        commitment = _hash_of(record)
        if commitment is None:
            problems.append(f"commitment line without a hash (symbol {record.get('symbol')!r})")
        else:
            published.add(commitment)

    checks: list[RevealCheck] = []
    seen: set[str] = set()
    for reveal in reveals:
        # Ten sam reveal dopisany dwa razy (retry sweepu) to jeden dowód.
        fingerprint = json.dumps(dict(reveal), sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        checks.append(_check_reveal(reveal, published, roots))

    revealed = {check.commitment for check in checks}
    return LedgerReport(
        checks=tuple(checks),
        commitments=len(published),
        pending=len(published - revealed),
        problems=tuple(problems),
        cycles=len(roots),
        rekor_entries=rekor_entries,
    )


def _cycle_problems(cycle: CycleRoot, roots: dict[str, str]) -> list[str]:
    """Root niezgodny z liśćmi albo drugi, inny root tego samego cyklu (zapamiętuje root)."""
    problems: list[str] = []
    if not cycle.consistent:
        problems.append(f"cycle {cycle.cycle_id}: root or n_leaves does not match its leaves")
    known = roots.setdefault(cycle.cycle_id, cycle.root)
    if known != cycle.root:
        problems.append(f"cycle {cycle.cycle_id}: published twice with different roots")
    return problems


def _hash_of(record: Mapping[str, Any]) -> str | None:
    value = record.get("commitment")
    return value.lower() if isinstance(value, str) and value else None


def _check_reveal(
    reveal: Mapping[str, Any], published: set[str], roots: Mapping[str, str]
) -> RevealCheck:
    commitment = _hash_of(reveal) or ""
    symbol = str(reveal.get("symbol") or "?")

    def result(status: str, detail: str = "") -> RevealCheck:
        return RevealCheck(commitment=commitment, symbol=symbol, status=status, detail=detail)

    if commitment not in published:
        if reveal.get("status") == REVEAL_STATUS_INCOMPLETE_CYCLE:
            return result(
                STATUS_INCOMPLETE_CYCLE,
                "the root of its cycle was never published, so the reveal proves nothing",
            )
        return result(STATUS_UNPUBLISHED, "the reveal's hash is not in the commitment ledger")
    inclusion = reveal.get("inclusion")
    if inclusion is not None and not _proof_reaches_published_root(commitment, inclusion, roots):
        return result(
            STATUS_TAMPERED, "the inclusion proof does not reach the published cycle root"
        )
    salt = reveal.get("salt")
    if not isinstance(salt, str) or not salt:
        return result(STATUS_TAMPERED, "the reveal has no salt")
    payload = reveal.get("payload")
    if isinstance(payload, str):
        return _check_with_payload(reveal, payload, salt, commitment, result)
    return _check_legacy(reveal, salt, commitment, result)


def _proof_reaches_published_root(
    commitment: str, inclusion: Any, roots: Mapping[str, str]
) -> bool:
    """Ścieżka dowodu revealu prowadzi do rootu, który ledger opublikował dla jego cyklu."""
    if not isinstance(inclusion, Mapping):
        return False
    root = roots.get(str(inclusion.get("cycle_id")))
    index, size, proof = (
        inclusion.get("leaf_index"),
        inclusion.get("tree_size"),
        inclusion.get("proof"),
    )
    if root is None or not isinstance(index, int) or not isinstance(size, int):
        return False
    if not isinstance(proof, list) or not all(isinstance(node, str) for node in proof):
        return False
    return inclusion.get("root") in (None, root) and verify_inclusion(
        commitment, index, size, proof, root
    )


def _check_with_payload(
    reveal: Mapping[str, Any], payload: str, salt: str, commitment: str, result: _Outcome
) -> RevealCheck:
    if not verify_canonical(payload, salt, commitment):
        return result(STATUS_TAMPERED, "SHA-256(payload + salt) differs from the commitment")
    contradiction = _contradicted_field(reveal, payload)
    if contradiction:
        return result(STATUS_TAMPERED, contradiction)
    return result(STATUS_VALID)


def _check_legacy(
    reveal: Mapping[str, Any], salt: str, commitment: str, result: _Outcome
) -> RevealCheck:
    """Reveal sprzed 043: payload v1 z czterech kolumn, ceny jako `str()` wartości."""
    rebuilt = {
        "symbol": reveal.get("symbol"),
        "predicted_trend": reveal.get("predicted_trend"),
        "predicted_target_price": str(reveal.get("predicted_target_price")),
        "price_at_prediction": str(reveal.get("price_at_prediction")),
    }
    if verify(rebuilt, salt, commitment, schema_version=SCHEMA_VERSION_V1):
        return result(STATUS_VALID, _LEGACY_NOTE)
    return result(
        STATUS_UNVERIFIABLE,
        "legacy reveal without payload does not reproduce the hash from its columns "
        "(prices may have lost precision in the database)",
    )


def _contradicted_field(reveal: Mapping[str, Any], payload: str) -> str:
    """Opis pierwszej kolumny revealu sprzecznej z zahashowanym plaintextem ('' = zgodne)."""
    try:
        committed = json.loads(payload)
    except ValueError:
        return "the payload is not JSON"
    if not isinstance(committed, dict):
        return "the payload is not a JSON object"
    for field in _TEXT_FIELDS:
        if field in reveal and reveal[field] != committed.get(field):
            return f"{field} in the reveal differs from the committed payload"
    for field in _PRICE_FIELDS:
        if field in reveal and not _same_price(reveal[field], committed.get(field)):
            return f"{field} in the reveal differs from the committed payload"
    return ""


def _same_price(revealed: Any, committed: Any) -> bool:
    """Cena z kolumny (liczba z PostgREST) kontra tekst o czterech miejscach z payloadu."""
    if revealed is None or committed is None:
        return revealed is None and committed is None
    try:
        return Decimal(str(revealed)) == Decimal(str(committed))
    except InvalidOperation:
        return False


def format_report(report: LedgerReport) -> str:
    """Raport dla człowieka; ostatnia linia to werdykt zaczynający się od `OK` albo `FAIL`."""
    lines = [
        f"Commitments in the ledger: {report.commitments} ({report.pending} not revealed yet)",
        f"Reveals checked: {len(report.checks)}",
    ]
    if report.cycles:
        lines.insert(1, f"Cycle roots: {report.cycles}")
    if report.rekor_entries:
        lines.insert(2, f"Cycle roots logged in Rekor: {report.rekor_entries}")
    lines.extend(f"  ! {problem}" for problem in report.problems)
    for check in report.checks:
        if not check.valid:
            lines.append(
                f"  FAIL {check.symbol} {check.commitment[:12]}…  {check.status}: {check.detail}"
            )
    bad = sum(1 for check in report.checks if not check.valid)
    if report.ok:
        lines.append(
            f"OK: all {len(report.checks)} reveals reproduce their published commitments"
        )
    else:
        lines.append(
            f"FAIL: {bad} of {len(report.checks)} reveals do not reproduce their commitments"
            f"{f', {len(report.problems)} ledger problem(s)' if report.problems else ''}"
        )
    return "\n".join(lines)
