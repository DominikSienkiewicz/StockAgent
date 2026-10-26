"""WIA-73: sprawdzenie wpisu Sigstore Rekor w linii rootu cyklu (czyste funkcje, offline).

`RekorAttestationPublisher` dopisuje do linii rootu cyklu pole `rekor` — odpowiedź Rekora
na wpis `hashedrekord` rootu. Weryfikator sprawdza z samego pliku ledgera:

1. czy wpis loguje TEN root: `spec.data.hash` = SHA-256 32 bajtów rootu
   (`root_artifact_sha256`), a podpis w ciele wpisu weryfikuje się kluczem certyfikatu,
2. inclusion proof: ciało wpisu jest liściem drzewa RFC 9162 o korzeniu `rootHash`
   (`verify_inclusion`), a checkpoint wpisu nazywa ten sam rozmiar i korzeń,
3. czas wpisu: `integratedTime` nadany przez Rekor leży w oknie `committed_at` linii
   (od `MAX_CLOCK_SKEW` przed do `MAX_LOG_DELAY` po) i w ważności certyfikatu,
4. tożsamość workflowu: SAN certyfikatu to URI pliku workflowu w repo
   (`https://github.com/<repo>/<ścieżka>@<ref>`), a issuer to OIDC GitHub Actions.

Czego offline NIE sprawdza (UCZCIWOŚĆ CLAIMU): łańcucha certyfikatu do roota Fulcio
ani podpisu checkpointu i SET-u kluczem Rekora — to wymaga zaufanego materiału Sigstore
(TUF). Bez tego zielony wynik mówi, że wpis jest spójny i nazywa właściwy workflow, a to,
że Rekor go faktycznie zawiera, potwierdza `rekor-cli get --uuid <uuid>` albo
search.sigstore.dev (instrukcja w docs/configuration.md).
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from src.domain.attestation import root_artifact, root_artifact_sha256, verify_inclusion

REKOR_FIELD = "rekor"
GITHUB_ACTIONS_ISSUER = "https://token.actions.githubusercontent.com"
# Tożsamość, którą obiecuje docs/configuration.md: Fast Loop tego repo.
DEFAULT_REKOR_REPOSITORY = "DominikSienkiewicz/StockAgent"
DEFAULT_REKOR_WORKFLOW = ".github/workflows/fast_loop_12h.yml"
# Rekor dostaje root sekundy po `committed_at` (seal → token → Fulcio → Rekor); okno
# z zapasem na retry. Root zalogowany później niż kwadrans po deklarowanym czasie linii
# nie potwierdza tego czasu.
MAX_LOG_DELAY = timedelta(minutes=15)
MAX_CLOCK_SKEW = timedelta(minutes=5)
# Rozszerzenia Sigstore z issuerem OIDC: v1 (surowy tekst) i v2 (DER UTF8String).
_ISSUER_V1_OID = x509.ObjectIdentifier("1.3.6.1.4.1.57264.1.1")
_ISSUER_V2_OID = x509.ObjectIdentifier("1.3.6.1.4.1.57264.1.8")
_UTF8_STRING_TAG = 0x0C
_SHORT_LENGTH_LIMIT = 0x80
_CHECKPOINT_SIZE_LINE = 1
_CHECKPOINT_ROOT_LINE = 2


@dataclass(frozen=True)
class WorkflowIdentity:
    """Workflow, którego certyfikat Fulcio ma podpisać root: repo i ścieżka pliku."""

    repository: str = DEFAULT_REKOR_REPOSITORY
    workflow: str = DEFAULT_REKOR_WORKFLOW
    issuer: str = GITHUB_ACTIONS_ISSUER

    @property
    def uri_prefix(self) -> str:
        """Początek SAN certyfikatu GitHub Actions; po `@` stoi ref (gałąź albo tag)."""
        return f"https://github.com/{self.repository}/{self.workflow}@"


FAST_LOOP_IDENTITY = WorkflowIdentity()


class _EntryError(ValueError):
    """Wpis, którego nie da się odczytać — jeden opis problemu zamiast wyjątku."""


@dataclass(frozen=True)
class _Entry:
    body: bytes
    spec: Mapping[str, Any]
    integrated_at: datetime
    proof: Mapping[str, Any]
    certificate: x509.Certificate


def rekor_problems(
    line: Mapping[str, Any], root: str, identity: WorkflowIdentity = FAST_LOOP_IDENTITY
) -> list[str]:
    """Problemy wpisu Rekora linii rootu cyklu `root` ([] = wpis potwierdza root)."""
    cycle = f"cycle {line.get('cycle_id')}"
    try:
        root_artifact(root)
        entry = _read_entry(line.get(REKOR_FIELD))
    except ValueError as exc:
        return [f"{cycle}: unreadable Rekor entry ({exc})"]
    checks = (
        _binding_problems(entry, root),
        _inclusion_problems(entry),
        _time_problems(entry, line.get("committed_at")),
        _identity_problems(entry, identity),
    )
    return [f"{cycle}: {problem}" for problems in checks for problem in problems]


def _object(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _EntryError(f"{what} is not an object")
    return value


def _read_entry(raw: Any) -> _Entry:
    entry = _object(raw, "the entry")
    try:
        body = base64.b64decode(str(entry.get("body")), validate=True)
        spec = _object(_object(json.loads(body), "the body").get("spec"), "spec")
        content = _object(_object(spec.get("signature"), "signature").get("publicKey"), "key")
        pem = base64.b64decode(str(content.get("content")), validate=True)
        certificate = x509.load_pem_x509_certificate(pem)
    except (binascii.Error, ValueError) as exc:
        raise _EntryError(str(exc)) from exc
    integrated = entry.get("integratedTime")
    if not isinstance(integrated, int) or isinstance(integrated, bool):
        raise _EntryError("integratedTime is not a number")
    verification = _object(entry.get("verification"), "verification")
    proof = _object(verification.get("inclusionProof"), "inclusionProof")
    return _Entry(
        body=body,
        spec=spec,
        integrated_at=datetime.fromtimestamp(integrated, UTC),
        proof=proof,
        certificate=certificate,
    )


def _binding_problems(entry: _Entry, root: str) -> list[str]:
    """Wpis loguje SHA-256 bajtów rootu i podpis tych bajtów kluczem certyfikatu."""
    expected = {"algorithm": "sha256", "value": root_artifact_sha256(root)}
    data = entry.spec.get("data")
    if not isinstance(data, Mapping) or data.get("hash") != expected:
        return ["the Rekor entry does not log this cycle root"]
    signature = entry.spec.get("signature")
    content = signature.get("content") if isinstance(signature, Mapping) else None
    key = entry.certificate.public_key()
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(content, str):
        return ["the Rekor entry has no ECDSA signature to check"]
    try:
        key.verify(
            base64.b64decode(content, validate=True),
            root_artifact(root),
            ec.ECDSA(hashes.SHA256()),
        )
    except (InvalidSignature, binascii.Error, ValueError):
        return ["the signature in the Rekor entry does not verify over the root"]
    return []


def _inclusion_problems(entry: _Entry) -> list[str]:
    """Ciało wpisu jest liściem drzewa logu, a checkpoint nazywa to samo drzewo."""
    index, size = entry.proof.get("logIndex"), entry.proof.get("treeSize")
    tree_root, path = entry.proof.get("rootHash"), entry.proof.get("hashes")
    if not (
        isinstance(index, int)
        and isinstance(size, int)
        and isinstance(tree_root, str)
        and isinstance(path, list)
        and all(isinstance(node, str) for node in path)
    ):
        return ["the Rekor inclusion proof is incomplete"]
    # Liść RFC 9162 to dowolne bajty — tu ciało wpisu, podane jako hex.
    if not verify_inclusion(entry.body.hex(), index, size, path, tree_root):
        return ["the Rekor inclusion proof does not reach its tree root"]
    if not _checkpoint_names(entry.proof.get("checkpoint"), size, tree_root):
        return ["the Rekor checkpoint does not name the tree of the inclusion proof"]
    return []


def _checkpoint_names(checkpoint: Any, size: int, tree_root: str) -> bool:
    """Checkpoint (signed note): pochodzenie, rozmiar drzewa, korzeń base64, podpisy."""
    if not isinstance(checkpoint, str):
        return False
    lines = checkpoint.split("\n")
    if len(lines) <= _CHECKPOINT_ROOT_LINE:
        return False
    try:
        named_root = base64.b64decode(lines[_CHECKPOINT_ROOT_LINE], validate=True)
        root_bytes = bytes.fromhex(tree_root)
    except (binascii.Error, ValueError):
        return False
    return lines[_CHECKPOINT_SIZE_LINE] == str(size) and named_root == root_bytes


def _time_problems(entry: _Entry, committed_at: Any) -> list[str]:
    """`integratedTime` w oknie deklarowanego `committed_at` i w ważności certyfikatu."""
    problems: list[str] = []
    try:
        declared = datetime.fromisoformat(str(committed_at))
    except ValueError:
        return ["the line has no committed_at to compare with the Rekor time"]
    if declared.tzinfo is None:
        return ["the line's committed_at has no time zone"]
    logged = entry.integrated_at
    if not declared - MAX_CLOCK_SKEW <= logged <= declared + MAX_LOG_DELAY:
        problems.append(
            f"Rekor logged the root at {logged.isoformat()}, outside the window of its "
            f"committed_at {declared.astimezone(UTC).isoformat()}"
        )
    certificate = entry.certificate
    if not certificate.not_valid_before_utc <= logged <= certificate.not_valid_after_utc:
        problems.append("Rekor logged the root outside the validity of its certificate")
    return problems


def _identity_problems(entry: _Entry, identity: WorkflowIdentity) -> list[str]:
    """SAN certyfikatu nazywa workflow `identity`, a issuer to OIDC GitHub Actions."""
    problems: list[str] = []
    uris = _san_uris(entry.certificate)
    if not any(uri.startswith(identity.uri_prefix) for uri in uris):
        problems.append(
            f"the Rekor certificate names {', '.join(uris) or 'no workflow'} instead of the "
            f"workflow {identity.uri_prefix}…"
        )
    if identity.issuer not in _issuers(entry.certificate):
        problems.append(f"the Rekor certificate was not issued for {identity.issuer}")
    return problems


def _san_uris(certificate: x509.Certificate) -> list[str]:
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return []
    return list(san.value.get_values_for_type(x509.UniformResourceIdentifier))


def _issuers(certificate: x509.Certificate) -> set[str]:
    """Issuer OIDC z rozszerzeń Sigstore: v1 to surowy tekst, v2 to DER UTF8String."""
    found: set[str] = set()
    for extension in certificate.extensions:
        value = extension.value
        if not isinstance(value, x509.UnrecognizedExtension):
            continue
        if extension.oid == _ISSUER_V1_OID:
            found.add(_text(value.value))
        elif extension.oid == _ISSUER_V2_OID:
            found.add(_utf8_string(value.value))
    return found - {""}


def _text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _utf8_string(der: bytes) -> str:
    """Krótki DER UTF8String (tag 0x0C, długość < 128) — URL issuera się w nim mieści."""
    if len(der) < 2 or der[0] != _UTF8_STRING_TAG or der[1] >= _SHORT_LENGTH_LIMIT:
        return ""
    length = der[1]
    if len(der) != 2 + length:
        return ""
    return _text(der[2:])
