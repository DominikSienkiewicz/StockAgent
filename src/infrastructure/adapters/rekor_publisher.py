"""WIA-73 — root cyklu w publicznym logu przejrzystości Sigstore Rekor.

`RekorAttestationPublisher` owija publisher pliku tak jak `ReadOnlyAttestationPublisher`
i `CycleCommitmentCollector`, więc graf dalej ma jedno pole `attestation_publisher`.
Linię rootu cyklu (WIA-60) z co najmniej jednym liściem najpierw loguje w Rekorze jako wpis
`hashedrekord` — SHA-256 32 bajtów rootu (`root_artifact_sha256`) i podpis tych bajtów — a
dopiero potem oddaje ją publisherowi z polem `rekor`: odpowiedzią Rekora (uuid, `logIndex`,
`integratedTime`, ciało wpisu, inclusion proof), z której `main_tools.py verify-attestation`
sprawdza wpis offline. Pola linii nie wchodzą do żadnego hasha.

Podpis keyless (Sigstore), jak w oficjalnym kliencie `sigstore-python`: token OIDC joba
GitHub Actions z audience `sigstore` (job musi mieć `id-token: write`) → efemeryczny klucz
P-256 → CSR podpisany tym kluczem → certyfikat Fulcio z tożsamością workflowu (repo
i ścieżka pliku workflowu w SAN) → podpis rootu. Klucz żyje w pamięci jednego wywołania.

Linie pojedyncze (sprzed WIA-60), pusty cykl (root = SHA-256 pustego ciągu, ten sam dla
każdego pustego cyklu, więc niczego nie kotwiczy) i revealy idą prosto do publishera.
Awaria tokenu, Fulcio albo Rekora tylko loguje WARNING: linia wychodzi bez pola `rekor`,
a cykl biegnie dalej (kontrakt portu: publikacja nigdy nie wywala cyklu). Wpis w Rekorze
jest publiczny na zawsze.

FinOps: zero płatnych wywołań — publiczne instancje Sigstore są darmowe; trzy żądania HTTP
na cykl z predykcjami.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from src.application.errors import MalformedPayload, PortError
from src.application.ports import AttestationPublisherPort
from src.domain.attestation import CycleRoot, root_artifact, root_artifact_sha256
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translate_request_error,
    translated_get,
)

logger = logging.getLogger(__name__)

REKOR_URL = "https://rekor.sigstore.dev"
FULCIO_URL = "https://fulcio.sigstore.dev"
REKOR_ENTRIES_URL = f"{REKOR_URL}/api/v1/log/entries"
FULCIO_SIGNING_CERT_URL = f"{FULCIO_URL}/api/v2/signingCert"
SIGSTORE_AUDIENCE = "sigstore"
# Zmienne, które runner GitHub Actions ustawia tylko jobowi z `id-token: write`.
OIDC_REQUEST_URL_ENV = "ACTIONS_ID_TOKEN_REQUEST_URL"
OIDC_REQUEST_TOKEN_ENV = "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
# Pole linii rootu cyklu z odpowiedzią Rekora (czyta je verify-attestation).
REKOR_FIELD = "rekor"
_TIMEOUT_SECONDS = 30


def _post(session: requests.Session, url: str, **kwargs: Any) -> requests.Response:
    """POST, którego błąd sieci wychodzi w taksonomii portów (ARCH-33)."""
    try:
        return session.post(url, **kwargs)
    except requests.RequestException as exc:
        raise translate_request_error(exc) from None


def _object(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedPayload(f"{what}: oczekiwany obiekt JSON")
    return value


class GitHubActionsIdentityToken:
    """Token OIDC joba GitHub Actions z audience `sigstore`.

    Runner wystawia go tylko jobowi z `id-token: write`; bez tego uprawnienia brak
    zmiennych środowiska to `PortError`, a nie cichy brak wpisu."""

    def __init__(self, session: requests.Session, environ: Mapping[str, str]) -> None:
        self._session = session
        self._environ = environ

    def __call__(self) -> str:
        url = self._environ.get(OIDC_REQUEST_URL_ENV, "")
        bearer = self._environ.get(OIDC_REQUEST_TOKEN_ENV, "")
        if not url or not bearer:
            raise PortError(
                f"brak {OIDC_REQUEST_URL_ENV} / {OIDC_REQUEST_TOKEN_ENV} — "
                "job bez `id-token: write` nie dostaje tokenu OIDC"
            )
        response = translated_get(
            self._session,
            url,
            params={"audience": SIGSTORE_AUDIENCE},
            headers={"Authorization": f"bearer {bearer}"},
            timeout=_TIMEOUT_SECONDS,
        )
        raise_for_status_translated(response)
        token = _object(json_payload(response), "token OIDC").get("value")
        if not isinstance(token, str) or not token:
            raise MalformedPayload("token OIDC: brak pola `value`")
        return token


def token_subject(token: str) -> str:
    """Claim `sub` tokenu OIDC (bez sprawdzania podpisu — to robi Fulcio)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise MalformedPayload("token OIDC nie jest JWT")
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (binascii.Error, ValueError) as exc:
        raise MalformedPayload("token OIDC: nieczytelne claimy") from exc
    subject = _object(claims, "claimy tokenu OIDC").get("sub")
    if not isinstance(subject, str) or not subject:
        raise MalformedPayload("token OIDC bez claimu `sub`")
    return subject


@dataclass(frozen=True)
class SignedArtifact:
    """Podpis artefaktu i certyfikat Fulcio (PEM) klucza, który go złożył."""

    signature: bytes
    certificate_pem: str


class SigstoreKeylessSigner:
    """Podpis keyless: efemeryczny klucz P-256 i certyfikat Fulcio na tożsamość tokenu."""

    def __init__(
        self,
        session: requests.Session,
        identity_token: Callable[[], str],
        signing_cert_url: str = FULCIO_SIGNING_CERT_URL,
    ) -> None:
        self._session = session
        self._identity_token = identity_token
        self._signing_cert_url = signing_cert_url

    def sign(self, artifact: bytes) -> SignedArtifact:
        token = self._identity_token()
        key = ec.generate_private_key(ec.SECP256R1())
        # CSR jest dowodem posiadania klucza; temat (claim `sub`) jak w sigstore-python.
        csr = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(
                x509.Name([x509.NameAttribute(NameOID.EMAIL_ADDRESS, token_subject(token))])
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        response = _post(
            self._session,
            self._signing_cert_url,
            json={
                "credentials": {"oidcIdentityToken": token},
                "certificateSigningRequest": base64.b64encode(
                    csr.public_bytes(serialization.Encoding.PEM)
                ).decode("ascii"),
            },
            timeout=_TIMEOUT_SECONDS,
        )
        raise_for_status_translated(response)
        certificate = leaf_certificate(_object(json_payload(response), "odpowiedź Fulcio"))
        signature = key.sign(artifact, ec.ECDSA(hashes.SHA256()))
        return SignedArtifact(signature=signature, certificate_pem=certificate)


def leaf_certificate(payload: Mapping[str, Any]) -> str:
    """Pierwszy certyfikat łańcucha (liść) z SCT wbudowanym albo odłączonym."""
    signed = payload.get("signedCertificateEmbeddedSct") or payload.get(
        "signedCertificateDetachedSct"
    )
    chain = _object(signed, "odpowiedź Fulcio").get("chain")
    certificates = _object(chain, "łańcuch Fulcio").get("certificates")
    if not isinstance(certificates, list) or not certificates:
        raise MalformedPayload("odpowiedź Fulcio bez certyfikatów")
    leaf = certificates[0]
    if not isinstance(leaf, str) or "BEGIN CERTIFICATE" not in leaf:
        raise MalformedPayload("odpowiedź Fulcio: liść nie jest certyfikatem PEM")
    return leaf


def hashedrekord(artifact_sha256: str, signed: SignedArtifact) -> dict[str, Any]:
    """Proponowany wpis `hashedrekord` v0.0.1 (kształt jak w sigstore-python)."""
    return {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {"hash": {"algorithm": "sha256", "value": artifact_sha256}},
            "signature": {
                "content": base64.b64encode(signed.signature).decode("ascii"),
                "publicKey": {
                    "content": base64.b64encode(signed.certificate_pem.encode("ascii")).decode(
                        "ascii"
                    )
                },
            },
        },
    }


class RekorClient:
    """`POST /api/v1/log/entries` publicznej instancji Rekor."""

    def __init__(self, session: requests.Session, entries_url: str = REKOR_ENTRIES_URL) -> None:
        self._session = session
        self._entries_url = entries_url

    def log(self, proposed: Mapping[str, Any]) -> dict[str, Any]:
        """Wpis z logu: odpowiedź Rekora z jej `uuid` w polu obok (do linii ledgera)."""
        response = _post(
            self._session, self._entries_url, json=dict(proposed), timeout=_TIMEOUT_SECONDS
        )
        raise_for_status_translated(response)
        return log_entry(_object(json_payload(response), "odpowiedź Rekora"))


def log_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Jedyny wpis odpowiedzi Rekora (mapa uuid → wpis) z polem `uuid`."""
    if len(payload) != 1:
        raise MalformedPayload(f"odpowiedź Rekora: {len(payload)} wpisów zamiast jednego")
    ((uuid, raw),) = payload.items()
    entry = _object(raw, "wpis Rekora")
    verification = _object(entry.get("verification"), "wpis Rekora: verification")
    _object(verification.get("inclusionProof"), "wpis Rekora: inclusionProof")
    if not isinstance(entry.get("body"), str):
        raise MalformedPayload("wpis Rekora bez `body`")
    for field in ("integratedTime", "logIndex"):
        if not isinstance(entry.get(field), int):
            raise MalformedPayload(f"wpis Rekora bez liczbowego `{field}`")
    return {"uuid": uuid, **entry}


class RekorAttestationPublisher(AttestationPublisherPort):
    """Root cyklu najpierw do Rekora, potem — z polem `rekor` — do publishera pliku."""

    def __init__(
        self,
        delegate: AttestationPublisherPort,
        signer: SigstoreKeylessSigner,
        rekor: RekorClient,
    ) -> None:
        self._delegate = delegate
        self._signer = signer
        self._rekor = rekor

    def publish_commitment(self, record: Mapping[str, Any]) -> str:
        cycle = CycleRoot.from_record(record)
        if cycle is None or not cycle.leaves:
            return self._delegate.publish_commitment(record)
        try:
            artifact = root_artifact(cycle.root)
            signed = self._signer.sign(artifact)
            entry = self._rekor.log(hashedrekord(root_artifact_sha256(cycle.root), signed))
        except (PortError, ValueError) as exc:
            logger.warning(
                "Rekor: root cyklu %s bez wpisu w logu przejrzystości — %s", cycle.cycle_id, exc
            )
            return self._delegate.publish_commitment(record)
        logger.info(
            "Rekor: root cyklu %s w logu (logIndex %s)", cycle.cycle_id, entry.get("logIndex")
        )
        return self._delegate.publish_commitment({**record, REKOR_FIELD: entry})

    def publish_reveal(self, record: Mapping[str, Any]) -> str:
        return self._delegate.publish_reveal(record)


def keyless_rekor_publisher(
    delegate: AttestationPublisherPort, environ: Mapping[str, str]
) -> RekorAttestationPublisher:
    """Rekor z podpisem keyless tokenem OIDC z `environ` (env joba GitHub Actions)."""
    session = build_session()
    return RekorAttestationPublisher(
        delegate,
        SigstoreKeylessSigner(session, GitHubActionsIdentityToken(session, environ)),
        RekorClient(session),
    )
