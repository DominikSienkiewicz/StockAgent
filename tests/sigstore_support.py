"""WIA-73: atrapy Sigstore dla testów Rekora — token OIDC joba GitHub Actions, odpowiedzi
HTTP w kształcie API Fulcio i Rekora oraz spójny kryptograficznie wpis Rekora rootu cyklu.

Token to JWT bez ważnego podpisu: adapter czyta z niego tylko claim `sub` (temat CSR-a),
a podpis tokenu sprawdza Fulcio, którego tu nie ma. Certyfikat wpisu jest samopodpisany
(nie od Fulcio), bo weryfikator offline nie sprawdza łańcucha do roota Fulcio.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from src.domain.attestation import inclusion_proof, merkle_root

WORKFLOW_SUBJECT = "repo:DominikSienkiewicz/StockAgent:ref:refs/heads/main"
OIDC_REQUEST_URL = "https://pipelines.actions.example/idtoken?api-version=2.0"
OIDC_REQUEST_TOKEN = "oidc-request-bearer-0123456789"
OIDC_ENVIRON = {
    "ACTIONS_ID_TOKEN_REQUEST_URL": OIDC_REQUEST_URL,
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": OIDC_REQUEST_TOKEN,
}
LEAF_CERTIFICATE_PEM = (
    "-----BEGIN CERTIFICATE-----\nMIIBleafcertificate\n-----END CERTIFICATE-----\n"
)
ROOT_CERTIFICATE_PEM = (
    "-----BEGIN CERTIFICATE-----\nMIIBrootcertificate\n-----END CERTIFICATE-----\n"
)


def _b64url(data: dict[str, Any]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def identity_token(sub: str = WORKFLOW_SUBJECT) -> str:
    """JWT w kształcie tokenu OIDC GitHub Actions (nagłówek.claimy.podpis)."""
    header = _b64url({"alg": "RS256", "typ": "JWT"})
    claims = _b64url(
        {"iss": "https://token.actions.githubusercontent.com", "aud": "sigstore", "sub": sub}
    )
    return f"{header}.{claims}.c2lnbmF0dXJl"


def json_response(payload: object, status: int = 200) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status} Client Error", response=response
        )
    else:
        response.raise_for_status = Mock()
    return response


def fulcio_response(*chain: str) -> dict[str, Any]:
    """Odpowiedź `POST /api/v2/signingCert` z SCT wbudowanym w certyfikat."""
    certificates = list(chain) or [LEAF_CERTIFICATE_PEM, ROOT_CERTIFICATE_PEM]
    return {"signedCertificateEmbeddedSct": {"chain": {"certificates": certificates}}}


def rekor_response(
    uuid: str = "24296fb24b8ad77a" + "ab" * 32,
    *,
    body: str = "eyJraW5kIjoiaGFzaGVkcmVrb3JkIn0=",
    integrated_time: int = 1_793_000_000,
    log_index: int = 412_345_678,
) -> dict[str, Any]:
    """Odpowiedź `POST /api/v1/log/entries` (201): mapa uuid → wpis."""
    return {
        uuid: {
            "body": body,
            "integratedTime": integrated_time,
            "logID": "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d",
            "logIndex": log_index,
            "verification": {
                "inclusionProof": {
                    "checkpoint": "rekor.sigstore.dev - 1193050959916656506\n2\nAAAA\n\n",
                    "hashes": ["cd" * 32],
                    "logIndex": 0,
                    "rootHash": "ef" * 32,
                    "treeSize": 2,
                },
                "signedEntryTimestamp": "MEUCIQ==",
            },
        }
    }


# --- Spójny kryptograficznie wpis Rekora dla rootu cyklu (verify-attestation) ----------

GITHUB_ACTIONS_ISSUER = "https://token.actions.githubusercontent.com"
WORKFLOW_URI = (
    "https://github.com/DominikSienkiewicz/StockAgent/.github/workflows/fast_loop_12h.yml"
    "@refs/heads/main"
)
_ISSUER_V1_OID = x509.ObjectIdentifier("1.3.6.1.4.1.57264.1.1")
_ISSUER_V2_OID = x509.ObjectIdentifier("1.3.6.1.4.1.57264.1.8")
_CERT_LIFETIME = timedelta(minutes=10)
_OTHER_LEAVES = 6
_LEAF_POSITION = 4


def workflow_certificate(
    key: ec.EllipticCurvePrivateKey,
    *,
    issued_at: datetime,
    uri: str = WORKFLOW_URI,
    issuer: str = GITHUB_ACTIONS_ISSUER,
    issuer_extension: str = "v2",
) -> str:
    """Certyfikat w kształcie liścia Fulcio: pusty temat, SAN z URI workflowu i issuer
    OIDC w rozszerzeniu Sigstore (v1 = surowy tekst, v2 = DER UTF8String)."""
    if issuer_extension == "v1":
        extension = x509.UnrecognizedExtension(_ISSUER_V1_OID, issuer.encode("ascii"))
    else:
        raw = issuer.encode("ascii")
        extension = x509.UnrecognizedExtension(_ISSUER_V2_OID, bytes([0x0C, len(raw)]) + raw)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "sigstore.dev")]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(issued_at)
        .not_valid_after(issued_at + _CERT_LIFETIME)
        .add_extension(
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(uri)]), critical=True
        )
        .add_extension(extension, critical=False)
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _random_leaf(index: int) -> str:
    return hashlib.sha256(f"other-entry-{index}".encode()).hexdigest()


def rekor_entry_for(
    root: str,
    *,
    integrated_at: datetime,
    uri: str = WORKFLOW_URI,
    issuer: str = GITHUB_ACTIONS_ISSUER,
    issuer_extension: str = "v2",
    certified_at: datetime | None = None,
    signed_root: str | None = None,
) -> dict[str, Any]:
    """Pole `rekor` linii rootu cyklu: wpis `hashedrekord` rootu z prawdziwym podpisem,
    certyfikatem workflowu i inclusion proof w drzewie RFC 9162 kilku wpisów."""
    key = ec.generate_private_key(ec.SECP256R1())
    pem = workflow_certificate(
        key,
        issued_at=certified_at or integrated_at - timedelta(seconds=5),
        uri=uri,
        issuer=issuer,
        issuer_extension=issuer_extension,
    )
    signature = key.sign(bytes.fromhex(signed_root or root), ec.ECDSA(hashes.SHA256()))
    body = json.dumps(
        {
            "apiVersion": "0.0.1",
            "kind": "hashedrekord",
            "spec": {
                "data": {
                    "hash": {
                        "algorithm": "sha256",
                        "value": hashlib.sha256(bytes.fromhex(root)).hexdigest(),
                    }
                },
                "signature": {
                    "content": base64.b64encode(signature).decode("ascii"),
                    "publicKey": {"content": base64.b64encode(pem.encode("ascii")).decode()},
                },
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    leaves = [_random_leaf(index) for index in range(_OTHER_LEAVES)]
    leaves.insert(_LEAF_POSITION, body.hex())
    tree_root = merkle_root(leaves)
    checkpoint = (
        f"rekor.sigstore.dev - 1193050959916656506\n{len(leaves)}\n"
        f"{base64.b64encode(bytes.fromhex(tree_root)).decode('ascii')}\n\n"
        "— rekor.sigstore.dev wNI9ajBFAiEA\n"
    )
    return {
        "uuid": "24296fb24b8ad77a" + hashlib.sha256(b"\x00" + body).hexdigest(),
        "body": base64.b64encode(body).decode("ascii"),
        "integratedTime": int(integrated_at.timestamp()),
        "logID": "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d",
        "logIndex": 412_345_678,
        "verification": {
            "inclusionProof": {
                "checkpoint": checkpoint,
                "hashes": list(inclusion_proof(leaves, _LEAF_POSITION)),
                "logIndex": _LEAF_POSITION,
                "rootHash": tree_root,
                "treeSize": len(leaves),
            },
            "signedEntryTimestamp": "MEUCIQDsigned",
        },
    }
