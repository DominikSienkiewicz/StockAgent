"""WIA-73: atrapy Sigstore dla testów Rekora — token OIDC joba GitHub Actions i odpowiedzi
HTTP w kształcie API Fulcio i Rekora.

Token to JWT bez ważnego podpisu: adapter czyta z niego tylko claim `sub` (temat CSR-a),
a podpis tokenu sprawdza Fulcio, którego tu nie ma.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import Mock

import requests

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
