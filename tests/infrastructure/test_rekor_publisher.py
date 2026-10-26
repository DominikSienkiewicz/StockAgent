"""WIA-73: `RekorAttestationPublisher` loguje root cyklu w Sigstore Rekor.

HTTP mockujemy przez `requests.Session.get` (token OIDC joba) i `requests.Session.post`
(Fulcio, Rekor), jak w pozostałych testach adapterów. Klucz efemeryczny i podpis są
prawdziwe (`cryptography`), więc testy sprawdzają też, że podpis weryfikuje się kluczem
z CSR-a wysłanego do Fulcio.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from pytest_mock import MockerFixture

from src.application.errors import MalformedPayload, PortError
from src.application.ports import AttestationPublisherPort
from src.domain.attestation import cycle_root_record
from src.infrastructure.adapters._http import build_session
from src.infrastructure.adapters.rekor_publisher import (
    FULCIO_SIGNING_CERT_URL,
    REKOR_ENTRIES_URL,
    GitHubActionsIdentityToken,
    keyless_rekor_publisher,
    leaf_certificate,
    log_entry,
    token_subject,
)
from tests.sigstore_support import (
    LEAF_CERTIFICATE_PEM,
    OIDC_ENVIRON,
    OIDC_REQUEST_TOKEN,
    OIDC_REQUEST_URL,
    WORKFLOW_SUBJECT,
    fulcio_response,
    identity_token,
    json_response,
    rekor_response,
)

_NOW = datetime(2026, 10, 26, 5, 31, 7, tzinfo=UTC)
ROOT_LINE = cycle_root_record(
    cycle_id="18234.1.20261026T053000Z", committed_at=_NOW, commitments=["aa" * 32, "bb" * 32]
)


def _route_posts(mocker: MockerFixture, routes: dict[str, Callable[[], Mock]]) -> Mock:
    """`requests.Session.post` z odpowiedzią wybraną po URL-u (Fulcio albo Rekor)."""

    def post(url: str, **_kwargs: Any) -> Mock:
        return routes[url]()

    mocked: Mock = mocker.patch("requests.Session.post", side_effect=post)
    return mocked


def _sigstore_ok(mocker: MockerFixture) -> Mock:
    mocker.patch("requests.Session.get", return_value=json_response({"value": identity_token()}))
    return _route_posts(
        mocker,
        {
            FULCIO_SIGNING_CERT_URL: lambda: json_response(fulcio_response()),
            REKOR_ENTRIES_URL: lambda: json_response(rekor_response(), status=201),
        },
    )


def _posted_json(post: Mock, url: str) -> dict[str, Any]:
    calls = [call for call in post.call_args_list if call.args and call.args[0] == url]
    assert calls, f"brak POST-a do {url}"
    body = calls[-1].kwargs["json"]
    assert isinstance(body, dict)
    return body


def test_posts_hashedrekord_with_sha256_of_root(mocker: MockerFixture) -> None:
    post = _sigstore_ok(mocker)
    publisher = keyless_rekor_publisher(Mock(spec=AttestationPublisherPort), OIDC_ENVIRON)

    publisher.publish_commitment(ROOT_LINE)

    entry = _posted_json(post, REKOR_ENTRIES_URL)
    assert (entry["apiVersion"], entry["kind"]) == ("0.0.1", "hashedrekord")
    assert entry["spec"]["data"]["hash"] == {
        "algorithm": "sha256",
        "value": hashlib.sha256(bytes.fromhex(ROOT_LINE["root"])).hexdigest(),
    }


def _fulcio_csr(post: Mock) -> x509.CertificateSigningRequest:
    request = _posted_json(post, FULCIO_SIGNING_CERT_URL)
    pem = base64.b64decode(request["certificateSigningRequest"])
    return x509.load_pem_x509_csr(pem)


def test_root_signature_verifies_with_the_key_certified_by_fulcio(mocker: MockerFixture) -> None:
    post = _sigstore_ok(mocker)
    publisher = keyless_rekor_publisher(Mock(spec=AttestationPublisherPort), OIDC_ENVIRON)

    publisher.publish_commitment(ROOT_LINE)

    key = _fulcio_csr(post).public_key()
    assert isinstance(key, ec.EllipticCurvePublicKey)
    signature = _posted_json(post, REKOR_ENTRIES_URL)["spec"]["signature"]
    key.verify(
        base64.b64decode(signature["content"]),
        bytes.fromhex(ROOT_LINE["root"]),
        ec.ECDSA(hashes.SHA256()),
    )
    # Rekor dostaje liść łańcucha Fulcio, nie certyfikat roota.
    assert base64.b64decode(signature["publicKey"]["content"]).decode() == LEAF_CERTIFICATE_PEM


def test_fulcio_gets_the_oidc_token_and_a_signed_csr_for_its_subject(
    mocker: MockerFixture,
) -> None:
    post = _sigstore_ok(mocker)
    publisher = keyless_rekor_publisher(Mock(spec=AttestationPublisherPort), OIDC_ENVIRON)

    publisher.publish_commitment(ROOT_LINE)

    request = _posted_json(post, FULCIO_SIGNING_CERT_URL)
    csr = _fulcio_csr(post)
    assert request["credentials"] == {"oidcIdentityToken": identity_token()}
    assert csr.is_signature_valid
    assert csr.subject.get_attributes_for_oid(NameOID.EMAIL_ADDRESS)[0].value == WORKFLOW_SUBJECT


def test_oidc_token_is_requested_for_the_sigstore_audience(mocker: MockerFixture) -> None:
    _sigstore_ok(mocker)
    get = mocker.patch(
        "requests.Session.get", return_value=json_response({"value": identity_token()})
    )
    publisher = keyless_rekor_publisher(Mock(spec=AttestationPublisherPort), OIDC_ENVIRON)

    publisher.publish_commitment(ROOT_LINE)

    get.assert_called_once()
    assert get.call_args.args[0] == OIDC_REQUEST_URL
    assert get.call_args.kwargs["params"] == {"audience": "sigstore"}
    assert get.call_args.kwargs["headers"] == {"Authorization": f"bearer {OIDC_REQUEST_TOKEN}"}


def test_ledger_line_carries_the_rekor_entry(mocker: MockerFixture) -> None:
    _sigstore_ok(mocker)
    delegate = Mock(spec=AttestationPublisherPort)
    delegate.publish_commitment.return_value = "/ledger/commitments.jsonl"
    publisher = keyless_rekor_publisher(delegate, OIDC_ENVIRON)

    published = publisher.publish_commitment(ROOT_LINE)

    ((uuid, entry),) = rekor_response().items()
    assert published == "/ledger/commitments.jsonl"
    delegate.publish_commitment.assert_called_once_with(
        {**ROOT_LINE, "rekor": {"uuid": uuid, **entry}}
    )


def _rekor_status(status: int) -> Callable[[], Mock]:
    return lambda: json_response({"code": status, "message": "error"}, status=status)


def _network_error() -> Mock:
    raise requests.ConnectionError("connection reset")


@pytest.mark.parametrize(
    ("environ", "rekor"),
    [
        pytest.param({}, lambda: json_response(rekor_response(), status=201), id="no-oidc"),
        pytest.param(OIDC_ENVIRON, _rekor_status(409), id="rekor-conflict"),
        pytest.param(OIDC_ENVIRON, _rekor_status(503), id="rekor-down"),
        pytest.param(OIDC_ENVIRON, _network_error, id="network"),
        pytest.param(
            OIDC_ENVIRON, lambda: json_response({"a": {}, "b": {}}, status=201), id="malformed"
        ),
    ],
)
def test_failed_rekor_logging_still_publishes_the_line_and_warns(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
    environ: dict[str, str],
    rekor: Callable[[], Mock],
) -> None:
    mocker.patch("requests.Session.get", return_value=json_response({"value": identity_token()}))
    _route_posts(
        mocker,
        {
            FULCIO_SIGNING_CERT_URL: lambda: json_response(fulcio_response()),
            REKOR_ENTRIES_URL: rekor,
        },
    )
    delegate = Mock(spec=AttestationPublisherPort)
    publisher = keyless_rekor_publisher(delegate, environ)

    with caplog.at_level(logging.WARNING):
        publisher.publish_commitment(ROOT_LINE)

    delegate.publish_commitment.assert_called_once_with(ROOT_LINE)
    assert any("Rekor" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            cycle_root_record(cycle_id="empty", committed_at=_NOW, commitments=[]),
            id="empty-cycle",
        ),
        pytest.param({"symbol": "AAPL", "commitment": "aa" * 32}, id="single-line"),
    ],
)
def test_lines_without_a_root_to_anchor_skip_rekor(
    mocker: MockerFixture, record: dict[str, Any]
) -> None:
    post = _sigstore_ok(mocker)
    delegate = Mock(spec=AttestationPublisherPort)
    publisher = keyless_rekor_publisher(delegate, OIDC_ENVIRON)

    publisher.publish_commitment(record)

    post.assert_not_called()
    delegate.publish_commitment.assert_called_once_with(record)


def test_reveals_go_straight_to_the_ledger(mocker: MockerFixture) -> None:
    post = _sigstore_ok(mocker)
    delegate = Mock(spec=AttestationPublisherPort)
    reveal = {"commitment": "aa" * 32, "salt": "00" * 16}

    keyless_rekor_publisher(delegate, OIDC_ENVIRON).publish_reveal(reveal)

    post.assert_not_called()
    delegate.publish_reveal.assert_called_once_with(reveal)


def test_oidc_failure_message_does_not_carry_the_request_token(mocker: MockerFixture) -> None:
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            f"headers {{'Authorization': 'bearer {OIDC_REQUEST_TOKEN}'}} unreachable"
        ),
    )
    token = GitHubActionsIdentityToken(build_session(), OIDC_ENVIRON)

    with pytest.raises(PortError) as raised:
        token()

    assert OIDC_REQUEST_TOKEN not in str(raised.value)


def test_fulcio_certificate_with_a_detached_sct_is_read_too() -> None:
    payload = {
        "signedCertificateDetachedSct": {
            "chain": {"certificates": [LEAF_CERTIFICATE_PEM]},
            "signedCertificateTimestamp": "c2N0",
        }
    }

    assert leaf_certificate(payload) == LEAF_CERTIFICATE_PEM


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"signedCertificateEmbeddedSct": {"chain": {"certificates": []}}},
        {"signedCertificateEmbeddedSct": {"chain": {"certificates": ["not a certificate"]}}},
    ],
)
def test_fulcio_response_without_a_certificate_is_malformed(payload: dict[str, Any]) -> None:
    with pytest.raises(MalformedPayload):
        leaf_certificate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"a": {"body": "x"}, "b": {"body": "y"}},
        {"uuid": {"integratedTime": 1, "logIndex": 2, "verification": {"inclusionProof": {}}}},
        {
            "uuid": {
                "body": "x",
                "integratedTime": "1",
                "logIndex": 2,
                "verification": {"inclusionProof": {}},
            }
        },
        {"uuid": {"body": "x", "integratedTime": 1, "logIndex": 2, "verification": {}}},
    ],
)
def test_rekor_response_without_one_complete_entry_is_malformed(payload: dict[str, Any]) -> None:
    with pytest.raises(MalformedPayload):
        log_entry(payload)


@pytest.mark.parametrize("token", ["not-a-jwt", "a.!!!.c", "a.e30.c"])
def test_token_without_a_readable_subject_is_malformed(token: str) -> None:
    with pytest.raises(MalformedPayload):
        token_subject(token)
