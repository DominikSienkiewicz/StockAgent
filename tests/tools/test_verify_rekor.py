"""WIA-73: `verify_rekor` sprawdza wpis Rekora w linii rootu cyklu offline.

Wpisy buduje `tests.sigstore_support.rekor_entry_for`: prawdziwy podpis rootu kluczem
P-256, certyfikat w kształcie liścia Fulcio (SAN z URI workflowu, issuer w rozszerzeniu
Sigstore) i inclusion proof ciała wpisu w drzewie RFC 9162 policzonym funkcjami domeny.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.domain.attestation import cycle_root_record
from src.tools import verify_rekor
from src.tools.verify_rekor import WorkflowIdentity, rekor_problems
from tests.sigstore_support import WORKFLOW_URI, rekor_entry_for

ROOT = Path(__file__).resolve().parents[2]
_NOW = datetime(2026, 10, 26, 5, 31, 7, tzinfo=UTC)
_LINE = cycle_root_record(
    cycle_id="987.1.20261026T053000Z", committed_at=_NOW, commitments=["aa" * 32, "bb" * 32]
)
_ROOT = str(_LINE["root"])
_OTHER_ROOT = "cd" * 32


def _line(**entry: Any) -> dict[str, Any]:
    integrated_at = entry.pop("integrated_at", _NOW + timedelta(seconds=3))
    root = entry.pop("root", _ROOT)
    return {**_LINE, "rekor": rekor_entry_for(root, integrated_at=integrated_at, **entry)}


def _problems(line: dict[str, Any]) -> str:
    return " | ".join(rekor_problems(line, _ROOT))


@pytest.mark.parametrize("issuer_extension", ["v1", "v2"])
def test_an_entry_of_the_fast_loop_confirms_its_root(issuer_extension: str) -> None:
    assert rekor_problems(_line(issuer_extension=issuer_extension), _ROOT) == []


def test_an_entry_of_another_root_does_not_confirm_this_one() -> None:
    assert "does not log this cycle root" in _problems(_line(root=_OTHER_ROOT))


def test_a_signature_over_another_root_fails() -> None:
    assert "signature" in _problems(_line(signed_root=_OTHER_ROOT))


def test_a_proof_that_does_not_reach_the_tree_root_fails() -> None:
    line = _line()
    line["rekor"]["verification"]["inclusionProof"]["hashes"][0] = "00" * 32

    assert "inclusion proof does not reach" in _problems(line)


@pytest.mark.parametrize(
    "field", ["hashes", "logIndex", "treeSize", "rootHash"]
)
def test_an_incomplete_proof_fails(field: str) -> None:
    line = _line()
    del line["rekor"]["verification"]["inclusionProof"][field]

    assert "incomplete" in _problems(line)


@pytest.mark.parametrize(
    "checkpoint",
    [None, "rekor.sigstore.dev - 1\n7\n", "rekor.sigstore.dev - 1\n8\nAAAA\n\n", "x\n7\n!!\n\n"],
)
def test_a_checkpoint_that_names_another_tree_fails(checkpoint: str | None) -> None:
    line = _line()
    line["rekor"]["verification"]["inclusionProof"]["checkpoint"] = checkpoint

    assert "checkpoint" in _problems(line)


@pytest.mark.parametrize(
    "integrated_at",
    [_NOW + verify_rekor.MAX_LOG_DELAY + timedelta(seconds=1), _NOW - timedelta(minutes=6)],
)
def test_a_root_logged_outside_the_window_of_its_line_fails(integrated_at: datetime) -> None:
    assert "outside the window" in _problems(_line(integrated_at=integrated_at))


def test_a_root_logged_after_its_certificate_expired_fails() -> None:
    line = _line(certified_at=_NOW - timedelta(minutes=11))

    assert "validity of its certificate" in _problems(line)


@pytest.mark.parametrize("committed_at", [None, "yesterday", "2026-10-26T05:31:07"])
def test_a_line_without_a_usable_committed_at_fails(committed_at: str | None) -> None:
    line = {**_line(), "committed_at": committed_at}

    assert "committed_at" in _problems(line)


@pytest.mark.parametrize(
    "uri",
    [
        "https://github.com/DominikSienkiewicz/StockAgent/.github/workflows/other.yml@refs/heads/main",
        "https://github.com/someone/StockAgent/.github/workflows/fast_loop_12h.yml@refs/heads/main",
    ],
)
def test_an_entry_signed_by_another_workflow_fails(uri: str) -> None:
    assert "instead of the workflow" in _problems(_line(uri=uri))


def test_an_entry_issued_for_another_oidc_provider_fails() -> None:
    line = _line(issuer="https://accounts.google.com")

    assert "was not issued for" in _problems(line)


def test_the_identity_names_the_workflow_file_and_any_ref() -> None:
    identity = WorkflowIdentity()

    assert WORKFLOW_URI.startswith(identity.uri_prefix)
    assert identity.uri_prefix.endswith("/.github/workflows/fast_loop_12h.yml@")


def _with_body(line: dict[str, Any], body: object) -> dict[str, Any]:
    raw = json.dumps(body).encode() if not isinstance(body, bytes) else body
    line["rekor"]["body"] = base64.b64encode(raw).decode()
    return line


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda line: {**line, "rekor": "x"}, id="not-an-object"),
        pytest.param(lambda line: {**line, "rekor": {**line["rekor"], "body": "%%"}}, id="b64"),
        pytest.param(lambda line: _with_body(line, b"not json"), id="json"),
        pytest.param(lambda line: _with_body(line, {"spec": {}}), id="no-certificate"),
        pytest.param(
            lambda line: _with_body(
                line, {"spec": {"signature": {"publicKey": {"content": "bm90IGEgY2VydA=="}}}}
            ),
            id="bad-certificate",
        ),
        pytest.param(
            lambda line: {**line, "rekor": {**line["rekor"], "integratedTime": "1"}}, id="time"
        ),
        pytest.param(
            lambda line: {**line, "rekor": {**line["rekor"], "verification": {}}}, id="no-proof"
        ),
    ],
)
def test_an_unreadable_entry_is_one_problem_not_a_crash(mutate: Any) -> None:
    problems = rekor_problems(mutate(_line()), _ROOT)

    assert len(problems) == 1 and "unreadable Rekor entry" in problems[0], problems


def test_a_line_whose_root_is_not_32_bytes_cannot_be_confirmed() -> None:
    assert "unreadable" in " ".join(rekor_problems(_line(), "ab" * 31))


def test_the_ledger_field_is_the_one_the_publisher_writes() -> None:
    from src.infrastructure.adapters.rekor_publisher import REKOR_FIELD

    assert verify_rekor.REKOR_FIELD == REKOR_FIELD


def test_the_default_workflow_is_the_one_that_runs_the_agent() -> None:
    workflow = ROOT / verify_rekor.DEFAULT_REKOR_WORKFLOW

    assert "main_agent.py" in workflow.read_text(encoding="utf-8")
