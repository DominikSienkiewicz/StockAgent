"""WIA-15 — `FileAttestationLedger`: odczyt trwałego ledgera z checkoutu `attestation-log`.

Na starcie runu plik revealów to dokładnie to, co przeżyło push poprzedniego runu,
więc jego hashe rozstrzygają, którym wierszom wolno ostemplować `revealed_at`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.application.errors import ProviderUnavailable
from src.domain.attestation import cycle_root_record
from src.infrastructure.adapters.attestation_ledger import FileAttestationLedger


def _ledger(tmp_path: Path) -> FileAttestationLedger:
    return FileAttestationLedger(
        commitment_path=str(tmp_path / "commitments.jsonl"),
        reveal_path=str(tmp_path / "reveals.jsonl"),
    )


def _write(path: Path, *lines: object) -> None:
    path.write_text(
        "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines),
        encoding="utf-8",
    )


def test_revealed_commitments_are_the_hashes_of_every_reveal_line(tmp_path: Path) -> None:
    _write(
        tmp_path / "reveals.jsonl",
        {"commitment": "AA" * 32, "salt": "s"},
        {"commitment": "bb" * 32, "salt": "s", "status": "superseded"},
    )

    assert _ledger(tmp_path).revealed_commitments() == frozenset({"aa" * 32, "bb" * 32})


def test_missing_ledger_file_means_nothing_revealed(tmp_path: Path) -> None:
    assert _ledger(tmp_path).revealed_commitments() == frozenset()


def test_unreadable_lines_are_skipped(tmp_path: Path) -> None:
    _write(
        tmp_path / "reveals.jsonl",
        "{not json",
        "[1, 2]",
        {"salt": "no hash"},
        "",
        {"commitment": "cc" * 32},
    )

    assert _ledger(tmp_path).revealed_commitments() == frozenset({"cc" * 32})


def test_io_error_is_a_port_error(tmp_path: Path) -> None:
    (tmp_path / "reveals.jsonl").mkdir()

    with pytest.raises(ProviderUnavailable):
        _ledger(tmp_path).revealed_commitments()


def test_commitment_ledger_reads_single_lines_and_cycle_roots(tmp_path: Path) -> None:
    root_line = cycle_root_record(
        cycle_id="c1",
        committed_at=datetime(2026, 10, 26, tzinfo=UTC),
        commitments=["aa" * 32, "bb" * 32],
    )
    _write(
        tmp_path / "commitments.jsonl",
        {"symbol": "OLD", "commitment": "dd" * 32},
        "{broken",
        root_line,
    )

    ledger = _ledger(tmp_path).commitment_ledger()

    assert ledger.published == frozenset({"dd" * 32, "aa" * 32, "bb" * 32})
    inclusion = ledger.inclusion_of("bb" * 32)
    assert inclusion is not None and inclusion.verifies("bb" * 32)


def test_missing_commitment_file_is_an_empty_ledger(tmp_path: Path) -> None:
    assert _ledger(tmp_path).commitment_ledger().published == frozenset()
