"""WIA-60 — commitmenty cyklu zbierane jako liście jednego drzewa Merkle'a.

Graf publikuje commitment każdej zapisanej predykcji jak dotąd (`publish_commitment`),
ale do ledgera trafia jedna linia na cykl: root, `n_leaves` i liście — także dla
pustego zbioru. Reveal nadpisanej predykcji (WIA-01) idzie od razu, bez zmian.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.application.attestation_cycle import (
    CycleCommitmentCollector,
    attestation_cycle_id,
    seal_attestation_cycle,
)
from src.application.ports import AttestationPublisherPort
from src.domain.attestation import commitment_ledger_record, merkle_root

_AT = datetime(2026, 10, 26, 5, 31, tzinfo=UTC)


def _line(commitment: str) -> dict[str, object]:
    return commitment_ledger_record(
        symbol="NVDA", commitment=commitment, prediction_id="p", committed_at=_AT
    )


def _collector() -> tuple[CycleCommitmentCollector, MagicMock]:
    publisher = MagicMock(spec=AttestationPublisherPort)
    publisher.publish_commitment.return_value = "public/attestation/commitments.jsonl"
    return CycleCommitmentCollector(publisher), publisher


def test_cycle_publishes_one_root_line_instead_of_a_line_per_prediction() -> None:
    collector, publisher = _collector()

    collector.publish_commitment(_line("bb" * 32))
    collector.publish_commitment(_line("aa" * 32))
    publisher.publish_commitment.assert_not_called()

    location = collector.seal("cycle-1", _AT)

    assert location == "public/attestation/commitments.jsonl"
    line = publisher.publish_commitment.call_args.args[0]
    assert line["cycle_id"] == "cycle-1"
    assert line["leaves"] == ["aa" * 32, "bb" * 32]
    assert line["root"] == merkle_root(["aa" * 32, "bb" * 32])
    assert line["n_leaves"] == 2


def test_empty_cycle_still_publishes_its_root() -> None:
    collector, publisher = _collector()

    collector.seal("cycle-1", _AT)

    assert publisher.publish_commitment.call_args.args[0]["n_leaves"] == 0


def test_cycle_is_sealed_once() -> None:
    collector, publisher = _collector()

    collector.seal("cycle-1", _AT)

    assert collector.seal("cycle-1", _AT) == ""
    publisher.publish_commitment.assert_called_once()


def test_commitment_after_the_seal_is_refused() -> None:
    # Hash spoza opublikowanego rootu byłby tylko w bazie — lepiej głośno niż po cichu.
    collector, _ = _collector()
    collector.seal("cycle-1", _AT)

    assert collector.publish_commitment(_line("aa" * 32)) == ""


def test_line_without_a_hash_is_not_a_leaf() -> None:
    collector, publisher = _collector()

    assert collector.publish_commitment({"symbol": "NVDA"}) == ""
    collector.seal("cycle-1", _AT)

    assert publisher.publish_commitment.call_args.args[0]["n_leaves"] == 0


def test_reveals_pass_straight_through() -> None:
    collector, publisher = _collector()
    publisher.publish_reveal.return_value = "public/attestation/reveals.jsonl"

    assert collector.publish_reveal({"commitment": "aa" * 32, "status": "superseded"}) == (
        "public/attestation/reveals.jsonl"
    )


def test_seal_helper_seals_only_a_cycle_collector() -> None:
    collector, publisher = _collector()
    other = MagicMock(spec=AttestationPublisherPort)

    seal_attestation_cycle(None, cycle_id="c", committed_at=_AT)
    seal_attestation_cycle(other, cycle_id="c", committed_at=_AT)
    seal_attestation_cycle(collector, cycle_id="c", committed_at=_AT)

    other.publish_commitment.assert_not_called()
    publisher.publish_commitment.assert_called_once()


def test_failed_root_publish_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    collector, publisher = _collector()
    publisher.publish_commitment.return_value = ""

    seal_attestation_cycle(collector, cycle_id="c", committed_at=_AT)

    assert "root" in caplog.text


def test_cycle_id_names_the_run_its_attempt_and_its_start() -> None:
    started = datetime(2026, 10, 26, 5, 30, 7, tzinfo=UTC)

    assert attestation_cycle_id("123456", 2, started) == "123456.2.20261026T053007Z"
    assert attestation_cycle_id("local", None, started) == "local.1.20261026T053007Z"
