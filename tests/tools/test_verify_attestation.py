"""WIA-26: `verify_attestation` sprawdza ledger commitmentów i revealów offline.

Linie ledgera powstają tu tymi samymi funkcjami domeny, którymi pisze je agent
(`prediction_commitment_payload`, `canonicalize`, `commit_canonical`,
`commitment_ledger_record`), więc test jest oracle'em dla całego łańcucha.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.domain import attestation
from src.tools import verify_attestation as tool

ROOT = Path(__file__).resolve().parents[2]
_SALT = "0123456789abcdef0123456789abcdef"
_NOW = datetime(2026, 10, 26, 5, 31, 7, tzinfo=UTC)


def _commitment_and_reveal(
    symbol: str = "AAPL",
    *,
    trend: str = "BULLISH",
    target: str = "101.5",
    price: str = "100",
    salt: str = _SALT,
    prediction_id: str = "pred-1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Linia commitmentu i reveal rozliczonej predykcji, jak w prawdziwym ledgerze."""
    payload = attestation.prediction_commitment_payload(
        symbol=symbol,
        predicted_trend=trend,
        predicted_target_price=Decimal(target),
        price_at_prediction=Decimal(price),
        confidence_score=0.8,
    )
    plaintext = attestation.canonicalize(payload, schema_version=attestation.SCHEMA_VERSION)
    commitment = attestation.commit_canonical(plaintext, salt)
    line = attestation.commitment_ledger_record(
        symbol=symbol, commitment=commitment, prediction_id=prediction_id, committed_at=_NOW
    )
    reveal = {
        "symbol": symbol,
        "commitment": commitment,
        "salt": salt,
        "predicted_trend": trend,
        # PostgREST oddaje NUMERIC jako liczbę, nie tekst.
        "predicted_target_price": float(target),
        "price_at_prediction": float(price),
        "actual_price_after_12h": 102.0,
        "is_trend_correct": True,
        "payload": plaintext,
    }
    return line, reveal


def _legacy_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """Commitment sprzed WIA-25 (tylko symbol i hash) z revealem v1 bez `payload`."""
    v1_payload = {
        "symbol": "TSLA",
        "predicted_trend": "BEARISH",
        "predicted_target_price": "90.5",
        "price_at_prediction": "100.0",
    }
    commitment = attestation.commit(v1_payload, _SALT)
    reveal = {
        "symbol": "TSLA",
        "commitment": commitment,
        "salt": _SALT,
        "predicted_trend": "BEARISH",
        "predicted_target_price": 90.5,
        "price_at_prediction": 100.0,
    }
    return {"symbol": "TSLA", "commitment": commitment}, reveal


def test_a_genuine_pair_verifies() -> None:
    commitment, reveal = _commitment_and_reveal()

    report = tool.verify_ledger([commitment], [reveal])

    assert report.ok
    assert [check.status for check in report.checks] == [tool.STATUS_VALID]
    assert (report.commitments, report.pending) == (1, 0)


def test_detects_tampered_reveal() -> None:
    commitment, reveal = _commitment_and_reveal()
    reveal["payload"] = str(reveal["payload"]).replace("101.5000", "150.0000")

    report = tool.verify_ledger([commitment], [reveal])

    assert not report.ok
    assert [check.status for check in report.checks] == [tool.STATUS_TAMPERED]


def test_detects_a_changed_salt() -> None:
    commitment, reveal = _commitment_and_reveal()
    reveal["salt"] = "f" * 32

    report = tool.verify_ledger([commitment], [reveal])

    assert [check.status for check in report.checks] == [tool.STATUS_TAMPERED]


def test_detects_a_reveal_column_that_contradicts_the_committed_payload() -> None:
    # Hash się zgadza, ale wiersz revealu mówi co innego niż zahashowany plaintext:
    # kolumna zmieniona po fakcie.
    commitment, reveal = _commitment_and_reveal(trend="BULLISH")
    reveal["predicted_trend"] = "BEARISH"

    report = tool.verify_ledger([commitment], [reveal])

    (check,) = report.checks
    assert check.status == tool.STATUS_TAMPERED
    assert "predicted_trend" in check.detail


def test_numeric_columns_are_compared_by_value_not_by_text() -> None:
    commitment, reveal = _commitment_and_reveal(target="101.5", price="100")
    reveal["predicted_target_price"] = 101.5
    reveal["price_at_prediction"] = 100

    assert tool.verify_ledger([commitment], [reveal]).ok


def test_detects_a_price_column_that_contradicts_the_committed_payload() -> None:
    commitment, reveal = _commitment_and_reveal()
    reveal["price_at_prediction"] = 99.0

    (check,) = tool.verify_ledger([commitment], [reveal]).checks

    assert check.status == tool.STATUS_TAMPERED
    assert "price_at_prediction" in check.detail


def test_a_reveal_without_a_published_commitment_fails() -> None:
    _, reveal = _commitment_and_reveal()

    report = tool.verify_ledger([], [reveal])

    assert not report.ok
    assert [check.status for check in report.checks] == [tool.STATUS_UNPUBLISHED]


def test_unrevealed_commitments_are_counted_but_do_not_fail() -> None:
    first, reveal = _commitment_and_reveal("AAPL")
    second, _ = _commitment_and_reveal("MSFT", salt="a" * 32, prediction_id="pred-2")

    report = tool.verify_ledger([first, second], [reveal])

    assert report.ok
    assert (report.commitments, report.pending) == (2, 1)


def test_a_rerun_that_published_the_same_hash_twice_counts_once() -> None:
    commitment, reveal = _commitment_and_reveal()

    report = tool.verify_ledger([commitment, dict(commitment)], [reveal, dict(reveal)])

    assert report.ok
    assert (report.commitments, len(report.checks)) == (1, 1)


def test_the_hash_comparison_ignores_letter_case() -> None:
    commitment, reveal = _commitment_and_reveal()
    commitment["commitment"] = str(commitment["commitment"]).upper()

    assert tool.verify_ledger([commitment], [reveal]).ok


def test_a_superseded_reveal_verifies_from_its_payload_alone() -> None:
    commitment, reveal = _commitment_and_reveal()
    superseded = {
        "symbol": reveal["symbol"],
        "commitment": reveal["commitment"],
        "salt": reveal["salt"],
        "payload": reveal["payload"],
        "status": attestation.REVEAL_STATUS_SUPERSEDED,
    }

    assert tool.verify_ledger([commitment], [superseded]).ok


def test_lines_written_before_wia_25_still_verify() -> None:
    commitment, reveal = _legacy_pair()

    report = tool.verify_ledger([commitment], [reveal])

    assert report.ok, report.checks
    assert "legacy" in report.checks[0].detail


def test_a_legacy_reveal_that_does_not_reproduce_the_hash_is_unverifiable() -> None:
    commitment, reveal = _legacy_pair()
    reveal["predicted_trend"] = "BULLISH"

    report = tool.verify_ledger([commitment], [reveal])

    assert not report.ok
    assert [check.status for check in report.checks] == [tool.STATUS_UNVERIFIABLE]


def test_parse_jsonl_skips_blank_lines_and_reports_broken_ones() -> None:
    text = '{"a": 1}\n\nnot json\n[1, 2]\n{"b": 2}\n'

    records, malformed = tool.parse_jsonl(text, "commitments.jsonl")

    assert records == [{"a": 1}, {"b": 2}]
    assert malformed == ["commitments.jsonl:3", "commitments.jsonl:4"]


def test_a_broken_line_fails_the_verdict() -> None:
    commitment, reveal = _commitment_and_reveal()

    report = tool.verify_ledger([commitment], [reveal], malformed=["reveals.jsonl:7"])

    assert not report.ok


def test_a_commitment_line_without_a_hash_fails_the_verdict() -> None:
    commitment, reveal = _commitment_and_reveal()

    report = tool.verify_ledger([commitment, {"symbol": "AAPL"}], [reveal])

    assert not report.ok


def test_the_report_ends_with_one_verdict_line() -> None:
    commitment, reveal = _commitment_and_reveal()
    good = tool.format_report(tool.verify_ledger([commitment], [reveal]))
    reveal["salt"] = "f" * 32
    bad = tool.format_report(tool.verify_ledger([commitment], [reveal]))

    assert good.splitlines()[-1].startswith("OK")
    assert bad.splitlines()[-1].startswith("FAIL")
    assert "AAPL" in bad


def test_default_paths_are_the_ones_of_config_toml() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))

    assert config["attestation_commitments_path"] == tool.DEFAULT_COMMITMENTS_PATH
    assert config["attestation_reveals_path"] == tool.DEFAULT_REVEALS_PATH


def test_a_ledger_written_by_the_file_publisher_verifies(tmp_path: Path) -> None:
    from src.infrastructure.adapters.attestation_publisher import FileAttestationPublisher

    commitment, reveal = _commitment_and_reveal()
    publisher = FileAttestationPublisher(
        commitment_path=str(tmp_path / "commitments.jsonl"),
        reveal_path=str(tmp_path / "reveals.jsonl"),
    )
    publisher.publish_commitment(commitment)
    publisher.publish_reveal(reveal)
    commitments, bad_commitments = tool.parse_jsonl(
        (tmp_path / "commitments.jsonl").read_text(encoding="utf-8"), "commitments.jsonl"
    )
    reveals, bad_reveals = tool.parse_jsonl(
        (tmp_path / "reveals.jsonl").read_text(encoding="utf-8"), "reveals.jsonl"
    )

    report = tool.verify_ledger(commitments, reveals, malformed=[*bad_commitments, *bad_reveals])

    assert report.ok
    assert json.loads((tmp_path / "commitments.jsonl").read_text(encoding="utf-8"))[
        "prediction_id"
    ] == "pred-1"


@pytest.mark.parametrize("value", [None, 5, ["x"]])
def test_non_text_hash_or_salt_never_crashes_the_verifier(value: Any) -> None:
    commitment, reveal = _commitment_and_reveal()
    reveal["salt"] = value

    report = tool.verify_ledger([commitment], [reveal])

    assert not report.ok
