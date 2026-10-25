"""WIA-59: plik głów paper ledgera — append-only JSONL obok atestacji."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.infrastructure.adapters.ledger_head_publisher import FileLedgerHeadPublisher

RECORD = {
    "kind": "paper_ledger_head",
    "ledger_version": 1,
    "published_on": "2026-10-19",
    "seq": 7,
    "head": "a" * 64,
}


def test_publish_appends_one_compact_line(tmp_path: Path) -> None:
    path = tmp_path / "attestation" / "paper_ledger_heads.jsonl"
    publisher = FileLedgerHeadPublisher(str(path))

    first = publisher.publish_head(RECORD)
    publisher.publish_head({**RECORD, "published_on": "2026-10-20", "seq": 9})

    assert first == str(path.resolve())
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [7, 9]
    assert lines[0] == json.dumps(RECORD, sort_keys=True, ensure_ascii=False)


def test_last_published_day_is_the_day_of_the_last_line(tmp_path: Path) -> None:
    path = tmp_path / "paper_ledger_heads.jsonl"
    publisher = FileLedgerHeadPublisher(str(path))
    publisher.publish_head(RECORD)
    publisher.publish_head({**RECORD, "published_on": "2026-10-20"})

    assert publisher.last_published_on() == date(2026, 10, 20)


def test_missing_or_unreadable_file_has_no_published_day(tmp_path: Path) -> None:
    missing = FileLedgerHeadPublisher(str(tmp_path / "none.jsonl"))
    garbled = tmp_path / "garbled.jsonl"
    garbled.write_text("not json\n", encoding="utf-8")

    assert missing.last_published_on() is None
    assert FileLedgerHeadPublisher(str(garbled)).last_published_on() is None


def test_write_error_returns_empty(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    publisher = FileLedgerHeadPublisher(str(blocker / "heads.jsonl"))

    assert publisher.publish_head(RECORD) == ""
