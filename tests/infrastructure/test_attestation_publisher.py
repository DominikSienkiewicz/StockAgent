"""Testy dla adapterów `AttestationPublisherPort` (#16 — commit-reveal):
- `FileAttestationPublisher` — append-only JSONL (osobne pliki commit/reveal).
- `NullAttestationPublisher` — no-op gdy feature wyłączony.

Publikacja NIGDY nie może wywalić cyklu: błąd zapisu jest połykany i zwraca "".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.infrastructure.adapters.attestation_publisher import (
    FileAttestationPublisher,
    NullAttestationPublisher,
)


class TestFileAttestationPublisher:
    def _make(self, tmp_path: Path) -> FileAttestationPublisher:
        return FileAttestationPublisher(
            commitment_path=str(tmp_path / "commitments.jsonl"),
            reveal_path=str(tmp_path / "reveals.jsonl"),
        )

    def test_append_keeps_previous_lines(self, tmp_path: Path) -> None:
        target = tmp_path / "commitments.jsonl"
        publisher = self._make(tmp_path)

        publisher.publish_commitment({"symbol": "NVDA", "commitment": "aaa"})
        publisher.publish_commitment({"symbol": "TSLA", "commitment": "bbb"})

        lines = target.read_text(encoding="utf-8").splitlines()
        # Dwa wpisy = dwie linie; pierwszy wpis NIE zniknął.
        assert len(lines) == 2
        assert json.loads(lines[0])["symbol"] == "NVDA"
        assert json.loads(lines[1])["symbol"] == "TSLA"

    def test_each_line_is_valid_compact_json(self, tmp_path: Path) -> None:
        target = tmp_path / "commitments.jsonl"
        publisher = self._make(tmp_path)

        publisher.publish_commitment({"b": 2, "a": 1, "znak": "ł"})

        raw = target.read_text(encoding="utf-8")
        lines = raw.splitlines()
        # Jeden wpis = jedna linia (brak wewnętrznych newline'ów).
        assert len(lines) == 1
        line = lines[0]
        # Każda linia to poprawny JSON z posortowanymi kluczami; polski znak
        # zachowany dosłownie (ensure_ascii=False), nie zescape'owany do \\uXXXX.
        assert json.loads(line) == {"b": 2, "a": 1, "znak": "ł"}
        assert line == '{"a": 1, "b": 2, "znak": "ł"}'

    def test_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        commitment = tmp_path / "deep" / "nested" / "commitments.jsonl"
        reveal = tmp_path / "deep" / "nested" / "reveals.jsonl"
        publisher = FileAttestationPublisher(
            commitment_path=str(commitment), reveal_path=str(reveal)
        )

        result = publisher.publish_commitment({"x": 1})

        assert commitment.exists()
        assert result == str(commitment.resolve())

    def test_commitment_and_reveal_go_to_different_files(
        self, tmp_path: Path
    ) -> None:
        commitment = tmp_path / "commitments.jsonl"
        reveal = tmp_path / "reveals.jsonl"
        publisher = self._make(tmp_path)

        publisher.publish_commitment({"kind": "commit"})
        publisher.publish_reveal({"kind": "reveal"})

        assert json.loads(commitment.read_text(encoding="utf-8"))["kind"] == "commit"
        assert json.loads(reveal.read_text(encoding="utf-8"))["kind"] == "reveal"
        # Reveal nie trafił do pliku commitmentów i odwrotnie.
        assert "reveal" not in commitment.read_text(encoding="utf-8")
        assert "commit" not in reveal.read_text(encoding="utf-8")

    def test_write_failure_is_swallowed_and_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, mocker
    ) -> None:
        publisher = self._make(tmp_path)
        # Symulujemy błąd I/O — publikacja nie może wywalić cyklu.
        mocker.patch("builtins.open", side_effect=OSError("disk full"))

        with caplog.at_level(logging.WARNING):
            result = publisher.publish_commitment({"x": 1})

        assert result == ""
        assert "disk full" in caplog.text or "publish" in caplog.text.lower()

    def test_mkdir_failure_is_swallowed_and_returns_empty(
        self, tmp_path: Path, mocker
    ) -> None:
        publisher = self._make(tmp_path)
        mocker.patch.object(Path, "mkdir", side_effect=OSError("perm denied"))

        result = publisher.publish_reveal({"x": 1})

        assert result == ""


class TestNullAttestationPublisher:
    def test_commitment_returns_empty_without_writing(
        self, mocker
    ) -> None:
        opener = mocker.patch("builtins.open")
        publisher = NullAttestationPublisher()

        result = publisher.publish_commitment({"x": 1})

        assert result == ""
        opener.assert_not_called()

    def test_reveal_returns_empty_without_writing(self, mocker) -> None:
        opener = mocker.patch("builtins.open")
        publisher = NullAttestationPublisher()

        result = publisher.publish_reveal({"x": 1})

        assert result == ""
        opener.assert_not_called()
