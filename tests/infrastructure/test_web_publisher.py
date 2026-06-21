"""Testy dla adapterów `WebPublisherPort`:
- `FileWebPublisher` — zapis raportu HTML na dysk (artefakt GitHub Pages / commit).
- `NullWebPublisher` — no-op gdy feature wyłączony (mirror `NullNotifier`).

Publikacja NIGDY nie może wywalić cyklu: błąd zapisu jest połykany i zwraca "".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.infrastructure.adapters.web_publisher import (
    FileWebPublisher,
    NullWebPublisher,
)


class TestFileWebPublisher:
    def test_writes_html_to_output_path(self, tmp_path: Path) -> None:
        target = tmp_path / "digest.html"
        publisher = FileWebPublisher(output_path=str(target))

        result = publisher.publish("<html><body>Hi</body></html>")

        assert target.read_text(encoding="utf-8") == "<html><body>Hi</body></html>"
        # Zwracamy użyteczny artefakt — absolutną ścieżkę.
        assert result == str(target.resolve())

    def test_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir" / "index.html"
        publisher = FileWebPublisher(output_path=str(target))

        result = publisher.publish("<p>ok</p>")

        assert target.exists()
        assert target.read_text(encoding="utf-8") == "<p>ok</p>"
        assert result == str(target.resolve())

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "digest.html"
        target.write_text("OLD", encoding="utf-8")
        publisher = FileWebPublisher(output_path=str(target))

        publisher.publish("NEW")

        assert target.read_text(encoding="utf-8") == "NEW"

    def test_write_failure_is_swallowed_and_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, mocker
    ) -> None:
        target = tmp_path / "digest.html"
        publisher = FileWebPublisher(output_path=str(target))
        # Symulujemy błąd I/O — publikacja nie może wywalić cyklu.
        mocker.patch.object(
            Path, "write_text", side_effect=OSError("disk full")
        )

        with caplog.at_level(logging.WARNING):
            result = publisher.publish("<p>boom</p>")

        assert result == ""
        # Błąd jest zalogowany, nie podniesiony.
        assert "publish" in caplog.text.lower() or "disk full" in caplog.text

    def test_mkdir_failure_is_swallowed_and_returns_empty(
        self, tmp_path: Path, mocker
    ) -> None:
        target = tmp_path / "x" / "digest.html"
        publisher = FileWebPublisher(output_path=str(target))
        mocker.patch.object(Path, "mkdir", side_effect=OSError("perm denied"))

        result = publisher.publish("<p>boom</p>")

        assert result == ""


class TestNullWebPublisher:
    def test_publish_returns_empty_without_writing(
        self, tmp_path: Path, mocker
    ) -> None:
        # Gdyby Null próbował cokolwiek zapisać, ten patch by to wychwycił.
        write = mocker.patch.object(Path, "write_text")
        publisher = NullWebPublisher()

        result = publisher.publish("<html>ignored</html>")

        assert result == ""
        write.assert_not_called()

    def test_logs_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            NullWebPublisher().publish("<html>x</html>")

        assert "web digest" in caplog.text.lower() or "disabled" in caplog.text.lower()
