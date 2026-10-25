# tests/tools/test_validate_personas.py
"""Smoke testy CLI walidatora person."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.validate_personas import main as validate_main


def _write_persona(dirpath: Path, name: str, style: str = "Valid style " * 5) -> None:
    slug = name.lower().replace(" ", "-")
    (dirpath / f"{slug}.json").write_text(
        json.dumps({"name": name, "style": style}),
        encoding="utf-8",
    )


class TestValidatePersonasCLI:
    def test_exits_zero_when_directory_valid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_persona(tmp_path, "Warren Buffett")
        _write_persona(tmp_path, "Cathie Wood")

        rc = validate_main(["--dir", str(tmp_path)])

        assert rc == 0
        captured = capsys.readouterr()
        assert "2 persona" in captured.out
        assert "Warren Buffett" in captured.out

    def test_exits_one_when_directory_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "no-such-dir"
        rc = validate_main(["--dir", str(missing)])

        assert rc == 1
        assert str(missing) in capsys.readouterr().err

    def test_exits_one_on_malformed_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "broken.json").write_text("{not json}")
        rc = validate_main(["--dir", str(tmp_path)])

        assert rc == 1
        assert "broken.json" in capsys.readouterr().err

    def test_exits_one_on_empty_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = validate_main(["--dir", str(tmp_path)])
        assert rc == 1

    def test_default_dir_is_data_council_personas(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Brak --dir → użyj DEFAULT_PERSONAS_DIR. W repo ten katalog istnieje
        # z 7 wygenerowanymi plikami (rada ograniczona do 7 doradców).
        rc = validate_main([])
        assert rc == 0
        assert "7 persona" in capsys.readouterr().out


class TestValidateArchetypes:
    """MAIL-38: walidator pilnuje, że plik archetypów zna dokładnie persony rady."""

    def test_lists_the_archetype_of_each_persona(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_persona(tmp_path, "Cathie Wood")
        (tmp_path / "archetypes.toml").write_text(
            '"Cathie Wood" = "Innowacja i wzrost"\n', encoding="utf-8"
        )

        assert validate_main(["--dir", str(tmp_path)]) == 0
        assert "Cathie Wood → Innowacja i wzrost" in capsys.readouterr().out

    def test_exits_one_when_a_persona_has_no_archetype(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_persona(tmp_path, "Cathie Wood")
        _write_persona(tmp_path, "Warren Buffett")
        (tmp_path / "archetypes.toml").write_text(
            '"Cathie Wood" = "Innowacja i wzrost"\n', encoding="utf-8"
        )

        assert validate_main(["--dir", str(tmp_path)]) == 1
        assert "Warren Buffett" in capsys.readouterr().err
