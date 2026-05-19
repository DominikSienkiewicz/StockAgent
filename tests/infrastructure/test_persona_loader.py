# tests/infrastructure/test_persona_loader.py
"""Loader plików JSON z personami członków rady doradczej.

Każdy persona = jeden plik JSON `{"name": str, "style": str}`. Loader:
- czyta wszystkie *.json w katalogu (rekurencyjnie? NIE — flat),
- waliduje schemat (klucze name/style, niepuste, name unikatowe),
- zwraca deterministyczną kolejność (sort by name),
- ignoruje pliki nie-*.json (.gitkeep, README, .DS_Store).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.council import InvestorPersona
from src.infrastructure.persona_loader import (
    PersonaSchemaError,
    load_council_personas,
)


def _write_persona(dirpath: Path, name: str, style: str = "Detailed style text " * 5) -> Path:
    slug = name.lower().replace(" ", "-")
    path = dirpath / f"{slug}.json"
    path.write_text(
        json.dumps({"name": name, "style": style}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class TestLoadCouncilPersonas:
    def test_loads_all_json_files_in_directory(self, tmp_path: Path) -> None:
        _write_persona(tmp_path, "Warren Buffett")
        _write_persona(tmp_path, "Cathie Wood")

        personas = load_council_personas(tmp_path)

        assert len(personas) == 2
        assert all(isinstance(p, InvestorPersona) for p in personas)
        assert {p.name for p in personas} == {"Warren Buffett", "Cathie Wood"}

    def test_returns_tuple_sorted_by_name(self, tmp_path: Path) -> None:
        # Deterministyczna kolejność — bez tego rada miałaby losowy order
        # zależny od OS / FS i ewentualne testy by się rwały.
        _write_persona(tmp_path, "Zeta Investor")
        _write_persona(tmp_path, "Alpha Investor")
        _write_persona(tmp_path, "Mike Investor")

        personas = load_council_personas(tmp_path)

        names = [p.name for p in personas]
        assert names == ["Alpha Investor", "Mike Investor", "Zeta Investor"]

    def test_returns_tuple_not_list(self, tmp_path: Path) -> None:
        # Tuple = immutable contract dla downstream (LLMAdvisoryCouncil).
        _write_persona(tmp_path, "Warren Buffett")
        personas = load_council_personas(tmp_path)
        assert isinstance(personas, tuple)

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        _write_persona(tmp_path, "Warren Buffett")
        (tmp_path / ".gitkeep").write_text("")
        (tmp_path / "README.md").write_text("# personas")
        (tmp_path / ".DS_Store").write_bytes(b"\x00")

        personas = load_council_personas(tmp_path)

        assert len(personas) == 1

    def test_raises_when_directory_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost"
        with pytest.raises(FileNotFoundError, match=str(missing)):
            load_council_personas(missing)

    def test_raises_when_directory_is_empty(self, tmp_path: Path) -> None:
        # Pusta rada = brak werdyktu. Lepiej fail-fast przy starcie niż
        # cisza pod runtime.
        with pytest.raises(ValueError, match="no persona"):
            load_council_personas(tmp_path)

    def test_raises_on_duplicate_names_across_files(self, tmp_path: Path) -> None:
        _write_persona(tmp_path, "Warren Buffett")
        # Drugi plik z tym samym name (inna nazwa pliku)
        (tmp_path / "wb-dupe.json").write_text(
            json.dumps({"name": "Warren Buffett", "style": "a" * 30}),
            encoding="utf-8",
        )
        with pytest.raises(PersonaSchemaError, match="duplicate"):
            load_council_personas(tmp_path)

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not valid json}")
        with pytest.raises(PersonaSchemaError, match="broken.json"):
            load_council_personas(tmp_path)

    def test_raises_when_missing_name_field(self, tmp_path: Path) -> None:
        (tmp_path / "no-name.json").write_text(
            json.dumps({"style": "a" * 30}), encoding="utf-8"
        )
        with pytest.raises(PersonaSchemaError, match="name"):
            load_council_personas(tmp_path)

    def test_raises_when_missing_style_field(self, tmp_path: Path) -> None:
        (tmp_path / "no-style.json").write_text(
            json.dumps({"name": "Test"}), encoding="utf-8"
        )
        with pytest.raises(PersonaSchemaError, match="style"):
            load_council_personas(tmp_path)

    def test_raises_when_style_too_short(self, tmp_path: Path) -> None:
        # Sanity check: <20 znaków to prawie na pewno literówka, nie persona.
        (tmp_path / "short.json").write_text(
            json.dumps({"name": "Test", "style": "x"}), encoding="utf-8"
        )
        with pytest.raises(PersonaSchemaError, match="style.*too short"):
            load_council_personas(tmp_path)

    def test_raises_when_name_empty(self, tmp_path: Path) -> None:
        (tmp_path / "empty-name.json").write_text(
            json.dumps({"name": "", "style": "a" * 30}), encoding="utf-8"
        )
        with pytest.raises(PersonaSchemaError, match="name"):
            load_council_personas(tmp_path)


class TestRealPersonaFiles:
    """Sanity: pliki dostarczone razem z repo (data/council_personas/) ładują
    się czysto, dają tych samych 15 inwestorów co wcześniej."""

    def test_default_directory_yields_15_personas(self) -> None:
        from src.infrastructure.persona_loader import DEFAULT_PERSONAS_DIR

        personas = load_council_personas(DEFAULT_PERSONAS_DIR)
        names = {p.name for p in personas}
        assert len(personas) == 15
        assert "Warren Buffett" in names
        assert "Cathie Wood" in names
        assert "Joel Greenblatt" in names
