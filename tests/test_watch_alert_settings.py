"""Ustawienia alertów watcha (grupa W2-4): cisza nocna pushy (ALR-09).

Osobny plik zamiast dopisków do `test_config.py` / `test_config_toml.py`:
te same niezmienniki (domyślne wartości, klasyfikacja w `config_hash`,
produkcyjny `config.toml`) dla pól jednej grupy wpisów.
"""

from __future__ import annotations

import tomllib
from datetime import time
from pathlib import Path

import pytest

from src.config import NON_RECIPE_SETTINGS_FIELDS, Settings

CONFIG_TOML = Path(__file__).parent.parent / "config.toml"

_REQUIRED = {
    "openai_api_key": "sk-test",
    "finnhub_api_key": "fh",
    "alpha_vantage_api_keys": ["av1"],
    "supabase_url": "https://t.supabase.co",
    "supabase_key": "anon",
}


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **{**_REQUIRED, **overrides})  # type: ignore[arg-type]


def _toml() -> dict[str, object]:
    return tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))


class TestQuietHoursSettings:
    def test_defaults_off_with_the_night_window(self) -> None:
        settings = _settings()

        assert settings.quiet_hours_enabled is False
        assert settings.quiet_hours_start == time(22, 0)
        assert settings.quiet_hours_end == time(7, 0)

    def test_bounds_parse_from_hh_mm_text(self) -> None:
        settings = _settings(quiet_hours_start="23:30", quiet_hours_end="06:15")

        assert settings.quiet_hours_start == time(23, 30)
        assert settings.quiet_hours_end == time(6, 15)

    @pytest.mark.parametrize(
        "field", ["quiet_hours_enabled", "quiet_hours_start", "quiet_hours_end"]
    )
    def test_delivery_fields_stay_outside_the_recipe(self, field: str) -> None:
        assert field in NON_RECIPE_SETTINGS_FIELDS

    def test_real_config_toml_mutes_night_pushes(self) -> None:
        data = _toml()

        assert data["quiet_hours_enabled"] is True
        assert data["quiet_hours_start"] == "22:00"
        assert data["quiet_hours_end"] == "07:00"
