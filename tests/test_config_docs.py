"""DOC-08: docs/configuration.md opisuje każde niesekretne pole `Settings`.

README zapowiada „Every `Settings` field with its default”, a kilkanaście pól
(w tym płatne `tool_use_enabled` i modele routingu) nie miało w docs ani słowa.
Sekrety (`SECRET_SETTINGS_FIELDS`) opisuje running-locally.md (blok `.env`).
"""

from __future__ import annotations

from pathlib import Path

from src.config import SECRET_SETTINGS_FIELDS, Settings

ROOT = Path(__file__).parent.parent
CONFIG_DOC = ROOT / "docs" / "configuration.md"


def test_every_non_secret_field_is_documented() -> None:
    text = CONFIG_DOC.read_text(encoding="utf-8")

    missing = sorted(
        field
        for field in Settings.model_fields
        if field not in SECRET_SETTINGS_FIELDS and f"`{field}`" not in text
    )

    assert missing == [], f"docs/configuration.md nie opisuje pól: {missing}"
