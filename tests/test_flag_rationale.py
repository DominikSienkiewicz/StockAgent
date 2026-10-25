"""DOC-19: uzasadnienia flag mieszkają w docs/decisions.md, a configuration.md na nie linkuje.

`test_markdown_links` zdejmuje kotwicę `#…` przed sprawdzeniem pliku, więc literówka w
`decisions.md#flag-…` albo wpis bez żadnego linku przeszłyby niezauważone. Ten strażnik
pilnuje obu kierunków: każdy link ma kotwicę, a każda kotwica ma link.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIGURATION = ROOT / "docs" / "configuration.md"
DECISIONS = ROOT / "docs" / "decisions.md"
RATIONALE_HEADING = "## Flag rationale"
SECTION_ANCHOR = "flag-rationale"  # slug nagłówka sekcji, nie kotwica wpisu

_LINK = re.compile(r"\]\(decisions\.md#(flag-[\w-]+)\)")
_ANCHOR = re.compile(r'<a id="(flag-[\w-]+)"></a>')


def _rationale_section() -> str:
    text = DECISIONS.read_text(encoding="utf-8")
    assert RATIONALE_HEADING in text, "docs/decisions.md nie ma sekcji Flag rationale"
    start = text.index(RATIONALE_HEADING)
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _linked_anchors() -> set[str]:
    found = set(_LINK.findall(CONFIGURATION.read_text(encoding="utf-8")))
    return found - {SECTION_ANCHOR}


def _defined_anchors() -> set[str]:
    return set(_ANCHOR.findall(_rationale_section()))


def test_every_rationale_link_in_configuration_has_an_anchor() -> None:
    linked = _linked_anchors()

    assert linked, "configuration.md nie linkuje do żadnego uzasadnienia flagi"
    assert sorted(linked - _defined_anchors()) == []


def test_every_rationale_entry_is_linked_from_configuration() -> None:
    defined = _defined_anchors()

    assert defined, "sekcja Flag rationale nie ma żadnej kotwicy flag-…"
    assert sorted(defined - _linked_anchors()) == []
