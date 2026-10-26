"""BEZ-24: inwentarz sekretów w SECURITY.md jest kompletny i nie zawiera wartości.

Po G23 (klucze z query stringa w publicznych logach GHA) rotacja była konieczna, a nie
było nawet listy, co i gdzie zmienić. Tabela „Secrets inventory and rotation” ma wiersz
dla każdego sekretu, który czyta workflow (`secrets.X`), i dla każdego klucza
`.env.example` poza przełącznikami środowiska — z rolą, kontem i procedurą rotacji.
Wiersz bez źródła to sekret, którego już nikt nie używa: znika z tabeli razem z kodem.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
SECURITY = ROOT / "SECURITY.md"
INVENTORY_HEADING = "## Secrets inventory and rotation"
# `steps.secrets.outputs` to wyjście kroku o id `secrets`, nie sekret — stąd lookbehind.
_WORKFLOW_SECRET = re.compile(r"(?<![\w.])secrets\.([A-Z][A-Z0-9_]*)")
_ENV_KEY = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)
# Przełączniki środowiska (BEZ-08, FIN-35) żyją w `.env`, ale nie są sekretami.
ENVIRONMENT_SWITCHES = frozenset({"DB_WRITES_ENABLED", "PAID_CALLS_ENABLED"})
# Kształty wartości i prefiksy kluczy, które nie mogą trafić do publicznego dokumentu.
_KEY_SHAPES = re.compile(
    r"\bsk-|\bsb_secret_|\bsb_publishable_|\bre_\S|\bage1[a-z0-9]"
    r"|postgres(?:ql)?://|hooks\.slack\.com/services|hc-ping\.com|\beyJ[A-Za-z0-9_-]{10}"
)


def _workflow_secrets() -> set[str]:
    return {
        name
        for path in (ROOT / ".github" / "workflows").glob("*.yml")
        for name in _WORKFLOW_SECRET.findall(path.read_text(encoding="utf-8"))
    }


def _env_example_secrets() -> set[str]:
    keys = set(_ENV_KEY.findall((ROOT / ".env.example").read_text(encoding="utf-8")))
    return keys - ENVIRONMENT_SWITCHES


def _inventory() -> set[str]:
    """Nazwy z pierwszej kolumny tabeli inwentarza (wiersz może nazwać kilka sekretów)."""
    text = SECURITY.read_text(encoding="utf-8")
    start = text.find(INVENTORY_HEADING)
    if start == -1:
        return set()
    end = text.find("\n## ", start + 1)
    section = text[start:] if end == -1 else text[start:end]
    return {
        name
        for line in section.splitlines()
        if line.startswith("| `")
        for name in re.findall(r"`([A-Z][A-Z0-9_]*)`", line.split("|")[1])
    }


def test_workflow_secret_pattern_skips_step_outputs() -> None:
    sample = "if: steps.secrets.outputs.configured\nkey: ${{ secrets.OPENAI_API_KEY }}"

    assert _WORKFLOW_SECRET.findall(sample) == ["OPENAI_API_KEY"]


def test_every_workflow_secret_is_documented() -> None:
    required = _workflow_secrets()

    assert "SUPABASE_KEY" in required, "wzorzec nie znajduje sekretów w workflowach"
    missing = sorted(required - _inventory())
    assert missing == [], f"SECURITY.md nie ma wiersza dla sekretów z workflowów: {missing}"


def test_every_env_example_secret_is_documented() -> None:
    required = _env_example_secrets()

    assert "SUPABASE_KEY" in required, "wzorzec nie znajduje kluczy .env.example"
    missing = sorted(required - _inventory())
    assert missing == [], f"SECURITY.md nie ma wiersza dla kluczy .env.example: {missing}"


def test_inventory_lists_only_secrets_that_are_still_used() -> None:
    stale = sorted(_inventory() - _workflow_secrets() - _env_example_secrets())

    assert stale == [], f"inwentarz wymienia sekrety, których nic nie czyta: {stale}"


def test_security_policy_carries_no_key_values_or_prefixes() -> None:
    found = sorted(set(_KEY_SHAPES.findall(SECURITY.read_text(encoding="utf-8"))))

    assert found == [], f"SECURITY.md zawiera kształt albo prefiks klucza: {found}"


def test_key_shape_pattern_catches_the_env_example_placeholders() -> None:
    placeholders = (
        "sk-...",
        "sk-ant-...",
        "sb_secret_...",
        "re_...",
        "age1replace-with-your-age-public-key",
        "postgresql://postgres.ref:pass@host:5432/postgres",
        "https://hooks.slack.com/services/T000/B000/x",
        "https://hc-ping.com/your-check-uuid",
    )

    assert [value for value in placeholders if not _KEY_SHAPES.search(value)] == []


# BEZ-24: datą ostatniej rotacji jest `updatedAt` sekretu na GitHubie, więc
# `scripts/gh-secrets-check.sh` ostrzega o każdym starszym niż 180 dni. Atrapa `gh` na
# PATH odpowiada listą sekretów z datami (TSV, jak `--jq ... | @tsv`); skrypt czyta
# workflowy prawdziwego repo, a pusty plik env odcina go od lokalnego `.env`.
SECRETS_CHECK = ROOT / "scripts" / "gh-secrets-check.sh"
ROTATION_MAX_AGE_DAYS = 180
_FAKE_GH = """#!/usr/bin/env bash
case "$1 $2" in
  "secret list") cat "$FAKE_GH_SECRETS" ;;
  *) exit 0 ;;
esac
"""


def _run_secrets_check(tmp_path: Path, updated: dict[str, datetime]) -> str:
    bash = shutil.which("bash")
    assert bash is not None, "brak bash w PATH"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(_FAKE_GH, encoding="utf-8")
    fake_gh.chmod(0o755)
    listing = tmp_path / "secrets.tsv"
    listing.write_text(
        "".join(f"{name}\t{when:%Y-%m-%dT%H:%M:%SZ}\n" for name, when in updated.items()),
        encoding="utf-8",
    )
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items() if key not in ("GH_TOKEN", "GITHUB_TOKEN")
    }
    environment |= {
        "PATH": f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}",
        "FAKE_GH_SECRETS": str(listing),
    }

    result = subprocess.run(
        [bash, str(SECRETS_CHECK), "-R", "owner/repo", "-f", str(env_file)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode in (0, 1), result.stderr
    return result.stdout


def test_secrets_check_warns_about_secrets_older_than_180_days(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    output = _run_secrets_check(
        tmp_path,
        {
            "OPENAI_API_KEY": now - timedelta(days=ROTATION_MAX_AGE_DAYS + 20),
            "SUPABASE_KEY": now - timedelta(days=10),
        },
    )
    stale = [line for line in output.splitlines() if "older than 180 days" in line]

    assert any("OPENAI_API_KEY" in line for line in stale), output
    assert not any("SUPABASE_KEY" in line for line in stale), output
