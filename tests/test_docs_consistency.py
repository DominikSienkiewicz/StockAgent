"""Strażnik twierdzeń docs o wagach XGBoost — tekst to twierdzenie, `git ls-files` to dowód.

Docs mówiły, że model „żyje w pliku `.ubj` w repo”, a Slow Loop „commituje nowe
wagi”, podczas gdy w gicie nie było żadnego pliku wag: trener pomija symbole z
za małą próbką, a workflow commituje plik tylko wtedy, gdy kandydat pobije
baseline. Dopóki żaden `.ubj` nie jest śledzony, docs nie mogą tego twierdzić.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

PUBLIC_DOCS = ("README.md", "docs/how-it-works.md", "docs/running-locally.md")

# Twierdzenia prawdziwe tylko przy commitowanych wagach.
COMMITTED_WEIGHTS_CLAIMS = (
    r"\.ubj` file inside the repo",
    r"commits the new weights back to the repo",
    r"\+ commit new weights",
)


def _tracked_weight_files() -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.ubj"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("brak repozytorium git — nie da się sprawdzić śledzonych wag")
    return [line for line in listed.stdout.splitlines() if line]


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
def test_docs_do_not_claim_committed_weights_when_none_tracked(doc: str) -> None:
    if _tracked_weight_files():
        pytest.skip("wagi są śledzone — twierdzenie o modelu w repo jest prawdziwe")

    text = (ROOT / doc).read_text(encoding="utf-8")
    found = [claim for claim in COMMITTED_WEIGHTS_CLAIMS if re.search(claim, text)]
    assert not found, f"{doc} twierdzi o commitowanych wagach, których nie ma: {found}"


def test_keys_called_unused_are_not_referenced_by_any_workflow() -> None:
    # DOC-03: docs nazywały TELEGRAM_*, SLACK_WEBHOOK_URL i FRED_API_KEY kluczami,
    # których „żaden workflow nie używa”, choć workflowy je czytają — operator
    # zostawiał przez to kanały push puste.
    text = (ROOT / "docs" / "running-locally.md").read_text(encoding="utf-8")
    workflows = "".join(
        path.read_text(encoding="utf-8") for path in (ROOT / ".github" / "workflows").glob("*.yml")
    )
    # Fraza aż do „are reported” (albo końca zdania) — „e.g.” w środku nie może jej uciąć.
    claims = re.findall(r"no workflow references(.{0,200}?)(?:are reported|\.\s+[A-Z])", text, re.S)
    named = {name for claim in claims for name in re.findall(r"`([A-Z][A-Z0-9_]*)\*?`", claim)}
    referenced = sorted(name for name in named if f"secrets.{name}" in workflows)
    assert not referenced, f"docs nazywają nieużywanymi klucze z workflowów: {referenced}"


# DOC-02: `alpha_fusion_score` trafia do `ml_feature_store` jako KANDYDAT — nie
# należy do `ML_FEATURE_COLUMNS`, więc nazwanie go „ósmą cechą” to fałsz, który
# recenzent wyłapie przy przejściu z config.py do ml_features.py. Migracje SQL
# zostają historyczne i ich nie sprawdzamy.
EIGHTH_FEATURE_CLAIM = re.compile(r"ósm\w* cech|eighth (?:ML |XGBoost )?feature", re.IGNORECASE)
EIGHTH_FEATURE_SOURCES = ("src/**/*.py", "main_*.py", "config.toml", "README.md", "docs/*.md")


def test_alpha_fusion_score_is_not_called_a_model_feature() -> None:
    from src.application.ml_features import ML_FEATURE_COLUMNS

    assert "alpha_fusion_score" not in ML_FEATURE_COLUMNS
    claims = [
        f"{path.relative_to(ROOT)}:{number}"
        for pattern in EIGHTH_FEATURE_SOURCES
        for path in sorted(ROOT.glob(pattern))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if EIGHTH_FEATURE_CLAIM.search(line)
    ]
    assert claims == []


def test_eighth_feature_pattern_matches_both_languages() -> None:
    assert EIGHTH_FEATURE_CLAIM.search("dokłada ósmą cechę ML")
    assert EIGHTH_FEATURE_CLAIM.search("as an eighth feature")
    assert not EIGHTH_FEATURE_CLAIM.search("kandydat na cechę ML")

def test_how_it_works_names_the_prediction_horizon_honestly() -> None:
    # WIA-28: pętla chodzi raz na dobę handlową, więc „12h return” to fałsz
    # widoczny dla quanta — horyzont to zwrot do następnego cyklu.
    text = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")

    assert "predicts a 12h _return_" not in text
    assert "return to the next cycle" in text


# DOC-08: liczby w how-it-works mają odpowiadać kodowi, nie dawnym wartościom.


def test_how_it_works_reflection_default_matches_settings() -> None:
    from src.config import Settings

    text = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")
    quoted = re.search(r"`reflection_min_age_hours` guard \(config, default `(\d+)`\)", text)

    assert quoted is not None
    assert int(quoted.group(1)) == Settings.model_fields["reflection_min_age_hours"].default


def test_how_it_works_quotes_the_anthropic_max_tokens() -> None:
    from src.infrastructure.llm.anthropic_client import DEFAULT_MAX_TOKENS

    text = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")
    quoted = re.findall(r"max_tokens (\d+)", text)

    assert quoted, "how-it-works nie podaje max_tokens adaptera Anthropic"
    assert {int(value) for value in quoted} == {DEFAULT_MAX_TOKENS}


# DOC-19: flagi jako tabela sprawdzana testem — proza z uzasadnieniami zostaje
# niżej, ale default kodu, wartość w config.toml, migracja, sekret i status
# wdrożenia stoją w jednym miejscu, które nie może się rozjechać z kodem.
FLAG_TABLE_HEADING = "## Flag reference"
DEPLOYED = "deployed"
NOT_DEPLOYED = "built, not deployed"
# Produkcja ma migracje do 025; 026+ czekają na ręczny DB Migrate
# (docs/resume-checklist.md, krok 3). Po migracji podnieś próg i statusy.
PRODUCTION_MIGRATED_THROUGH = 42


def _config_doc() -> str:
    return (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")


def _flag_rows() -> dict[str, list[str]]:
    text = _config_doc()
    start = text.index(FLAG_TABLE_HEADING)
    end = text.find("\n## ", start + 1)
    rows: dict[str, list[str]] = {}
    for line in text[start:end].splitlines():
        row = re.match(r"^\| `(\w+)` \|(.*)\|\s*$", line)
        if row:
            rows[row.group(1)] = [cell.strip() for cell in row.group(2).split("|")]
    return rows


def _bool_fields() -> dict[str, object]:
    from src.config import Settings

    return {
        name: field.default
        for name, field in Settings.model_fields.items()
        if field.annotation in (bool, bool | None)
    }


def _literal(value: object) -> str:
    if value is None:
        return "None"
    return "true" if value is True else "false" if value is False else repr(value)


def _first_code(cell: str) -> str | None:
    found = re.search(r"`([^`]+)`", cell)
    return found.group(1) if found else None


def _toml() -> dict[str, object]:
    import tomllib

    return tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))


def test_flag_table_matches_settings_and_config() -> None:
    from src.config import SECRET_SETTINGS_FIELDS

    rows = _flag_rows()
    flags = _bool_fields()
    toml = _toml()
    migrations = {path.name[:3] for path in (ROOT / "supabase" / "migrations").glob("*.sql")}

    assert sorted(rows) == sorted(flags), "tabela flag ≠ pola bool w Settings"
    problems = []
    for flag, cells in rows.items():
        default, in_toml, migration, secret, _paid, status = cells
        if _first_code(default) != _literal(flags[flag]):
            problems.append(f"{flag}: default kodu {default!r}")
        if flag in toml:
            if _first_code(in_toml) != _literal(toml[flag]):
                problems.append(f"{flag}: config.toml {in_toml!r} ≠ {toml[flag]!r}")
        elif not in_toml.startswith("—"):
            problems.append(f"{flag}: nie ma go w config.toml, a tabela mówi {in_toml!r}")
        numbers = re.findall(r"`(\d{3})`", migration)
        if (migration != "—") != bool(numbers) or not set(numbers) <= migrations:
            problems.append(f"{flag}: migracja {migration!r} nie wskazuje istniejącego pliku")
        names = re.findall(r"`([A-Z_]+)`", secret)
        if (secret != "—") != bool(names) or any(
            name.lower() not in SECRET_SETTINGS_FIELDS for name in names
        ):
            problems.append(f"{flag}: sekret {secret!r} spoza SECRET_SETTINGS_FIELDS")
        if status not in (DEPLOYED, NOT_DEPLOYED):
            problems.append(f"{flag}: status {status!r}")
        elif flag in toml:
            live = toml[flag] is True and all(
                int(number) <= PRODUCTION_MIGRATED_THROUGH for number in numbers
            )
            if (status == DEPLOYED) != live:
                problems.append(f"{flag}: status {status!r} przy config.toml {toml[flag]!r}")

    assert problems == []


def test_config_toml_migration_comments_point_at_existing_files() -> None:
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    migrations = {path.name[:3] for path in (ROOT / "supabase" / "migrations").glob("*.sql")}

    cited = set(re.findall(r"migracj\w*\s+(\d{3})", text))

    assert cited, "config.toml nie wskazuje żadnej migracji — wzorzec przestał działać"
    assert sorted(cited - migrations) == []


# DOC-13: tabela „niezmiennik → test” w how-it-works i README. Każdy wiersz
# linkuje plik testu i nazywa test (`Klasa::test_x` albo `test_x`), który w tym
# pliku istnieje — inaczej tabela obiecywałaby strażnika, którego nie ma.
INVARIANT_TABLES = {
    "docs/how-it-works.md": "## Invariants and the tests behind them",
    "README.md": "## Invariants under test",
}
MIN_INVARIANTS = 7


def _section_of(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _test_ids_in(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            ids.add(node.name)
        elif isinstance(node, ast.ClassDef):
            ids.update(
                f"{node.name}::{item.name}"
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_")
            )
    return ids


@pytest.mark.parametrize("doc", sorted(INVARIANT_TABLES))
def test_invariant_table_links_tests_that_exist(doc: str) -> None:
    path = ROOT / doc
    section = _section_of(path.read_text(encoding="utf-8"), INVARIANT_TABLES[doc])
    rows = [line for line in section.splitlines() if line.startswith("| ") and "](" in line]

    assert len(rows) >= MIN_INVARIANTS
    problems = []
    for row in rows:
        files = [
            (path.parent / target).resolve()
            for target in re.findall(r"\]\(([^)#]+\.py)\)", row)
        ]
        named = re.findall(r"`((?:\w+::)?test_\w+)`", row)
        known = set().union(*(_test_ids_in(file) for file in files if file.is_file()))
        if not files or not named or not set(named) <= known:
            problems.append((row[:60], sorted(set(named) - known)))

    assert problems == []
