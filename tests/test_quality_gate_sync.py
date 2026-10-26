"""Strażnik jednej bramki jakości: CI ↔ `[tool.mypy]` ↔ pliki agentowe ↔ docs.

Bramka żyła w kilku rozjechanych wersjach (`CLAUDE.md`, `AGENTS.md`, `ci.yml`,
`docs/running-locally.md`, `pyproject.toml`). Skutek: `main_watch.py` — wtedy
jedyny entry-point na aktywnym cronie — nie był w CI ani lintowany, ani
typowany, a gołe `uv run mypy` zgodne z konfiguracją dawało czerwony wynik.

Kontrakt:
1. CI lintuje i typuje KAŻDY `main_*.py` (jawnie albo przez `.` / gołe `mypy`
   z `[tool.mypy].files`).
2. Dokumenty dla agentów i ludzi cytują dokładnie te same komendy co CI.
3. Bramka pytest w plikach agentowych pomija testy `integration` (TEST-07): gołe
   `uv run pytest` przy wyeksportowanych kluczach woła płatne API i wysyła mail.

Dokument czytamy tak, jak widzi go agent: linia `@plik` to import, który Claude Code
zastępuje treścią pliku (DOC-29: `CLAUDE.md` = `@AGENTS.md` + dodatki Claude'a).
"""

from __future__ import annotations

import ast
import io
import re
import shlex
import tokenize
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GATE_DOCS = ("CLAUDE.md", "AGENTS.md", "docs/running-locally.md", "README.md")
GATE_TOOLS = ("ruff", "mypy")
AGENT_DOCS = ("CLAUDE.md", "AGENTS.md")
# TEST-13: katalogi `tests/` już objęte bramką mypy — lista tylko rośnie.
MYPY_RATCHET = (
    "tests/domain",
    "tests/tools",
    "tests/infrastructure",
    "tests/application",
    "tests/*.py",
)


_IMPORT_LINE = re.compile(r"^@(\S+)$")


def _doc_lines(doc: str) -> list[str]:
    """Linie dokumentu z rozwiniętymi importami `@plik` (ścieżka względem dokumentu)."""
    path = ROOT / doc
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        imported = _IMPORT_LINE.match(line.strip())
        if imported is None:
            lines.append(line)
        else:
            target = (path.parent / imported.group(1)).relative_to(ROOT)
            lines.extend(_doc_lines(target.as_posix()))
    return lines


def _entrypoints() -> list[str]:
    return sorted(path.name for path in ROOT.glob("main_*.py"))


def _from_tool(command: str, tool: str) -> list[str]:
    """Tokeny komendy od nazwy narzędzia, bez komentarza i flag `uv run`.

    `uv run --frozen --no-build --no-sync ruff check .` i `uv run ruff check .`
    normalizują się do tego samego `["ruff", "check", "."]`.
    """
    tokens = command.split("#", 1)[0].split()
    return tokens[tokens.index(tool) :] if tool in tokens else []


def _ci_commands() -> dict[str, list[str]]:
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"]["steps"]
    commands: dict[str, list[str]] = {}
    for step in steps:
        run = str(step.get("run", ""))
        for tool in GATE_TOOLS:
            tokens = _from_tool(run, tool)
            if tokens and tool not in commands:
                commands[tool] = tokens
    return commands


def _ci_tool_commands(tool: str) -> set[str]:
    """Wszystkie komendy narzędzia w jobie `quality` (CI-17: ruff check i ruff format --check)."""
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"]["steps"]
    runs = (_from_tool(str(step.get("run", "")), tool) for step in steps)
    return {" ".join(tokens) for tokens in runs if tokens}


def _mypy_files() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    files: list[str] = config["tool"]["mypy"]["files"]
    return files


def test_ci_quality_job_runs_ruff_and_mypy() -> None:
    assert set(_ci_commands()) == set(GATE_TOOLS)


@pytest.mark.parametrize("entrypoint", _entrypoints())
def test_ci_lints_and_typechecks_every_entrypoint(entrypoint: str) -> None:
    commands = _ci_commands()

    ruff_paths = commands["ruff"][2:]
    assert "." in ruff_paths or entrypoint in ruff_paths, (
        f"ci.yml nie lintuje {entrypoint}: {' '.join(commands['ruff'])}"
    )

    mypy_paths = [token for token in commands["mypy"][1:] if not token.startswith("-")]
    typed = mypy_paths or _mypy_files()
    assert entrypoint in typed, (
        f"ci.yml nie typuje {entrypoint}: {' '.join(commands['mypy'])} (pliki: {typed})"
    )


def test_bare_mypy_config_does_not_cover_untyped_tests() -> None:
    # Decyzja 2026-10-16: `tests/` spłacamy ratchetem po katalogach. Cały
    # katalog w `files` czerwieni gołe `mypy` (1326 błędów), zanim dług zniknie.
    assert "tests" not in _mypy_files()


@pytest.mark.parametrize("doc", GATE_DOCS)
@pytest.mark.parametrize("tool", GATE_TOOLS)
def test_agent_docs_quote_the_ci_gate(doc: str, tool: str) -> None:
    ci_commands = _ci_tool_commands(tool)
    quoted = {
        " ".join(_from_tool(line, tool))
        for line in _doc_lines(doc)
        if line.lstrip().startswith(f"uv run {tool}")
    }
    assert quoted, f"{doc} nie cytuje komendy `{tool}` z bramki"
    assert quoted == ci_commands, (
        f"{doc} cytuje {sorted(quoted)}, a CI uruchamia {sorted(ci_commands)}."
    )


def _pytest_marker(command: str) -> str | None:
    """Wyrażenie `-m` komendy pytest ('' bez `-m`), None gdy to nie pytest."""
    if "pytest" not in command:
        return None
    tokens = shlex.split(command.split("#", 1)[0])
    if "pytest" not in tokens:
        return None
    args = tokens[tokens.index("pytest") + 1 :]
    return args[args.index("-m") + 1] if "-m" in args[:-1] else ""


def _ci_pytest_marker() -> str | None:
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"]["steps"]
    markers = (_pytest_marker(str(step.get("run", ""))) for step in steps)
    return next((marker for marker in markers if marker is not None), None)


def test_pytest_marker_reads_the_selection_without_the_comment() -> None:
    assert _pytest_marker('uv run pytest -m "not integration"  # -m containers') == (
        "not integration"
    )
    assert _pytest_marker("uv run pytest") == ""
    assert _pytest_marker("uv run mypy") is None


@pytest.mark.parametrize("doc", AGENT_DOCS)
def test_agent_docs_pytest_gate_skips_live_api_tests(doc: str) -> None:
    ci_marker = _ci_pytest_marker()
    assert ci_marker is not None and "not integration" in ci_marker, (
        f"pierwszy krok pytest w ci.yml nie pomija testów integration: -m {ci_marker!r}"
    )
    quoted = [
        _pytest_marker(line)
        for line in _doc_lines(doc)
        if line.lstrip().startswith("uv run pytest")
    ]

    assert quoted, f"{doc} nie cytuje komendy pytest z bramki"
    assert all(marker is not None and "not integration" in marker for marker in quoted), (
        f"{doc}: bramka pytest bez 'not integration' ({quoted}) uruchomi testy na żywym API"
    )
    assert quoted[0] == ci_marker, f"{doc} cytuje -m {quoted[0]!r}, a CI -m {ci_marker!r}"


def test_mypy_ratchet_keeps_the_typed_test_directories() -> None:
    files = _mypy_files()

    assert "tests/domain" in files
    missing = [directory for directory in MYPY_RATCHET if directory not in files]
    assert not missing, f"[tool.mypy].files zgubiło katalogi ratchetu: {missing}"


def _covered_by_mypy(path: str, files: list[str]) -> bool:
    return any(
        PurePosixPath(path).match(entry)
        if "*" in entry
        else path == entry or path.startswith(f"{entry}/")
        for entry in files
    )


def test_mypy_gate_covers_every_test_module() -> None:
    """TEST-13: ratchet doszedł do końca — każdy moduł `tests/` jest pod strict mypy.
    Nowy katalog testów dopisz do `[tool.mypy].files` i `MYPY_RATCHET`."""
    files = _mypy_files()
    uncovered = [
        relative
        for path in sorted((ROOT / "tests").rglob("*.py"))
        if not _covered_by_mypy(relative := path.relative_to(ROOT).as_posix(), files)
    ]

    assert not uncovered, f"moduły testów poza bramką mypy: {uncovered[:5]}"


def _job_runs(job: str) -> list[str]:
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"][job]["steps"]
    return [str(step.get("run", "")) for step in steps]


# CI-11: `uv sync --frozen` instaluje z locka bez sprawdzenia zgodności z pyproject.toml,
# więc PR zmieniający zależność bez `uv lock` przechodził zielono. Joby bramki
# synchronizują z `--locked` (albo mają krok `uv lock --check`); workflowy produkcyjne
# zostają przy `--frozen`. Flagi się wykluczają, więc `--locked` zastępuje `--frozen`.
GATE_JOBS = ("quality", "sonarcloud")


@pytest.mark.parametrize("job", GATE_JOBS)
def test_ci_sync_asserts_lockfile_is_current(job: str) -> None:
    runs = _job_runs(job)
    syncs = [run.split() for run in runs if "uv sync" in run]

    assert syncs, f"job `{job}` w ci.yml nie ma kroku `uv sync`"
    lock_checked = any("uv lock --check" in run for run in runs)
    unchecked = [" ".join(sync) for sync in syncs if "--locked" not in sync]
    assert lock_checked or not unchecked, (
        f"job `{job}` synchronizuje bez sprawdzenia locka: {unchecked} — "
        "PR ze zmienioną zależnością bez `uv lock` przeszedłby zielono."
    )
    assert not [sync for sync in syncs if {"--locked", "--frozen"} <= set(sync)]


# CI-04: SonarCloud i jego coverage widziały tylko `src/`, a entry-pointy (`main_*.py`,
# ponad 2 tys. linii, w tym logika main_agent) wypadały z odznak Coverage i Quality
# Gate. Spadku pokrycia nie ratujemy wykluczeniami.
SONAR_PROPERTIES = ROOT / "sonar-project.properties"


def _sonar_property(key: str) -> list[str]:
    for line in SONAR_PROPERTIES.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _sonar_coverage_targets() -> list[str]:
    commands = [run for run in _job_runs("sonarcloud") if _pytest_marker(run) is not None]
    assert commands, "job `sonarcloud` nie uruchamia pytest"
    tokens = shlex.split(commands[0].split("#", 1)[0])
    return [token.removeprefix("--cov=") for token in tokens if token.startswith("--cov=")]


@pytest.mark.parametrize("entrypoint", _entrypoints())
def test_sonar_sources_cover_every_entrypoint(entrypoint: str) -> None:
    sources = _sonar_property("sonar.sources")

    assert "src" in sources
    assert entrypoint in sources, f"sonar.sources = {sources} pomija {entrypoint}"
    assert not _sonar_property("sonar.exclusions"), "pokrycia nie ratujemy wykluczeniami"


@pytest.mark.parametrize("entrypoint", _entrypoints())
def test_sonar_coverage_measures_every_entrypoint(entrypoint: str) -> None:
    targets = _sonar_coverage_targets()

    assert "src" in targets
    assert entrypoint.removesuffix(".py") in targets, (
        f"pytest w jobie sonarcloud mierzy {targets} — bez `--cov={entrypoint[:-3]}` "
        f"Sonar pokazuje {entrypoint} jako niepokryty w całości."
    )


def test_ci_container_step_requires_docker() -> None:
    # CI-06: bez `REQUIRE_DOCKER=1` brak demona daje skip całego modułu i zielony krok
    # z zerem wykonanych testów kontenerowych.
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"]["steps"]
    container_steps = [s for s in steps if _pytest_marker(str(s.get("run", ""))) == "containers"]

    assert len(container_steps) == 1
    assert dict(container_steps[0].get("env") or {}).get("REQUIRE_DOCKER") == "1"


# CI-03 (G68): próg coverage z gałęziami na src/domain z zapadką. Zmierzone 2026-10-24
# na `main` + paczka F: 98,49% (4392 instrukcji, 1118 gałęzi), czyli próg 98. Bramka 80%
# istniała tylko w SonarCloud i tylko z SONAR_TOKEN. Zapadka: próg tylko rośnie — podnieś
# `fail_under`, `--cov-fail-under` w ci.yml i tę stałą razem.
COVERAGE_RATCHET = 98
COVERAGE_TARGET = "src/domain"


def _coverage_section(name: str) -> dict[str, object]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section: dict[str, object] = config.get("tool", {}).get("coverage", {}).get(name, {})
    return section


def _fail_under() -> int:
    value = _coverage_section("report").get("fail_under", 0)
    assert isinstance(value, int), f"fail_under musi być liczbą całkowitą, jest {value!r}"
    return value


def _pytest_tokens(job: str) -> list[str]:
    commands = [run for run in _job_runs(job) if _pytest_marker(run) is not None]
    assert commands, f"job `{job}` nie uruchamia pytest"
    return shlex.split(commands[0].split("#", 1)[0])


def test_domain_coverage_floor_never_drops_below_the_ratchet() -> None:
    assert _fail_under() >= COVERAGE_RATCHET, (
        f"fail_under = {_fail_under()} < {COVERAGE_RATCHET}: próg coverage tylko rośnie "
        "(CI-03) — dopisz testy zamiast obniżać próg."
    )


def test_coverage_measures_branches() -> None:
    assert _coverage_section("run").get("branch") is True


def test_ci_quality_job_enforces_the_domain_coverage_floor() -> None:
    tokens = _pytest_tokens("quality")

    assert f"--cov={COVERAGE_TARGET}" in tokens
    assert "--cov-branch" in tokens
    assert f"--cov-fail-under={_fail_under()}" in tokens, (
        "ci.yml i [tool.coverage.report].fail_under mają ten sam próg"
    )


def test_sonarcloud_coverage_run_does_not_inherit_the_domain_floor() -> None:
    # pytest-cov czyta `fail_under` z pyproject także tam, gdzie mierzy cały `src`
    # i entry-pointy (niższe pokrycie niż domena) — job dla Sonara raportuje, nie bramkuje.
    assert "--cov-fail-under=0" in _pytest_tokens("sonarcloud")


def test_coverage_plugin_is_a_declared_dev_dependency() -> None:
    dev = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["dependency-groups"][
        "dev"
    ]

    assert any(str(requirement).startswith("pytest-cov") for requirement in dev)


# CI-15: ucieczki z typów w kodzie produkcyjnym są dziś wąskie, ale nic nie pilnowało, żeby
# takie zostały. Goły `# type: ignore` (bez kodu błędu) odrzuca mypy (`ignore-without-code`),
# a liczba ignore z kodem i `cast(Any, …)` w src/ i main_*.py tylko maleje. Stan na eb11938
# (2026-10-26): 5 ignore i 1 cast; ARCH-20, ARCH-21, ARCH-22 i ARCH-41 usunęły wszystkie ignore.
# Gdy zmiana usuwa ucieczkę, obniż zapadkę w tym samym commicie.
TYPE_IGNORE_BASELINE = 0
CAST_ANY_BASELINE = 1
_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?P<codes>\[[^\]]*\])?")


def _production_sources() -> dict[str, str]:
    paths = sorted([*(ROOT / "src").rglob("*.py"), *ROOT.glob("main_*.py")])
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in paths}


def count_type_ignores(source: str) -> tuple[int, int]:
    """(ignore z kodem, gołe ignore) — tylko w komentarzach, nie w napisach."""
    coded = bare = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        for match in _TYPE_IGNORE.finditer(token.string):
            if match.group("codes"):
                coded += 1
            else:
                bare += 1
    return coded, bare


def _named(node: ast.expr, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name
    )


def count_cast_any(source: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and _named(node.func, "cast")
        and node.args
        and _named(node.args[0], "Any")
    )


def test_type_escape_counters_read_comments_and_calls() -> None:
    source = (
        "from typing import Any, cast\nimport typing\n"
        "a = f()  # type: ignore[arg-type]\n"
        "b = g()  # type: ignore\n"
        'c = "# type: ignore"\n'
        "d = cast(Any, x)\n"
        "e = typing.cast(typing.Any, x)\n"
        "f = cast(int, x)\n"
    )

    assert count_type_ignores(source) == (1, 1)
    assert count_cast_any(source) == 2


def test_type_escapes_have_codes_and_do_not_grow() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    enabled = config["tool"]["mypy"].get("enable_error_code", [])
    sources = _production_sources()
    ignores = {name: count_type_ignores(text) for name, text in sources.items()}
    coded = sum(count for count, _ in ignores.values())
    bare = [name for name, (_, count) in ignores.items() if count]
    casts = sum(count_cast_any(text) for text in sources.values())

    assert "ignore-without-code" in enabled, (
        "[tool.mypy] bez enable_error_code = ['ignore-without-code'] przepuszcza goły "
        "`# type: ignore`, który ucisza każdy błąd w linii"
    )
    assert bare == [], f"goły `# type: ignore` bez kodu błędu w {bare}"
    assert coded <= TYPE_IGNORE_BASELINE, (
        f"{coded} `# type: ignore[...]` w src/ i main_*.py > zapadka {TYPE_IGNORE_BASELINE} "
        "— popraw typy zamiast uciszać mypy"
    )
    assert casts <= CAST_ANY_BASELINE, (
        f"{casts} `cast(Any, ...)` w src/ i main_*.py > zapadka {CAST_ANY_BASELINE}"
    )


# CI-27: jedyny hook pre-commit walidował persony; lint, format, lock i sekrety sprawdzało
# dopiero CI. Hooki lokalne cytują komendy ruff z joba `quality` (te same tokeny), lock
# sprawdza `uv lock --check`, sekrety gitleaks na staged, a składnię TOML/YAML hooki
# pre-commit-hooks. Zdalne repo hooków przypięte pełnym SHA (`# frozen: vX.Y.Z`).
# mypy zostaje w CI: w hooku jest za wolny.
PRECOMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
PRECOMMIT_REMOTE_HOOKS = frozenset({"gitleaks", "check-toml", "check-yaml"})


def _precommit_repos() -> list[dict[str, Any]]:
    config: dict[str, Any] = yaml.safe_load(PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    repos: list[dict[str, Any]] = config["repos"]
    return repos


def _precommit_hooks(local: bool) -> list[dict[str, Any]]:
    return [
        hook
        for repo in _precommit_repos()
        if (repo["repo"] == "local") is local
        for hook in repo["hooks"]
    ]


def test_precommit_mirrors_ci_linters() -> None:
    local_entries = [str(hook.get("entry", "")) for hook in _precommit_hooks(local=True)]
    ruff_hooks = {" ".join(_from_tool(entry, "ruff")) for entry in local_entries} - {""}
    remote_ids = {str(hook["id"]) for hook in _precommit_hooks(local=False)}
    remote_revs = [str(repo["rev"]) for repo in _precommit_repos() if repo["repo"] != "local"]

    assert ruff_hooks == _ci_tool_commands("ruff"), (
        f"pre-commit uruchamia {sorted(ruff_hooks)}, a CI {sorted(_ci_tool_commands('ruff'))}"
    )
    assert any(entry.split() == ["uv", "lock", "--check"] for entry in local_entries)
    assert "uv run python -m src.tools.validate_personas" in local_entries
    assert remote_ids >= PRECOMMIT_REMOTE_HOOKS, sorted(PRECOMMIT_REMOTE_HOOKS - remote_ids)
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in remote_revs), remote_revs
