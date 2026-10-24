"""Regression guard — po centralizacji configu do `config.toml` workflow
wstrzykuje WYŁĄCZNIE sekrety; config niewrażliwy czytany jest wprost z pliku.

Historia: GitHub Actions nie eksportuje `vars.*`/`secrets.*` automatycznie —
trzeba je jawnie wymienić w `env:` kroku. Wcześniej trzymaliśmy tam też config
niewrażliwy (SYMBOLS, CRYPTO_SYMBOLS, COUNCIL_*…), co rodziło drift
`.env` ↔ `.env.example` ↔ workflow. Teraz ten config żyje w `config.toml`
(czytany przez `Settings`), więc workflow ma zostać czysty od niego.

Ten test pilnuje obu stron:
1. wszystkie SEKRETY są zmapowane (inaczej brak klucza w procesie → crash),
2. żaden config niewrażliwy nie wrócił do workflow (inaczej znów drift).
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "fast_loop_12h.yml"

# Sekrety, które MUSZĄ trafić do env kroku agenta (config.toml ich nie zawiera).
REQUIRED_SECRET_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEYS",
    "RESEND_API_KEY",
    "DIGEST_TO_EMAIL",
}

# Config niewrażliwy — należy do config.toml, NIE do workflow. Gdyby któryś tu
# wrócił, drift wraca razem z nim.
FORBIDDEN_NONSECRET_KEYS = {
    "SYMBOLS",
    "SYMBOLS_ETF",
    "CRYPTO_SYMBOLS",
    "CRYPTO_VOLATILITY_THRESHOLD",
    "VOLATILITY_THRESHOLD",
    "COUNCIL_VOLATILITY_THRESHOLD",
    "COUNCIL_LLM_PROVIDER",
    "COUNCIL_LLM_MODEL",
    "LLM_PROVIDER",
    "SYMBOL_THROTTLE_SECONDS",
    "SYMBOLS_UNSUPPORTED_PRICE",
    "RISK_SYMBOLS",
    "RISK_SYMBOL_TYPES",
    "NBP_ENABLED",
    "NOTIFICATIONS_ENABLED",
    "DIGEST_FROM_EMAIL",
    "ML_MODEL_PATH",
}


def _steps() -> list[dict[str, object]]:
    spec = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec["jobs"]["run-market-agent"]["steps"]


def _run_agent_env() -> dict[str, object]:
    run_step = next(s for s in _steps() if str(s.get("run", "")).endswith("main_agent.py"))
    return run_step.get("env", {})


def test_all_required_secrets_are_mapped() -> None:
    env_keys = set(_run_agent_env().keys())
    missing = REQUIRED_SECRET_KEYS - env_keys
    assert not missing, (
        f"fast_loop_12h.yml nie mapuje sekretów {sorted(missing)} do env "
        f"kroku agenta — bez nich proces dostanie pusty klucz i padnie."
    )


def test_no_non_secret_config_leaks_into_workflow() -> None:
    env_keys = set(_run_agent_env().keys())
    leaked = FORBIDDEN_NONSECRET_KEYS & env_keys
    assert not leaked, (
        f"Config niewrażliwy {sorted(leaked)} wrócił do workflow — miejsce na "
        f"to jest w config.toml (single source of truth), inaczej drift wraca."
    )


def test_config_toml_exists() -> None:
    assert (ROOT / "config.toml").is_file(), "brak config.toml w repo"


def test_dependency_sync_installs_anthropic_extra() -> None:
    sync_cmds = [
        str(s.get("run", "")) for s in _steps() if "uv sync" in str(s.get("run", ""))
    ]
    assert sync_cmds, "brak kroku `uv sync` w workflow"
    assert any("--extra anthropic" in cmd for cmd in sync_cmds), (
        "krok `uv sync` musi instalować `--extra anthropic` (LLM_PROVIDER=anthropic)."
    )


# Flagi, które obiecują PUBLICZNY artefakt (commitmenty/revealy, scorecard).
# Adaptery piszą wyłącznie na dysk runnera, który znika po jobie — bez kroku
# publikacji włączona flaga to twierdzenie bez pokrycia.
PUBLIC_ARTIFACT_FLAGS = ("attestation_enabled", "public_scorecard_enabled")
# Pliki ledgera atestacji (#16), które musi zabrać trwały `git push`.
ATTESTATION_LEDGER_KEYS = ("attestation_commitments_path", "attestation_reveals_path")


def _job_needs(job: Mapping[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _agent_publication_steps(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Kroki jobów z `main_agent.py` i jobów, które na nie czekają (`needs`, także
    przechodnio) — deploy Pages scorecardu idzie w osobnym jobie (WIA-16)."""
    jobs: Mapping[str, Any] = spec.get("jobs") or {}
    selected = {
        name
        for name, job in jobs.items()
        if any("main_agent.py" in str(step.get("run", "")) for step in job.get("steps", []))
    }
    grown = bool(selected)
    while grown:
        dependents = {name for name, job in jobs.items() if selected & set(_job_needs(job))}
        grown = not dependents <= selected
        selected |= dependents
    return [step for name, job in jobs.items() if name in selected for step in job.get("steps", [])]


def _agent_workflows() -> dict[str, list[dict[str, Any]]]:
    """Kroki publikacji (`_agent_publication_steps`) każdego workflowu z `main_agent.py`."""
    found = {name: _agent_publication_steps(spec) for name, spec in _all_workflow_specs().items()}
    return {name: steps for name, steps in found.items() if steps}


def _publishes_scorecard_to_pages(config: Mapping[str, Any], steps: list[dict[str, Any]]) -> bool:
    """Artefakt Pages z katalogu scorecardu i `actions/deploy-pages`."""
    scorecard_dir = posixpath.dirname(_normalized(str(config["public_scorecard_path"])))
    uploaded = any(
        str(step.get("uses", "")).startswith("actions/upload-pages-artifact")
        and _pages_upload_path(step) == scorecard_dir
        for step in steps
    )
    deployed = any(str(step.get("uses", "")).startswith("actions/deploy-pages") for step in steps)
    return uploaded and deployed


def _pushes_attestation_ledger(config: Mapping[str, Any], steps: list[dict[str, Any]]) -> bool:
    """`git add` obu plików ledgera i `git push` — deploy Pages podmienia całą witrynę,
    więc dopisywanej historii commitmentów nie przechowa."""
    scripts = [str(step.get("run", "")) for step in steps]
    added = [path for script in scripts for path in _git_added_paths(script)]
    ledger = [_normalized(str(config[key])) for key in ATTESTATION_LEDGER_KEYS]
    covered = all(any(_contains(path, entry) for path in added) for entry in ledger)
    return covered and any(_git_pushes(script) for script in scripts)


PUBLICATION_MECHANISMS = {
    "attestation_enabled": _pushes_attestation_ledger,
    "public_scorecard_enabled": _publishes_scorecard_to_pages,
}


def _missing_publications(config: Mapping[str, Any], steps: list[dict[str, Any]]) -> list[str]:
    """Włączone flagi publicznych artefaktów bez swojego mechanizmu publikacji."""
    return [
        flag
        for flag in PUBLIC_ARTIFACT_FLAGS
        if config.get(flag) is True and not PUBLICATION_MECHANISMS[flag](config, steps)
    ]


def test_enabled_public_artifacts_have_a_durable_publish_step() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    workflows = _agent_workflows()
    assert workflows, "żaden workflow nie uruchamia main_agent.py"

    unpublished = {
        name: missing
        for name, steps in workflows.items()
        if (missing := _missing_publications(config, steps))
    }
    assert not unpublished, (
        f"config.toml włącza {unpublished} bez publikacji — artefakt ginie z dyskiem "
        "runnera. `public_scorecard_enabled` wymaga uploadu katalogu scorecardu "
        "i `actions/deploy-pages`, `attestation_enabled` — `git add` ledgera i `git push`, "
        "w jobie agenta albo w jobie, który na niego czeka (`needs`)."
    )


def _agent_job(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"steps": [{"run": "uv run python main_agent.py"}, *steps]}


SCORECARD_UPLOAD = {
    "uses": "actions/upload-pages-artifact@v3",
    "with": {"path": "public/scorecard"},
}
SCORECARD_ONLY = {"public_scorecard_enabled": True, "attestation_enabled": False}
ATTESTATION_ONLY = {"public_scorecard_enabled": False, "attestation_enabled": True}


def _scorecard_deploy_workflow(needs: str | list[str] | None) -> dict[str, Any]:
    deploy: dict[str, Any] = {"steps": [{"uses": "actions/deploy-pages@v4"}]}
    if needs is not None:
        deploy["needs"] = needs
    return {"jobs": {"agent": _agent_job(SCORECARD_UPLOAD), "deploy": deploy}}


@pytest.mark.parametrize("needs", ["agent", ["agent"]])
def test_durable_guard_accepts_a_scorecard_deploy_in_a_job_that_needs_the_agent(
    needs: str | list[str],
) -> None:
    steps = _agent_publication_steps(_scorecard_deploy_workflow(needs))

    assert _missing_publications(_repo_config(**SCORECARD_ONLY), steps) == []


def test_durable_guard_follows_needs_transitively() -> None:
    spec = _scorecard_deploy_workflow("relay")
    spec["jobs"]["relay"] = {"needs": ["agent"], "steps": [{"run": "echo relay"}]}

    steps = _agent_publication_steps(spec)

    assert _missing_publications(_repo_config(**SCORECARD_ONLY), steps) == []


def test_durable_guard_ignores_a_deploy_job_without_needs() -> None:
    steps = _agent_publication_steps(_scorecard_deploy_workflow(None))

    assert _missing_publications(_repo_config(**SCORECARD_ONLY), steps) == [
        "public_scorecard_enabled"
    ]


def test_durable_guard_catches_an_enabled_scorecard_without_a_deploy() -> None:
    steps = _agent_publication_steps({"jobs": {"agent": _agent_job(SCORECARD_UPLOAD)}})

    assert _missing_publications(_repo_config(**SCORECARD_ONLY), steps) == [
        "public_scorecard_enabled"
    ]


def test_durable_guard_never_counts_a_pages_deploy_for_the_attestation_ledger() -> None:
    config = _repo_config(public_scorecard_enabled=True, attestation_enabled=True)
    steps = _agent_publication_steps(_scorecard_deploy_workflow("agent"))

    assert _missing_publications(config, steps) == ["attestation_enabled"]


@pytest.mark.parametrize(
    "script",
    [
        "git add public/attestation && git commit -m ledger && git push",
        "git -C public/attestation add commitments.jsonl reveals.jsonl\n"
        "git -C public/attestation commit -m ledger\n"
        "git -C public/attestation push",
    ],
)
def test_durable_guard_accepts_a_git_push_of_the_attestation_ledger(script: str) -> None:
    steps = _agent_publication_steps({"jobs": {"agent": _agent_job({"run": script})}})

    assert _missing_publications(_repo_config(**ATTESTATION_ONLY), steps) == []


def test_durable_guard_never_counts_a_push_without_the_ledger() -> None:
    script = "git add data/models/price_predictor.ubj && git commit -m weights && git push"
    steps = _agent_publication_steps({"jobs": {"agent": _agent_job({"run": script})}})

    assert _missing_publications(_repo_config(**ATTESTATION_ONLY), steps) == [
        "attestation_enabled"
    ]


def test_real_fast_loop_publishes_the_scorecard_once_its_flag_is_on() -> None:
    steps = _agent_workflows()["fast_loop_12h.yml"]

    assert _missing_publications(_repo_config(**SCORECARD_ONLY), steps) == []


# BEZ-09: `web_digest_enabled` zapisuje PEŁNY prywatny digest (watchlista, a przy
# `portfolio_positions_enabled` sekcja „💼 Twój portfel” z kwotami w zł) do
# `web_digest_path` pod `public/`. Deploy Pages albo commit `public/` przez git
# wypchnąłby go publicznie jedną flagą, więc flaga jest zakazana obok każdego
# takiego kroku, a artefakt Pages to wyłącznie katalog publicznego scorecardu.
PRIVATE_ARTIFACT_FLAGS = ("web_digest_enabled",)
PUBLIC_ROOT = "public"
# Domyślny `path` akcji `actions/upload-pages-artifact`, gdy krok go nie podaje.
PAGES_ARTIFACT_DEFAULT_PATH = "_site/"
# `git add` bez listy ścieżek, który zgarnia całe drzewo — `public/` nie jest
# w `.gitignore`, więc razem z nim leci digest.
GIT_ADD_SWEEP_FLAGS = ("-A", "--all")
# WIA-16: akcje gh-pages firm trzecich → (wejście z publikowanym katalogiem, jego
# domyślna wartość). `publish_dir` ma domyślnie `./public`; `folder` jest wymagany,
# więc brak wejścia liczymy jak całe drzewo.
THIRD_PARTY_PAGES_ACTIONS = {
    "peaceiris/actions-gh-pages": ("publish_dir", "public"),
    "jamesives/github-pages-deploy-action": ("folder", "."),
}


def _all_workflow_specs() -> dict[str, dict[str, Any]]:
    """Każdy plik workflow, nie tylko joby z `main_agent.py` — deploy Pages
    może siedzieć w osobnym jobie albo pliku."""
    workflows_dir = ROOT / ".github" / "workflows"
    paths = sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}


def _workflow_steps(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [step for job in (spec.get("jobs") or {}).values() for step in job.get("steps", [])]


def _normalized(path: str) -> str:
    return posixpath.normpath(path.strip().strip("'\""))


def _contains(parent: str, path: str) -> bool:
    """Czy publikacja `parent` zabiera `path` (ten sam katalog albo jego wnętrze)."""
    return parent in (".", path) or path.startswith(parent + "/")


def _overlaps(first: str, second: str) -> bool:
    return _contains(first, second) or _contains(second, first)


def _in_workdir(workdir: str, path: str) -> str:
    return posixpath.normpath(posixpath.join(workdir, path))


def _git_subcommand(tokens: list[str], index: int) -> tuple[int, str]:
    """Indeks podkomendy gita za opcjami `-C <dir>` i katalog, w którym działa (WIA-16)."""
    workdir = "."
    while tokens[index : index + 1] == ["-C"] and index + 1 < len(tokens):
        workdir = _in_workdir(workdir, _normalized(tokens[index + 1]))
        index += 2
    return index, workdir


def _git_commands(script: str) -> list[tuple[str, str, list[str]]]:
    """(katalog roboczy, podkomenda, argumenty) każdego wywołania gita w skrypcie kroku."""
    commands: list[tuple[str, str, list[str]]] = []
    for command in re.split(r"&&|\|\||[;\n]", script):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for start in (i for i, token in enumerate(tokens) if token == "git"):
            subcommand, workdir = _git_subcommand(tokens, start + 1)
            if subcommand < len(tokens):
                commands.append((workdir, tokens[subcommand], tokens[subcommand + 1 :]))
    return commands


def _git_added_paths(script: str) -> list[str]:
    """Ścieżki każdego `git add` w skrypcie kroku, względem katalogu repo (także
    `git -C <dir> add`); zgarnięcie całego drzewa jako katalog roboczy."""
    added: list[str] = []
    for workdir, subcommand, args in _git_commands(script):
        if subcommand != "add":
            continue
        for arg in args:
            if arg in GIT_ADD_SWEEP_FLAGS:
                added.append(workdir)
            elif not arg.startswith("-"):
                added.append(_in_workdir(workdir, _normalized(arg)))
    return added


def _git_pushes(script: str) -> bool:
    return any(subcommand == "push" for _, subcommand, _ in _git_commands(script))


def _third_party_pages_action(step: Mapping[str, Any]) -> tuple[str, str] | None:
    action = str(step.get("uses", "")).split("@", 1)[0].lower()
    return THIRD_PARTY_PAGES_ACTIONS.get(action)


def _is_pages_deploy(step: Mapping[str, Any]) -> bool:
    return (
        str(step.get("uses", "")).startswith("actions/deploy-pages")
        or _third_party_pages_action(step) is not None
    )


def _pages_upload_path(step: Mapping[str, Any]) -> str | None:
    """Katalog, który krok wysyła na Pages: artefakt `actions/upload-pages-artifact`
    albo katalog akcji gh-pages firmy trzeciej (WIA-16)."""
    inputs = step.get("with") or {}
    third_party = _third_party_pages_action(step)
    if third_party is not None:
        key, default = third_party
        return _normalized(str(inputs.get(key, default)))
    if not str(step.get("uses", "")).startswith("actions/upload-pages-artifact"):
        return None
    return _normalized(str(inputs.get("path", PAGES_ARTIFACT_DEFAULT_PATH)))


def _private_digest_publication_violations(
    config: Mapping[str, Any], workflows: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Kroki, które opublikowałyby prywatny digest: deploy Pages albo `git add`
    katalogu `public/` przy włączonym `web_digest_enabled`, oraz artefakt Pages
    inny niż katalog publicznego scorecardu (nigdy `public/` ani katalog digestu)."""
    scorecard_dir = posixpath.dirname(_normalized(str(config["public_scorecard_path"])))
    digest_dir = posixpath.dirname(_normalized(str(config["web_digest_path"])))
    private_enabled = [flag for flag in PRIVATE_ARTIFACT_FLAGS if config.get(flag) is True]
    violations: list[str] = []
    for name, spec in workflows.items():
        for step in _workflow_steps(spec):
            label = f"{name}: {step.get('name') or step.get('uses') or step.get('run')}"
            git_public = any(
                _overlaps(path, PUBLIC_ROOT) for path in _git_added_paths(str(step.get("run", "")))
            )
            if private_enabled and (_is_pages_deploy(step) or git_public):
                violations.append(
                    f"{label} publikuje Pages albo `{PUBLIC_ROOT}/`, a config.toml włącza "
                    f"{private_enabled} — prywatny digest (watchlista, kwoty portfela) "
                    f"z `{digest_dir}/` trafiłby publicznie."
                )
            upload = _pages_upload_path(step)
            if upload is None:
                continue
            if upload != scorecard_dir:
                violations.append(
                    f"{label} wysyła na Pages `{upload}` zamiast `{scorecard_dir}` "
                    f"(dirname `public_scorecard_path`)."
                )
            if _contains(upload, PUBLIC_ROOT) or _overlaps(upload, digest_dir):
                violations.append(
                    f"{label} wysyła na Pages `{upload}`, czyli całe `{PUBLIC_ROOT}/` albo "
                    f"prywatny digest z `{digest_dir}/`."
                )
    return violations


def _repo_config(**overrides: Any) -> dict[str, Any]:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    return {**config, **overrides}


def _synthetic_workflow(*steps: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Syntetyczny workflow z jednym jobem — Pages dziś nie istnieje (WIA-16)."""
    job = {"runs-on": "ubuntu-latest", "steps": list(steps)}
    return {"pages.yml": {"jobs": {"deploy": job}}}


def _pages_steps(path: str | None = "public/scorecard") -> tuple[dict[str, Any], ...]:
    upload: dict[str, Any] = {"uses": "actions/upload-pages-artifact@v3"}
    if path is not None:
        upload["with"] = {"path": path}
    return upload, {"uses": "actions/deploy-pages@v4"}


def test_pages_upload_never_includes_private_digest() -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow(*_pages_steps())
    )

    assert any("prywatny digest" in violation for violation in violations), violations


@pytest.mark.parametrize(
    "script",
    [
        "git add public/scorecard public/attestation && git commit -m pages && git push",
        "git add -A\ngit commit -m pages\ngit push",
        "git add .; git push",
        'git add -f "public/digest/index.html"',
    ],
)
def test_git_publication_of_public_dir_forbids_private_digest(script: str) -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow({"run": script})
    )

    assert any("prywatny digest" in violation for violation in violations), violations


@pytest.mark.parametrize(
    "script",
    [
        'MODEL_PATH="data/models/price_predictor.ubj"\ngit add "$MODEL_PATH"\ngit push',
        "git add data/models/price_predictor.ubj && git commit -m weights && git push",
    ],
)
def test_git_commit_outside_public_dir_allows_private_digest(script: str) -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow({"run": script})
    )

    assert violations == []


@pytest.mark.parametrize("path", ["public/scorecard", "./public/scorecard/"])
def test_pages_upload_of_the_scorecard_dir_alone_is_allowed(path: str) -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=False), _synthetic_workflow(*_pages_steps(path))
    )

    assert violations == []


@pytest.mark.parametrize("path", ["public", "./public/", "public/digest", ".", None])
def test_pages_upload_path_must_be_the_scorecard_dir(path: str | None) -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=False), _synthetic_workflow(*_pages_steps(path))
    )

    assert violations, f"artefakt Pages `{path}` przeszedł strażnika"


def test_pages_upload_never_publishes_public_root_even_for_a_scorecard_there() -> None:
    config = _repo_config(web_digest_enabled=False, public_scorecard_path="public/index.html")

    violations = _private_digest_publication_violations(
        config, _synthetic_workflow(*_pages_steps("public"))
    )

    assert violations, "artefakt Pages `public` zabrałby `public/digest/`"


# WIA-16: akcje gh-pages firm trzecich i `git -C <dir> add` publikują tak samo jak
# deploy Pages albo `git add` — strażnik nie może przepuścić ich bokiem.
THIRD_PARTY_PAGES_STEPS = (
    ("peaceiris/actions-gh-pages@v4", "publish_dir"),
    ("JamesIves/github-pages-deploy-action@v4", "folder"),
)


def _third_party_pages_step(uses: str, key: str, path: str | None) -> dict[str, Any]:
    step: dict[str, Any] = {"uses": uses}
    if path is not None:
        step["with"] = {key: path}
    return step


@pytest.mark.parametrize(("uses", "key"), THIRD_PARTY_PAGES_STEPS)
def test_third_party_pages_action_forbids_private_digest(uses: str, key: str) -> None:
    step = _third_party_pages_step(uses, key, "./public/scorecard")

    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow(step)
    )

    assert any("prywatny digest" in violation for violation in violations), violations


@pytest.mark.parametrize(("uses", "key"), THIRD_PARTY_PAGES_STEPS)
@pytest.mark.parametrize("path", ["public", "./public/", "public/digest", ".", None])
def test_third_party_pages_action_must_publish_the_scorecard_dir(
    uses: str, key: str, path: str | None
) -> None:
    step = _third_party_pages_step(uses, key, path)

    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=False), _synthetic_workflow(step)
    )

    assert violations, f"{uses} z `{key}: {path}` przeszedł strażnika"


@pytest.mark.parametrize(("uses", "key"), THIRD_PARTY_PAGES_STEPS)
def test_third_party_pages_action_of_the_scorecard_dir_alone_is_allowed(
    uses: str, key: str
) -> None:
    step = _third_party_pages_step(uses, key, "public/scorecard")

    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=False), _synthetic_workflow(step)
    )

    assert violations == []


@pytest.mark.parametrize(
    "script",
    [
        "git -C public add . && git -C public commit -m pages && git -C public push",
        "git -C public add -A",
        "git -C public/scorecard add index.html && git push",
        'git -C "./public" add "digest/index.html"',
    ],
)
def test_git_dash_c_add_of_public_dir_forbids_private_digest(script: str) -> None:
    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow({"run": script})
    )

    assert any("prywatny digest" in violation for violation in violations), violations


def test_git_dash_c_add_outside_public_dir_allows_private_digest() -> None:
    script = "git -C data/models add price_predictor.ubj && git -C data/models push"

    violations = _private_digest_publication_violations(
        _repo_config(web_digest_enabled=True), _synthetic_workflow({"run": script})
    )

    assert violations == []


def test_real_workflows_never_publish_private_digest() -> None:
    workflows = _all_workflow_specs()
    assert {"fast_loop_12h.yml", "slow_loop_weekly.yml"} <= set(workflows)

    violations = _private_digest_publication_violations(_repo_config(), workflows)

    assert not violations, violations


# WIA-16/WIA-52: deploy publicznego scorecardu na Pages. Artefakt to sam katalog
# scorecardu, wysyłany tylko gdy plik powstał (`hashFiles`), a deploy idzie
# w osobnym jobie z environment `github-pages` — wyłącznie ten job dostaje
# `pages: write` i `id-token: write`.
PAGES_WORKFLOWS = ("fast_loop_12h.yml", "scorecard_pages.yml")
PAGES_PERMISSIONS = ("pages", "id-token")
PAGES_ENVIRONMENT = "github-pages"


def _job_steps_using(jobs: Mapping[str, Any], action: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, step)
        for name, job in jobs.items()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith(action)
    ]


def _pages_permission_holders(permissions: Any) -> list[str]:
    granted = permissions if isinstance(permissions, Mapping) else {}
    return [permission for permission in PAGES_PERMISSIONS if permission in granted]


def _pages_artifact_violations(upload: Mapping[str, Any], scorecard_file: str) -> list[str]:
    violations: list[str] = []
    path = _pages_upload_path(upload)
    if path != posixpath.dirname(scorecard_file):
        violations.append(f"artefakt Pages `{path}` zamiast katalogu scorecardu")
    if f"hashFiles('{scorecard_file}')" not in str(upload.get("if", "")):
        violations.append(f"upload bez `if: hashFiles('{scorecard_file}')` (flaga off)")
    return violations


def _pages_deploy_job_violations(name: str, job: Mapping[str, Any], build_job: str) -> list[str]:
    violations: list[str] = []
    if build_job not in _job_needs(job):
        violations.append(f"job `{name}` nie czeka na `{build_job}` (`needs`)")
    if f"needs.{build_job}.outputs." not in str(job.get("if", "")):
        violations.append(f"job `{name}` rusza także bez artefaktu (`if` bez outputs)")
    environment = job.get("environment")
    environment_name = environment.get("name") if isinstance(environment, Mapping) else environment
    if environment_name != PAGES_ENVIRONMENT:
        violations.append(f"job `{name}` bez environment `{PAGES_ENVIRONMENT}`")
    permissions = job.get("permissions") or {}
    if any(permissions.get(permission) != "write" for permission in PAGES_PERMISSIONS):
        violations.append(f"job `{name}` bez `pages: write` i `id-token: write`")
    return violations


def _pages_deploy_violations(spec: Mapping[str, Any], scorecard_path: str) -> list[str]:
    """Odstępstwa workflowu od kontraktu deployu scorecardu na Pages."""
    jobs: Mapping[str, Any] = spec.get("jobs") or {}
    uploads = _job_steps_using(jobs, "actions/upload-pages-artifact")
    deploys = _job_steps_using(jobs, "actions/deploy-pages")
    if len(uploads) != 1 or len(deploys) != 1:
        return [f"oczekiwany jeden upload i jeden deploy Pages, są {len(uploads)} i {len(deploys)}"]
    (build_job, upload), (deploy_job, _) = uploads[0], deploys[0]
    violations = _pages_artifact_violations(upload, _normalized(scorecard_path))
    if build_job == deploy_job:
        return [*violations, f"deploy Pages w jobie budującym `{build_job}` zamiast osobnego"]
    violations += _pages_deploy_job_violations(deploy_job, jobs[deploy_job], build_job)
    elsewhere = [
        f"{name}: {held}"
        for name, permissions in [
            ("workflow", spec.get("permissions")),
            *((job_name, job.get("permissions")) for job_name, job in jobs.items()),
        ]
        if name != deploy_job
        for held in _pages_permission_holders(permissions)
    ]
    if elsewhere:
        violations.append(f"uprawnienia Pages poza jobem deploy: {elsewhere}")
    return violations


@pytest.mark.parametrize("workflow", PAGES_WORKFLOWS)
def test_pages_workflows_deploy_only_the_scorecard_dir_from_a_separate_job(workflow: str) -> None:
    spec = _all_workflow_specs()[workflow]

    violations = _pages_deploy_violations(spec, str(_repo_config()["public_scorecard_path"]))

    assert violations == []


def _pages_workflow() -> dict[str, Any]:
    """Syntetyczny workflow zgodny z kontraktem deployu Pages."""
    upload = {
        "id": "artifact",
        "if": "hashFiles('public/scorecard/index.html') != ''",
        "uses": "actions/upload-pages-artifact@v3",
        "with": {"path": "public/scorecard"},
    }
    build = {"outputs": {"page": "${{ steps.artifact.outcome }}"}, "steps": [upload]}
    deploy = {
        "needs": "build",
        "if": "needs.build.outputs.page == 'success'",
        "environment": {"name": "github-pages"},
        "permissions": {"contents": "read", "pages": "write", "id-token": "write"},
        "steps": [{"uses": "actions/deploy-pages@v4"}],
    }
    return {"permissions": {"contents": "read"}, "jobs": {"build": build, "deploy": deploy}}


def test_pages_deploy_guard_accepts_the_contract() -> None:
    assert _pages_deploy_violations(_pages_workflow(), "public/scorecard/index.html") == []


def test_pages_deploy_guard_catches_a_deploy_in_the_build_job() -> None:
    spec = _pages_workflow()
    spec["jobs"]["build"]["steps"].append({"uses": "actions/deploy-pages@v4"})
    del spec["jobs"]["deploy"]

    violations = _pages_deploy_violations(spec, "public/scorecard/index.html")

    assert any("osobnego" in violation for violation in violations), violations


def test_pages_deploy_guard_catches_pages_permissions_outside_the_deploy_job() -> None:
    spec = _pages_workflow()
    spec["permissions"]["pages"] = "write"

    violations = _pages_deploy_violations(spec, "public/scorecard/index.html")

    assert any("poza jobem deploy" in violation for violation in violations), violations


def _slow_loop_trainer_env() -> dict[str, object]:
    workflow = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    steps = [s for job in spec["jobs"].values() for s in job.get("steps", [])]
    run_step = next(s for s in steps if str(s.get("run", "")).endswith("main_trainer.py"))
    return run_step.get("env", {})


def test_slow_loop_maps_real_alpha_vantage_secret() -> None:
    # DAT-04: Slow Loop odświeża fundamenty z Alpha Vantage. Z atrapą klucza AV
    # odrzucał każde żądanie jako wyczerpany limit i `fundamentals_cache` zostawał pusty.
    expected = "${{ secrets.ALPHA_VANTAGE_API_KEYS }}"
    assert _slow_loop_trainer_env().get("ALPHA_VANTAGE_API_KEYS") == expected


# MAIL-04: funkcje push w Fast Loopie i sekrety kanałów, bez których milczą.
PUSH_FEATURE_FLAGS = ("messenger_digest_enabled", "realtime_alerts_enabled")
PUSH_CHANNEL_SECRETS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SLACK_WEBHOOK_URL")


def test_enabled_push_features_have_channel_secrets_mapped() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    enabled = [flag for flag in PUSH_FEATURE_FLAGS if config.get(flag) is True]
    if not enabled:
        pytest.skip("żadna funkcja push nie jest włączona")
    env = _run_agent_env()
    unmapped = [
        key for key in PUSH_CHANNEL_SECRETS if env.get(key) != f"${{{{ secrets.{key} }}}}"
    ]
    assert not unmapped, (
        f"{enabled} włączone, ale krok agenta nie mapuje {unmapped} — push nie ma kanału."
    )


def test_slow_loop_commits_weights_even_when_training_step_failed() -> None:
    # ML-04: trener kończy się exit 1, gdy padł choć jeden symbol. Bez `!cancelled()`
    # krok commitu był wtedy pomijany i wagi wspólnego modelu przepadały z runnerem.
    workflow = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    steps = [s for job in spec["jobs"].values() for s in job.get("steps", [])]
    commit_step = next(s for s in steps if "gh release upload" in str(s.get("run", "")))
    assert str(commit_step.get("if", "")).replace(" ", "") == "${{!cancelled()}}"


# CI-13 (G78): wagi modelu żyją jako asset Release `model-latest`, nie commit na
# main — bot nie omija branch protection. Slow Loop wysyła plik `ml_model_path`,
# Fast Loop pobiera go przed agentem, a dwa treningi nigdy nie biegną naraz.
MODEL_RELEASE_TAG = "model-latest"
SLOW_LOOP = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"


def _slow_loop_spec() -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(SLOW_LOOP.read_text(encoding="utf-8"))
    return spec


def _ml_model_path() -> str:
    return str(_repo_config()["ml_model_path"])


def _release_step(steps: list[dict[str, Any]], command: str) -> dict[str, Any] | None:
    return next(
        (s for s in steps if f"{command} {MODEL_RELEASE_TAG}" in str(s.get("run", ""))), None
    )


def test_slow_loop_never_pushes_to_main() -> None:
    steps = _workflow_steps(_slow_loop_spec())
    scripts = [str(step.get("run", "")) for step in steps]

    assert not [run for run in scripts if re.search(r"\bgit\s+(push|commit)\b", run)], (
        "Slow Loop commituje albo pushuje do repo — wagi idą do Release model-latest."
    )


def test_slow_loop_uploads_the_weights_file_to_the_model_release() -> None:
    upload = _release_step(_workflow_steps(_slow_loop_spec()), "gh release upload")

    assert upload is not None, "Slow Loop nie wysyła wag do Release model-latest"
    assert _ml_model_path() in str(upload.get("run", "")) + str(upload.get("env", ""))
    assert "--clobber" in str(upload.get("run", ""))
    assert dict(upload.get("env") or {}).get("GH_TOKEN") == "${{ github.token }}"


def test_slow_loop_runs_queue_instead_of_cancelling() -> None:
    concurrency = _slow_loop_spec().get("concurrency") or {}

    assert concurrency.get("group") == "slow-loop"
    assert concurrency.get("cancel-in-progress") is False


def test_fast_loop_downloads_the_weights_before_the_agent_runs() -> None:
    steps = _steps()
    download = _release_step(steps, "gh release download")
    agent = next(s for s in steps if str(s.get("run", "")).endswith("main_agent.py"))

    assert download is not None, "Fast Loop nie pobiera wag z Release model-latest"
    assert steps.index(download) < steps.index(agent)
    model_path = _ml_model_path()
    run = str(download.get("run", ""))
    assert posixpath.basename(model_path) in run
    assert posixpath.dirname(model_path) in run
    assert dict(download.get("env") or {}).get("GH_TOKEN") == "${{ github.token }}"


def test_slow_loop_write_permission_is_scoped_to_the_trainer_job() -> None:
    # BEZ-12: `contents: write` na poziomie workflowu dostałby każdy dopisany job.
    # Upload wag do Release (CI-13) idzie domyślnym GITHUB_TOKEN, więc checkout nie
    # potrzebuje `token:`.
    workflow = ROOT / ".github" / "workflows" / "slow_loop_weekly.yml"
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    job = spec["jobs"]["retrain-predictive-model"]
    checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))

    assert "write" not in dict(spec.get("permissions") or {}).values()
    assert job.get("permissions") == {"contents": "write"}
    assert "token" not in dict(checkout.get("with") or {})


# BEZ-03: GitHub maskuje sekret tylko w całości. Element sekretu CSV (jeden klucz
# z `ALPHA_VANTAGE_API_KEYS=k1,k2`) wyszedłby w logu jawnie, więc każdy job, który
# wstrzykuje ten sekret, rejestruje `::add-mask::` per klucz, zanim ruszy Python.
CSV_SECRET = "ALPHA_VANTAGE_API_KEYS"
CSV_SECRET_REF = f"${{{{ secrets.{CSV_SECRET} }}}}"
MASK_SCRIPT = ROOT / "scripts" / "mask-csv-secret.sh"


def _jobs_injecting_csv_secret() -> list[tuple[str, list[dict[str, object]]]]:
    """(workflow/job, kroki) dla każdego jobu, który mapuje prawdziwy sekret CSV."""
    found: list[tuple[str, list[dict[str, object]]]] = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in spec.get("jobs", {}).items():
            steps = job.get("steps", [])
            if any(dict(s.get("env") or {}).get(CSV_SECRET) == CSV_SECRET_REF for s in steps):
                found.append((f"{path.name}/{job_name}", steps))
    return found


def _is_mask_step(step: dict[str, object]) -> bool:
    env = dict(step.get("env") or {})
    run = str(step.get("run", ""))
    return env.get(CSV_SECRET) == CSV_SECRET_REF and (
        f"scripts/mask-csv-secret.sh {CSV_SECRET}" in run
    )


def test_guard_sees_every_workflow_injecting_the_alpha_vantage_secret() -> None:
    workflows = {name.split("/")[0] for name, _ in _jobs_injecting_csv_secret()}

    assert {"fast_loop_12h.yml", "slow_loop_weekly.yml"} <= workflows


def test_every_alpha_vantage_key_is_masked_before_python_runs() -> None:
    unmasked: list[str] = []
    for name, steps in _jobs_injecting_csv_secret():
        mask = next((i for i, s in enumerate(steps) if _is_mask_step(s)), None)
        python = [i for i, s in enumerate(steps) if "python" in str(s.get("run", ""))]
        if mask is None or (python and mask > min(python)):
            unmasked.append(name)

    assert not unmasked, (
        f"{unmasked} wstrzykują {CSV_SECRET} bez kroku `scripts/mask-csv-secret.sh "
        f"{CSV_SECRET}` przed Pythonem — pojedynczy klucz z listy wyjdzie w logu jawnie."
    )


def _run_mask_script(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ["PATH"]}
    if value is not None:
        env[CSV_SECRET] = value
    return subprocess.run(
        [str(MASK_SCRIPT), CSV_SECRET], env=env, capture_output=True, text=True, check=False
    )


def test_mask_script_is_executable() -> None:
    assert MASK_SCRIPT.is_file()
    assert os.access(MASK_SCRIPT, os.X_OK)


def test_mask_script_masks_each_trimmed_key_and_prints_nothing_else() -> None:
    # Split i trim jak `Settings._parse_av_keys`: przecinek, spacje, puste elementy, CR.
    result = _run_mask_script(" av-key-one , av-key-two,,\tav-key-three\r")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "::add-mask::av-key-one",
        "::add-mask::av-key-two",
        "::add-mask::av-key-three",
    ]
    assert result.stderr == ""


@pytest.mark.parametrize("value", [None, "", " , "])
def test_mask_script_without_keys_is_a_quiet_no_op(value: str | None) -> None:
    # Fork / brak sekretu: krok nie może wywalić jobu ani niczego wypisać.
    result = _run_mask_script(value)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


# NIEZ-13: zewnętrzny heartbeat — każdy cron pętli pinguje go na koniec przebiegu.
# Brak sekretu w repo daje pusty string, a pusty `HEARTBEAT_URL` = NullHeartbeat.
HEARTBEAT_ENTRYPOINTS = {
    "fast_loop_12h.yml": "main_agent.py",
    "slow_loop_weekly.yml": "main_trainer.py",
    "shock_watch_hourly.yml": "main_watch.py",
}


@pytest.mark.parametrize(("workflow", "entrypoint"), sorted(HEARTBEAT_ENTRYPOINTS.items()))
def test_loop_workflows_map_the_heartbeat_secret(workflow: str, entrypoint: str) -> None:
    spec = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8"))
    steps = [step for job in spec["jobs"].values() for step in job.get("steps", [])]
    run_step = next(s for s in steps if str(s.get("run", "")).endswith(entrypoint))

    assert run_step.get("env", {}).get("HEARTBEAT_URL") == "${{ secrets.HEARTBEAT_URL }}"


def test_env_example_ships_a_heartbeat_placeholder() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1] for line in lines if line.startswith("HEARTBEAT_URL=")]

    assert values == ["https://hc-ping.com/your-check-uuid"]


def test_env_example_leaves_gh_token_empty() -> None:
    """scripts/gh-secrets-*.sh eksportują każdy niepusty GH_TOKEN z `.env` jako token
    `gh`, więc placeholder skopiowany z szablonu przesłaniał działający login."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1] for line in lines if line.startswith("GH_TOKEN=")]

    assert values == [""]


def test_env_example_ships_a_session_pooler_db_url_placeholder() -> None:
    """DB Migrate czyta `secrets.SUPABASE_DB_URL`. Direct connection (`db.<ref>`) jest
    tylko IPv6, a runnery GitHuba nie mają IPv6 — działa jedynie session pooler (5432)."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1] for line in lines if line.startswith("SUPABASE_DB_URL=")]

    assert len(values) == 1
    assert re.fullmatch(
        r"postgresql://postgres\.[^:@]+:[^@]+@[^/]+\.pooler\.supabase\.com:5432/postgres",
        values[0],
    )


# WIA-11: warunki wznowienia crona Fast Loopa sprawdzalne z repo. Cron zakomentowano
# w 02fc397 (decyzja 2), a cisza 08-17…10-15 przeszła bez alarmu — włączony
# `schedule` wymaga deadmana z aktywnym cronem, sufitu kosztu cyklu, budżetu
# tygodniowego, progu liveness Fast Loopa oraz jawnego zapisu i płatnych wywołań
# w kroku agenta. Zgodność `fast_loop_cycles_per_day` z cronem pilnuje FIN-32
# (tests/test_config_toml.py). Reszta checklisty: docs/resume-checklist.md.
FAST_LOOP_WORKFLOW = "fast_loop_12h.yml"


def _active_schedule(spec: Mapping[str, Any]) -> list[Any]:
    """Wpisy `schedule` workflow; zakomentowany blok = pusta lista."""
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True)) or {}
    schedule = triggers.get("schedule") if isinstance(triggers, Mapping) else None
    return list(schedule or [])


def _entrypoint_env(spec: Mapping[str, Any], entrypoint: str) -> dict[str, Any]:
    step = next(
        (s for s in _workflow_steps(spec) if str(s.get("run", "")).endswith(entrypoint)), {}
    )
    return dict(step.get("env") or {})


def _scheduled_deadmen(workflows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Workflowy, które uruchamiają `main_deadman.py` z aktywnego crona."""
    return [
        name
        for name, spec in workflows.items()
        if _active_schedule(spec)
        and any("main_deadman.py" in str(s.get("run", "")) for s in _workflow_steps(spec))
    ]


def _assert_fast_loop_resume_conditions(
    config: Mapping[str, Any], workflows: Mapping[str, Mapping[str, Any]]
) -> None:
    assert _scheduled_deadmen(workflows), (
        f"{FAST_LOOP_WORKFLOW} ma aktywny cron, ale żaden workflow nie uruchamia "
        "main_deadman.py z crona — deadman nie zauważy, że Fast Loop zamilkł "
        "(cisza 08-17…10-15 przeszła bez alarmu)."
    )
    assert config.get("max_paid_llm_calls_per_cycle") is not None, (
        "aktywny cron Fast Loopa bez `max_paid_llm_calls_per_cycle` w config.toml — "
        "cykl nie ma sufitu płatnych wywołań LLM (FIN-12)."
    )
    assert config.get("fast_loop_weekly_budget_usd") is not None, (
        "aktywny cron Fast Loopa bez `fast_loop_weekly_budget_usd` w config.toml — "
        "start nie porówna tygodniowego najgorszego przypadku z budżetem."
    )
    assert float(config.get("liveness_fast_loop_max_gap_h", 0)) > 0, (
        "aktywny cron Fast Loopa przy `liveness_fast_loop_max_gap_h = 0` w config.toml — "
        "deadman traktuje Fast Loop jako świadomie wstrzymany i go nie sprawdza."
    )
    env = _entrypoint_env(workflows[FAST_LOOP_WORKFLOW], "main_agent.py")
    assert env.get("PAID_CALLS_ENABLED") == "true", (
        f"{FAST_LOOP_WORKFLOW} z aktywnym cronem musi ustawiać PAID_CALLS_ENABLED: "
        '"true" — inaczej cykl idzie na atrapach (FIN-35).'
    )
    assert env.get("DB_WRITES_ENABLED") == "true", (
        f"{FAST_LOOP_WORKFLOW} z aktywnym cronem musi ustawiać DB_WRITES_ENABLED: "
        '"true" — inaczej cykl nie zapisuje snapshotów ani predykcji (BEZ-08).'
    )


def test_enabled_fast_loop_schedule_requires_deadman() -> None:
    workflows = _all_workflow_specs()
    if not _active_schedule(workflows[FAST_LOOP_WORKFLOW]):
        pytest.skip("cron Fast Loopa zakomentowany — warunki wznowienia: docs/resume-checklist.md")

    _assert_fast_loop_resume_conditions(_repo_config(), workflows)


def _resume_ready_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "max_paid_llm_calls_per_cycle": 100,
        "fast_loop_weekly_budget_usd": 7.0,
        "liveness_fast_loop_max_gap_h": 26,
    }
    return {**config, **overrides}


def _scheduled_workflow(entrypoint: str, env: Mapping[str, Any] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"run": f"uv run python {entrypoint}"}
    if env is not None:
        step["env"] = dict(env)
    job = {"runs-on": "ubuntu-latest", "steps": [step]}
    triggers = {"schedule": [{"cron": "30 5 * * 1-5"}], "workflow_dispatch": None}
    return {True: triggers, "jobs": {"run": job}}


_AGENT_ENV = {"PAID_CALLS_ENABLED": "true", "DB_WRITES_ENABLED": "true"}


def _resume_ready_workflows() -> dict[str, dict[str, Any]]:
    return {
        FAST_LOOP_WORKFLOW: _scheduled_workflow("main_agent.py", _AGENT_ENV),
        "deadman_daily.yml": _scheduled_workflow("main_deadman.py"),
    }


def test_resume_guard_accepts_a_fully_prepared_resume() -> None:
    _assert_fast_loop_resume_conditions(_resume_ready_config(), _resume_ready_workflows())


def test_resume_guard_catches_a_missing_deadman() -> None:
    workflows = _resume_ready_workflows()
    del workflows["deadman_daily.yml"]

    with pytest.raises(AssertionError, match="main_deadman.py"):
        _assert_fast_loop_resume_conditions(_resume_ready_config(), workflows)


def test_resume_guard_catches_a_deadman_without_its_cron() -> None:
    workflows = _resume_ready_workflows()
    workflows["deadman_daily.yml"][True] = {"workflow_dispatch": None}

    with pytest.raises(AssertionError, match="main_deadman.py"):
        _assert_fast_loop_resume_conditions(_resume_ready_config(), workflows)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_paid_llm_calls_per_cycle": None}, "max_paid_llm_calls_per_cycle"),
        ({"fast_loop_weekly_budget_usd": None}, "fast_loop_weekly_budget_usd"),
        ({"liveness_fast_loop_max_gap_h": 0}, "liveness_fast_loop_max_gap_h"),
    ],
)
def test_resume_guard_catches_each_missing_config_condition(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        _assert_fast_loop_resume_conditions(
            _resume_ready_config(**overrides), _resume_ready_workflows()
        )


@pytest.mark.parametrize("switch", ["PAID_CALLS_ENABLED", "DB_WRITES_ENABLED"])
def test_resume_guard_catches_each_missing_workflow_switch(switch: str) -> None:
    workflows = _resume_ready_workflows()
    env = {key: value for key, value in _AGENT_ENV.items() if key != switch}
    workflows[FAST_LOOP_WORKFLOW] = _scheduled_workflow("main_agent.py", env)

    with pytest.raises(AssertionError, match=switch):
        _assert_fast_loop_resume_conditions(_resume_ready_config(), workflows)


def test_resume_guard_reads_the_real_deadman_schedule() -> None:
    # Parser crona musi widzieć prawdziwy `on: schedule` (klucz True w PyYAML) —
    # inaczej warunkowy test wyżej byłby pomijany także przy odkomentowanym cronie.
    assert _scheduled_deadmen(_all_workflow_specs()) == ["deadman_daily.yml"]


# CI-08: tag akcji (`@v4`) jest ruchomy — właściciel repo akcji przesuwa go na dowolny
# commit, także pod jobem z `contents: write`. Każde `uses:` wskazuje więc pełny SHA
# commitu, a komentarz `# vX.Y.Z` obok mówi człowiekowi i Dependabotowi, jaka to wersja.
# Linie, nie YAML: komentarz z wersją ginie przy parsowaniu.
_USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S.*?)\s*$")
_PINNED_USES = re.compile(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}\s+#\s*v\d+(?:\.\d+)*")


def _uses_lines(text: str) -> list[str]:
    """Wartości `uses:` (z komentarzem) każdej linii workflowu."""
    return [match.group(1) for line in text.splitlines() if (match := _USES_LINE.match(line))]


def _unpinned_actions(text: str) -> list[str]:
    """`uses:` bez pełnego SHA i komentarza wersji; akcje lokalne (`./`) są w repo."""
    return [
        uses
        for uses in _uses_lines(text)
        if not uses.startswith("./") and not _PINNED_USES.fullmatch(uses)
    ]


def _parsed_uses(spec: Mapping[str, Any]) -> list[str]:
    jobs: Mapping[str, Any] = spec.get("jobs") or {}
    return [str(job["uses"]) for job in jobs.values() if "uses" in job] + [
        str(step["uses"]) for step in _workflow_steps(spec) if "uses" in step
    ]


def test_every_action_is_pinned_by_sha() -> None:
    workflows_dir = ROOT / ".github" / "workflows"
    unpinned: dict[str, list[str]] = {}
    for path in sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]):
        text = path.read_text(encoding="utf-8")
        # Strażnik skanu linii: tyle samo `uses:` co w sparsowanym YAML-u.
        assert len(_uses_lines(text)) == len(_parsed_uses(yaml.safe_load(text))), path.name
        if found := _unpinned_actions(text):
            unpinned[path.name] = found

    assert not unpinned, (
        f"akcje bez pełnego SHA commitu i komentarza `# vX.Y.Z`: {unpinned} — "
        "SHA tagu: `gh api repos/<owner>/<repo>/git/ref/tags/<tag>`."
    )


@pytest.mark.parametrize(
    "uses",
    [
        "actions/checkout@v4",
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "actions/checkout@main # v4",
        "actions/checkout@11d5960 # v4.4.0",
    ],
)
def test_pin_guard_rejects_tags_short_shas_and_missing_versions(uses: str) -> None:
    assert _unpinned_actions(f"      - uses: {uses}\n") == [uses]


def test_pin_guard_accepts_a_full_sha_with_its_version_and_local_actions() -> None:
    text = (
        "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n"
        "        uses: ./.github/actions/setup\n"
    )

    assert _unpinned_actions(text) == []


# CI-09: `version: latest` instaluje przy każdym przebiegu inną binarkę, a Supabase CLI
# wykonuje DDL na produkcji. Podbicie wersji to świadomy commit — sprawdź wtedy, że
# parser nazw migracji (`^([0-9]+)_(.*)\.sql$`, CLAUDE.md) się nie zmienił.
def _setup_inputs(action: str) -> list[dict[str, Any]]:
    return [
        dict(step.get("with") or {})
        for spec in _all_workflow_specs().values()
        for step in _workflow_steps(spec)
        if str(step.get("uses", "")).startswith(f"{action}@")
    ]


def test_supabase_cli_version_is_pinned() -> None:
    inputs = _setup_inputs("supabase/setup-cli")

    assert inputs, "żaden workflow nie instaluje Supabase CLI"
    versions = [str(step_inputs.get("version", "")) for step_inputs in inputs]
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in versions), versions


def test_no_workflow_installs_a_tool_at_latest() -> None:
    latest = [
        f"{name}: {step.get('uses')}"
        for name, spec in _all_workflow_specs().items()
        for step in _workflow_steps(spec)
        if str(dict(step.get("with") or {}).get("version", "")).strip().lower() == "latest"
    ]

    assert latest == []


# WIA-14: ledger atestacji żyje na osieroconej, chronionej gałęzi `attestation-log`.
# Agent dopisuje do checkoutu tej gałęzi (bez checkoutu świeży runner zaczynałby
# pusty plik i zgubił historię), a push robi osobny job z `contents: write` — job z
# kluczami API nie dostaje prawa zapisu do repo. Flagi `attestation_enabled` i
# `public_scorecard_enabled` zostają off do kroków z docs/resume-checklist.md.
LEDGER_BRANCH = "attestation-log"
LEDGER_AGENT_JOB = "run-market-agent"
LEDGER_SKIP_CI = "[skip ci]"
# Sekrety, które wolno mieć jobowi z prawem zapisu: wyłącznie mail o jego porażce.
LEDGER_JOB_ALLOWED_SECRETS = {"RESEND_API_KEY", "DIGEST_TO_EMAIL"}


def _fast_loop_spec() -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _ledger_dir() -> str:
    """Katalog obu plików ledgera z config.toml — musi być jeden, bo to jeden checkout."""
    config = _repo_config()
    directories = {
        posixpath.dirname(_normalized(str(config[key]))) for key in ATTESTATION_LEDGER_KEYS
    }
    assert len(directories) == 1, f"pliki ledgera w różnych katalogach: {sorted(directories)}"
    return directories.pop()


def _jobs_pushing(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    jobs: Mapping[str, Any] = spec.get("jobs") or {}
    return {
        name: job
        for name, job in jobs.items()
        if any(_git_pushes(str(step.get("run", ""))) for step in job.get("steps", []))
    }


def _ledger_checkout(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            step
            for step in steps
            if str(step.get("uses", "")).startswith("actions/checkout")
            and dict(step.get("with") or {}).get("ref") == LEDGER_BRANCH
        ),
        None,
    )


def test_fast_loop_checks_out_the_ledger_branch_before_the_agent_runs() -> None:
    steps = _steps()
    checkout = _ledger_checkout(steps)
    agent = next(s for s in steps if str(s.get("run", "")).endswith("main_agent.py"))

    assert checkout is not None, f"Fast Loop nie robi checkoutu gałęzi `{LEDGER_BRANCH}`"
    assert steps.index(checkout) < steps.index(agent)
    assert dict(checkout["with"])["path"] == _ledger_dir()


def test_ledger_checkout_in_the_agent_job_leaves_no_push_credentials() -> None:
    checkout = _ledger_checkout(_steps())

    assert checkout is not None
    assert dict(checkout["with"]).get("persist-credentials") is False


def _ledger_job() -> tuple[str, dict[str, Any]]:
    """Jedyny job Fast Loopa z `git push`; jego brak to porażka asercji, nie wyjątek."""
    pushing = _jobs_pushing(_fast_loop_spec())
    assert len(pushing) == 1, f"push ledgera powinien być w jednym jobie, jest: {sorted(pushing)}"
    ((name, job),) = pushing.items()
    return name, job


def test_ledger_is_pushed_by_one_separate_job_that_waits_for_the_agent() -> None:
    name, job = _ledger_job()

    assert name != LEDGER_AGENT_JOB
    assert LEDGER_AGENT_JOB in _job_needs(job)
    assert not any("main_agent.py" in str(s.get("run", "")) for s in job["steps"])


def test_only_the_ledger_job_has_write_access_to_the_repo() -> None:
    spec = _fast_loop_spec()
    name, job = _ledger_job()
    jobs: Mapping[str, Any] = spec["jobs"]

    assert dict(job.get("permissions") or {}).get("contents") == "write"
    assert "write" not in dict(spec.get("permissions") or {}).values()
    others = {
        other: dict(body.get("permissions") or {}).get("contents")
        for other, body in jobs.items()
        if other != name
    }
    assert "write" not in others.values(), others


def test_the_ledger_job_holds_no_api_secret() -> None:
    _, job = _ledger_job()

    referenced = set(re.findall(r"secrets\.(\w+)", yaml.safe_dump(job)))

    assert referenced <= LEDGER_JOB_ALLOWED_SECRETS, referenced - LEDGER_JOB_ALLOWED_SECRETS


def _ledger_publish_step() -> dict[str, Any]:
    _, job = _ledger_job()
    return next(s for s in job["steps"] if _git_pushes(str(s.get("run", ""))))


def test_ledger_push_goes_only_to_the_ledger_branch_never_to_main() -> None:
    for name, spec in _all_workflow_specs().items():
        for job_name, job in _jobs_pushing(spec).items():
            for step in job["steps"]:
                for _, subcommand, args in _git_commands(str(step.get("run", ""))):
                    if subcommand != "push":
                        continue
                    refspecs = [arg for arg in args if not arg.startswith("-")]
                    assert refspecs, f"{name}:{job_name}: `git push` bez jawnego refspeca"
                    assert all(
                        ref.split(":")[-1].removeprefix("refs/heads/") == LEDGER_BRANCH
                        for ref in refspecs[1:]
                    ), f"{name}:{job_name}: push poza `{LEDGER_BRANCH}`: {refspecs}"


def test_ledger_commit_skips_ci_and_the_push_rebases_and_retries() -> None:
    script = str(_ledger_publish_step()["run"])
    commands = _git_commands(script)
    commits = [args for _, sub, args in commands if sub == "commit"]

    assert commits and all(LEDGER_SKIP_CI in " ".join(args) for args in commits)
    assert any(sub == "pull" and "--rebase" in args for _, sub, args in commands), (
        "push ledgera bez `git pull --rebase`"
    )
    assert re.search(r"\b(for|while|until)\b", script), "push ledgera bez pętli ponowień"


def test_the_ledger_job_runs_even_when_the_agent_job_failed() -> None:
    # Predykcje zapisane przed awarią (exit 1, gdy padły wszystkie symbole) mają
    # już commitment w bazie, więc ich hash musi trafić do ledgera mimo porażki joba.
    _, job = _ledger_job()

    assert "!cancelled()" in str(job.get("if", ""))


def test_ledger_artifact_is_uploaded_even_when_the_agent_step_failed() -> None:
    uploads = [
        step
        for step in _steps()
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
        and "ledger" in str(dict(step.get("with") or {}).get("name", ""))
    ]

    assert len(uploads) == 1, "agent nie wysyła artefaktu ledgera do jobu z pushem"
    assert "!cancelled()" in str(uploads[0].get("if", ""))


def test_ledger_artifact_carries_both_ledger_files_from_config() -> None:
    upload = next(
        step
        for step in _steps()
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
        and "ledger" in str(dict(step.get("with") or {}).get("name", ""))
    )
    config = _repo_config()

    uploaded = {
        _normalized(line)
        for line in str(dict(upload["with"])["path"]).splitlines()
        if line.strip()
    }

    assert uploaded == {_normalized(str(config[key])) for key in ATTESTATION_LEDGER_KEYS}


def test_real_fast_loop_publishes_the_attestation_ledger_once_its_flag_is_on() -> None:
    steps = _agent_workflows()[FAST_LOOP_WORKFLOW]

    assert _missing_publications(_repo_config(**ATTESTATION_ONLY), steps) == []


# Skrypt kroku pusha uruchamiamy naprawdę, na lokalnym "origin" w katalogu tymczasowym:
# to jedyny dowód, że dopisanie nie gubi linii innego przebiegu i że ponowienia działają.
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **_GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class _LedgerRemote:
    """Bare `origin` z gałęzią `attestation-log` i checkout w układzie runnera."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.origin = root / "origin.git"
        self.workspace = root / "workspace"
        self.checkout = self.workspace / "public" / "attestation"
        branch = f"--initial-branch={LEDGER_BRANCH}"
        _git(root, "init", "--quiet", "--bare", branch, str(self.origin))
        seed = root / "seed"
        _git(root, "clone", "--quiet", str(self.origin), str(seed))
        _git(seed, "checkout", "--quiet", "-b", LEDGER_BRANCH)
        (seed / "commitments.jsonl").write_text('{"n": 1}\n', encoding="utf-8")
        (seed / "reveals.jsonl").write_text("", encoding="utf-8")
        _git(seed, "add", ".")
        _git(seed, "commit", "--quiet", "-m", "ledger: init [skip ci]")
        _git(seed, "push", "--quiet", "origin", LEDGER_BRANCH)
        self.seed = seed
        self.workspace.mkdir()
        (self.workspace / "public").mkdir()
        _git(
            root,
            *("clone", "--quiet", "--branch", LEDGER_BRANCH, str(self.origin), str(self.checkout)),
        )
        self.base = _git(self.checkout, "rev-parse", "HEAD")

    def stage_artifact(self, commitments: str, reveals: str = "") -> None:
        artifact = self.workspace / "ledger-artifact"
        artifact.mkdir(exist_ok=True)
        (artifact / "commitments.jsonl").write_text(commitments, encoding="utf-8")
        (artifact / "reveals.jsonl").write_text(reveals, encoding="utf-8")

    def push_from_another_run(self, line: str) -> None:
        with (self.seed / "commitments.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)
        _git(self.seed, "commit", "--quiet", "-am", "ledger: other run [skip ci]")
        _git(self.seed, "push", "--quiet", "origin", LEDGER_BRANCH)

    def run_publish_step(self, **env: str) -> subprocess.CompletedProcess[str]:
        script = str(_ledger_publish_step()["run"])
        return subprocess.run(
            ["bash", "-c", script],
            cwd=self.workspace,
            env={
                **os.environ,
                **_GIT_ENV,
                "LEDGER_BASE": self.base,
                "GITHUB_RUN_ID": "4242",
                "LEDGER_RETRY_DELAY_SECONDS": "0",
                **env,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def published(self, name: str = "commitments.jsonl") -> list[str]:
        blob = f"{LEDGER_BRANCH}:{name}"
        return _git(self.root, f"--git-dir={self.origin}", "show", blob).splitlines()

    def history(self) -> list[str]:
        return _git(
            self.root, f"--git-dir={self.origin}", "log", "--format=%s", LEDGER_BRANCH
        ).splitlines()


@pytest.fixture
def ledger_remote(tmp_path: Path) -> _LedgerRemote:
    return _LedgerRemote(tmp_path)


def test_publish_step_appends_the_new_lines_in_a_skip_ci_commit(
    ledger_remote: _LedgerRemote,
) -> None:
    ledger_remote.stage_artifact('{"n": 1}\n{"n": 2}\n')

    result = ledger_remote.run_publish_step()

    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger_remote.published() == ['{"n": 1}', '{"n": 2}']
    assert ledger_remote.history() == ["ledger: run 4242 [skip ci]", "ledger: init [skip ci]"]


def test_publish_step_keeps_lines_another_run_pushed_in_the_meantime(
    ledger_remote: _LedgerRemote,
) -> None:
    ledger_remote.stage_artifact('{"n": 1}\n{"n": 2}\n')
    ledger_remote.push_from_another_run('{"n": 3}\n')

    result = ledger_remote.run_publish_step()

    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(ledger_remote.published()) == ['{"n": 1}', '{"n": 2}', '{"n": 3}']
    assert ledger_remote.history()[-1] == "ledger: init [skip ci]"
    assert len(ledger_remote.history()) == 3


def test_publish_step_pushes_nothing_when_the_ledger_did_not_change(
    ledger_remote: _LedgerRemote,
) -> None:
    ledger_remote.stage_artifact('{"n": 1}\n')

    result = ledger_remote.run_publish_step()

    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger_remote.history() == ["ledger: init [skip ci]"]


def _reject_first_pushes(remote: _LedgerRemote, count: int) -> None:
    hook = remote.origin / "hooks" / "pre-receive"
    counter = remote.root / "rejected"
    hook.write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
        f'if [ "$n" -lt {count} ]; then echo $((n + 1)) > "{counter}"; exit 1; fi\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)


def test_publish_step_retries_a_rejected_push(ledger_remote: _LedgerRemote) -> None:
    ledger_remote.stage_artifact('{"n": 1}\n{"n": 2}\n')
    _reject_first_pushes(ledger_remote, count=2)

    result = ledger_remote.run_publish_step()

    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger_remote.published() == ['{"n": 1}', '{"n": 2}']


def test_publish_step_fails_loudly_when_every_push_is_rejected(
    ledger_remote: _LedgerRemote,
) -> None:
    ledger_remote.stage_artifact('{"n": 1}\n{"n": 2}\n')
    _reject_first_pushes(ledger_remote, count=99)

    result = ledger_remote.run_publish_step()

    assert result.returncode != 0
    assert "::error" in result.stdout
    assert ledger_remote.published() == ['{"n": 1}']
