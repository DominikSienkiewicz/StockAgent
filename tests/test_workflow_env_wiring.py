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


def _agent_workflows() -> dict[str, list[dict[str, object]]]:
    """Kroki wszystkich jobów, które uruchamiają `main_agent.py` (per workflow)."""
    found: dict[str, list[dict[str, object]]] = {}
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in spec.get("jobs", {}).values():
            steps = job.get("steps", [])
            if any("main_agent.py" in str(s.get("run", "")) for s in steps):
                found.setdefault(path.name, []).extend(steps)
    return found


def _has_durable_publish_step(steps: list[dict[str, object]]) -> bool:
    return any(
        str(s.get("uses", "")).startswith("actions/deploy-pages")
        or "git push" in str(s.get("run", ""))
        for s in steps
    )


def test_enabled_public_artifacts_have_a_durable_publish_step() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    enabled = [flag for flag in PUBLIC_ARTIFACT_FLAGS if config.get(flag) is True]
    workflows = _agent_workflows()
    assert workflows, "żaden workflow nie uruchamia main_agent.py"

    unpublished = [
        name for name, steps in workflows.items() if not _has_durable_publish_step(steps)
    ]
    assert not (enabled and unpublished), (
        f"config.toml włącza {enabled}, ale {unpublished} nie ma kroku "
        "`actions/deploy-pages` ani `git push` — artefakt ginie z dyskiem runnera."
    )


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


def _git_added_paths(script: str) -> list[str]:
    """Ścieżki każdego `git add` w skrypcie kroku; zgarnięcie całego drzewa jako "."."""
    added: list[str] = []
    for command in re.split(r"&&|\|\||[;\n]", script):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        starts = [i for i in range(len(tokens) - 1) if tokens[i : i + 2] == ["git", "add"]]
        for start in starts:
            for arg in tokens[start + 2 :]:
                if arg in GIT_ADD_SWEEP_FLAGS:
                    added.append(".")
                elif not arg.startswith("-"):
                    added.append(_normalized(arg))
    return added


def _is_pages_deploy(step: Mapping[str, Any]) -> bool:
    return str(step.get("uses", "")).startswith("actions/deploy-pages")


def _pages_upload_path(step: Mapping[str, Any]) -> str | None:
    if not str(step.get("uses", "")).startswith("actions/upload-pages-artifact"):
        return None
    inputs = step.get("with") or {}
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


def test_real_workflows_never_publish_private_digest() -> None:
    workflows = _all_workflow_specs()
    assert {"fast_loop_12h.yml", "slow_loop_weekly.yml"} <= set(workflows)

    violations = _private_digest_publication_violations(_repo_config(), workflows)

    assert not violations, violations


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
    commit_step = next(s for s in steps if "git push" in str(s.get("run", "")))
    assert str(commit_step.get("if", "")).replace(" ", "") == "${{!cancelled()}}"


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

    assert {"fast_loop_12h.yml", "slow_loop_weekly.yml", "shock_watch_hourly.yml"} <= workflows


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
