"""Strażnik read-only dla `scripts/prod-state-check.sh` (NIEZ-47).

Skrypt czyta produkcję: GitHub przez `gh` i Supabase kluczem aplikacji. Każdy zapis
zmieniałby stan, który skrypt ma tylko zmierzyć, a wypisany klucz trafiłby do
pliku z wynikiem. Polityka jest więc sprawdzana statycznie na treści skryptu,
bez sieci i bez kluczy.

Tu też polityka workflowów bez kodu aplikacji: ręczny smoke testów na żywym API
(CI-05), grupy Dependabota (CI-07), wspólna kolejka zapisów do bazy (NIEZ-27)
oraz strażnik polityk każdego workflowu (CI-10, CI-25).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "prod-state-check.sh"

# (wzorzec, dlaczego zakazany) — każdy oznacza zapis albo wyciek sekretu.
FORBIDDEN = (
    (r"\bgh\s+(secret|variable)\s+(set|remove)\b", "zapis sekretów albo zmiennych"),
    (r"\bgh\s+workflow\s+(enable|disable|run)\b", "zmiana stanu workflowu"),
    (r"\bgh\s+run\s+(rerun|cancel)\b", "sterowanie przebiegiem"),
    (
        r"\bgh\s+api\b[^\n]*(\s-X\b|--method\b|\s-[fF]\b|--field\b|--raw-field\b|--input\b)",
        "gh api z metodą albo ciałem żądania",
    ),
    (r"rest/v1/rpc", "wywołanie funkcji bazy przez RPC"),
    (r"(?i)\b(insert|update|delete|alter|drop|truncate|grant|revoke|create)\b", "zapis SQL"),
    (r"\bset\s+-[a-z]*x|xtrace", "śledzenie powłoki wypisałoby klucze"),
)


def _code_lines() -> list[str]:
    return [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_prod_state_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


@pytest.mark.parametrize("pattern, reason", FORBIDDEN)
def test_prod_state_script_is_read_only(pattern: str, reason: str) -> None:
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    found = [line for line in lines if re.search(pattern, line)]
    assert not found, f"{reason}: {found}"


def test_prod_state_script_requests_are_get_only() -> None:
    calls = [line for line in _code_lines() if re.search(r"\bcurl\s+-", line)]
    assert calls, "skrypt nie robi żadnego żądania do Supabase"
    for call in calls:
        assert "--get" in call, call
        assert "-K -" in call, f"klucz ma iść przez stdin, nie przez argv: {call}"
        assert not re.search(r"\s(-X|--request|-d|--data\S*)\s", call), call


def test_prod_state_script_never_prints_the_key() -> None:
    for line in _code_lines():
        if re.search(r"\b(echo|printf)\b", line) and re.search(r"\$\{?SB_KEY\b", line):
            assert "| curl -K -" in line, f"klucz wypisany poza konfiguracją curl: {line}"


# CI-05: testy `-m integration` (prawdziwe API, ~5 płatnych wywołań i jeden mail)
# nie biegły w żadnym workflowie. Pełny smoke jest WYŁĄCZNIE ręczny (decyzja właściciela):
# bez crona, push i PR-ów, z osobnymi sekretami `SMOKE_*`, żeby nie zjadał limitów
# kluczy produkcyjnych i nie mailował na adres digestu. TEST-16: cron uruchamia osobny
# job z samym testem kształtu dostawców (tests/test_integration_smoke_schedule.py).
SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "integration_smoke.yml"


def _smoke_spec() -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    return spec


def _pytest_marker_expression(command: str) -> str | None:
    """Wyrażenie `-m` komendy pytest ('' bez `-m`), None gdy to nie pytest."""
    tokens = shlex.split(command.split("#", 1)[0])
    if "pytest" not in tokens:
        return None
    args = tokens[tokens.index("pytest") + 1 :]
    return args[args.index("-m") + 1] if "-m" in args[:-1] else ""


def test_integration_smoke_runs_only_integration_marker() -> None:
    spec: dict[Any, Any] = _smoke_spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True))
    steps = [step for job in spec["jobs"].values() for step in job.get("steps", [])]
    markers = [
        marker
        for step in steps
        if (marker := _pytest_marker_expression(str(step.get("run", "")))) is not None
    ]

    assert isinstance(triggers, dict), triggers
    assert set(triggers) == {"workflow_dispatch", "schedule"}, triggers
    assert markers == ["integration", "integration"], markers


def test_integration_smoke_uses_only_its_own_secrets() -> None:
    secrets = set(
        re.findall(r"secrets\.([A-Za-z0-9_]+)", SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    )

    assert secrets, "smoke bez sekretów pominie każdy test na żywym API"
    assert all(name.startswith("SMOKE_") for name in secrets), sorted(secrets)


# CI-07: SHA akcji (CI-08) nikt nie podbijał, a zależności nie były kuratorowane.
# Dependabot co tydzień, jedna grupa na ekosystem — bez grup zalałby repo PR-ami.
DEPENDABOT = ROOT / ".github" / "dependabot.yml"


def test_dependabot_groups_weekly_updates_for_actions_and_uv() -> None:
    spec = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    updates = {update["package-ecosystem"]: update for update in spec["updates"]}

    assert spec["version"] == 2
    assert set(updates) == {"github-actions", "uv"}
    for ecosystem, update in updates.items():
        assert update["directory"] == "/", ecosystem
        assert update["schedule"]["interval"] == "weekly", ecosystem
        groups = update.get("groups") or {}
        assert any(group.get("patterns") == ["*"] for group in groups.values()), (
            f"{ecosystem}: bez grupy `*` każda zależność dostaje osobny PR"
        )


# NIEZ-27: ręczne DB Migrate w trakcie cyklu (022 robi DROP i CREATE ml_feature_store)
# dawało błędy PGRST w połowie cyklu albo kolizję z odświeżeniem feature store w Slow
# Loopie. Każdy job piszący do bazy bierze tę samą grupę, a kolejka nie anuluje nikogo.
#
# Grupa jest na poziomie JOBA, nie workflowu. GitHub trzyma grupę workflowu przez cały
# run, a grupę joba tylko przez ten job, i oba poziomy działają niezależnie — więc:
# - joby bez zapisu do bazy (push ledgera, deploy Pages) nie blokują innych piszących;
# - grupa workflowu zostaje własna (`fast-loop`, `slow-loop`, `db-migrate`): serializuje
#   kolejne runy tego samego workflowu, także ich joby spoza `supabase-writers`;
# - ta sama nazwa na obu poziomach jednego runu zakleszczyłaby go (job czekałby na
#   grupę, którą trzyma jego własny workflow), więc nazwy obu poziomów są rozłączne.
SUPABASE_WRITERS_GROUP = "supabase-writers"
# Shock watch pisze `shock_alerts`, ale chodzi co godzinę i jest darmowy. GitHub trzyma
# w grupie jeden oczekujący job i anuluje starszego oczekującego, więc godzinny watch
# stojący w tej kolejce wypychałby z niej dzienny Fast Loop albo tygodniowy Slow Loop.
# Bot właściciela (MAIL-48) pisze offset i wyciszenia co 15 minut — ten sam powód.
GROUP_EXEMPT_WRITERS = frozenset({"shock_watch_hourly.yml", "bot_poll.yml"})
EXPECTED_GROUP_WRITERS = frozenset(
    {
        ("fast_loop_12h.yml", "run-market-agent"),
        ("slow_loop_weekly.yml", "retrain-predictive-model"),
        ("db_migrate.yml", "migrate"),
    }
)

WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def _workflow_specs() -> dict[str, dict[str, Any]]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted([*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")])
    }


def _jobs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = spec.get("jobs") or {}
    return jobs


def _concurrency(node: dict[str, Any]) -> tuple[str | None, Any]:
    """(grupa, cancel-in-progress) z `concurrency:` — mapa albo skrót `concurrency: grupa`."""
    declared = node.get("concurrency")
    if declared is None:
        return None, None
    if isinstance(declared, str):
        return declared, False
    return declared.get("group"), declared.get("cancel-in-progress", False)


def _step_envs(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(step.get("env") or {}) for step in job.get("steps", [])]


def _writes_to_database(job: dict[str, Any]) -> bool:
    """Job z `DB_WRITES_ENABLED: "true"` (BEZ-08) albo z connection stringiem migracji."""
    return any(
        env.get("DB_WRITES_ENABLED") == "true" or "SUPABASE_DB_URL" in env
        for env in _step_envs(job)
    )


def _database_writing_jobs() -> set[tuple[str, str]]:
    return {
        (workflow, name)
        for workflow, spec in _workflow_specs().items()
        for name, job in _jobs(spec).items()
        if _writes_to_database(job)
    }


def test_db_writing_workflows_share_concurrency_group() -> None:
    writers = {pair for pair in _database_writing_jobs() if pair[0] not in GROUP_EXEMPT_WRITERS}

    assert writers == EXPECTED_GROUP_WRITERS, (
        "nowy albo zniknięty job piszący do bazy: dopisz go do EXPECTED_GROUP_WRITERS "
        "albo do GROUP_EXEMPT_WRITERS z uzasadnieniem"
    )
    for workflow, name in sorted(writers):
        job = _jobs(_workflow_specs()[workflow])[name]
        assert _concurrency(job) == (SUPABASE_WRITERS_GROUP, False), (
            f"{workflow}:{name} pisze do bazy, ale nie czeka w grupie "
            f"{SUPABASE_WRITERS_GROUP} z cancel-in-progress: false"
        )


def test_job_concurrency_groups_never_reuse_a_workflow_group() -> None:
    specs = _workflow_specs()
    workflow_groups = {_concurrency(spec)[0] for spec in specs.values()} - {None}
    job_groups = {
        _concurrency(job)[0]
        for spec in specs.values()
        for job in _jobs(spec).values()
        if _concurrency(job)[0] is not None
    }

    assert job_groups, "żaden job nie deklaruje grupy — test nic by nie pilnował"
    assert workflow_groups & job_groups == set(), (
        "ta sama grupa na poziomie workflowu i joba zakleszcza run: "
        f"{sorted(workflow_groups & job_groups, key=str)}"
    )


def test_hourly_shock_watch_stays_out_of_the_writers_queue() -> None:
    spec = _workflow_specs()["shock_watch_hourly.yml"]
    groups = [_concurrency(spec)[0], *(_concurrency(job)[0] for job in _jobs(spec).values())]

    assert SUPABASE_WRITERS_GROUP not in groups


# CI-10: polityki workflowów były dotrzymane ręcznie i nierówno. Test parsuje każdy plik
# z `.github/workflows/`, więc nowy workflow podlega im od pierwszego commitu.
def _triggers(spec: dict[Any, Any]) -> dict[str, Any]:
    """Wyzwalacze workflowu; PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True."""
    triggers = spec.get("on", spec.get(True)) or {}
    return triggers if isinstance(triggers, dict) else {}


def _job_ids() -> list[tuple[str, str]]:
    return [
        (workflow, name) for workflow, spec in _workflow_specs().items() for name in _jobs(spec)
    ]


def _is_scheduled(spec: dict[str, Any]) -> bool:
    return bool(_triggers(spec).get("schedule"))


@pytest.mark.parametrize("workflow", sorted(_workflow_specs()))
def test_every_workflow_declares_top_level_permissions(workflow: str) -> None:
    permissions = _workflow_specs()[workflow].get("permissions")

    assert isinstance(permissions, dict) and permissions, (
        f"{workflow}: bez `permissions:` na poziomie workflowu GITHUB_TOKEN dostaje "
        "domyślne uprawnienia repozytorium"
    )


@pytest.mark.parametrize(("workflow", "name"), _job_ids())
def test_every_job_has_a_timeout(workflow: str, name: str) -> None:
    job = _jobs(_workflow_specs()[workflow])[name]

    assert isinstance(job.get("timeout-minutes"), int | float), (
        f"{workflow}:{name} bez `timeout-minutes` — zawieszony job zjada minuty do limitu 6 h"
    )


def _needs_concurrency() -> list[str]:
    return [
        workflow
        for workflow, spec in _workflow_specs().items()
        if _is_scheduled(spec) or any(_writes_to_database(job) for job in _jobs(spec).values())
    ]


def test_the_concurrency_rule_covers_the_loops_and_the_migration() -> None:
    assert set(_needs_concurrency()) >= {
        "fast_loop_12h.yml",
        "slow_loop_weekly.yml",
        "shock_watch_hourly.yml",
        "deadman_daily.yml",
        "db_migrate.yml",
    }


@pytest.mark.parametrize("workflow", _needs_concurrency())
def test_scheduled_and_database_writing_workflows_have_concurrency(workflow: str) -> None:
    group, _ = _concurrency(_workflow_specs()[workflow])

    assert group, (
        f"{workflow} ma cron albo pisze do bazy, a nie deklaruje `concurrency:` na poziomie "
        "workflowu — dwa przebiegi nałożyłyby się"
    )


# Token checkoutu zostaje w `.git/config` na cały job; job instalujący ~150 zależności
# nie ma po co go widzieć, jeśli nic nie pushuje. Push jest tam, gdzie `git ... push`.
_GIT_PUSH = re.compile(r"\bgit\b[^\n]*\bpush\b")


def _pushes_with_git(job: dict[str, Any]) -> bool:
    return any(_GIT_PUSH.search(str(step.get("run", ""))) for step in job.get("steps", []))


def _checkouts(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]


def _pushing_jobs() -> list[tuple[str, str]]:
    return [
        (workflow, name)
        for workflow, name in _job_ids()
        if _pushes_with_git(_jobs(_workflow_specs()[workflow])[name])
    ]


def test_the_checkout_rule_finds_the_one_pushing_job() -> None:
    assert _pushing_jobs() == [("fast_loop_12h.yml", "publish-attestation-ledger")]


@pytest.mark.parametrize(
    ("workflow", "name"), [pair for pair in _job_ids() if pair not in _pushing_jobs()]
)
def test_non_pushing_checkouts_do_not_persist_credentials(workflow: str, name: str) -> None:
    job = _jobs(_workflow_specs()[workflow])[name]

    for checkout in _checkouts(job):
        assert dict(checkout.get("with") or {}).get("persist-credentials") is False, (
            f"{workflow}:{name}: checkout bez `persist-credentials: false` zostawia token "
            "w .git/config joba, który niczego nie pushuje"
        )


# Anulowanie zabija run, który już zapłacił za wywołanie (G81, FIN-18). Job „z płatnym
# API" rozpoznajemy po tym, co faktycznie dostaje: włączone płatne wywołania albo sekret
# LLM / Alpha Vantage (także `SMOKE_*`). Lista nie jest utrzymywana ręcznie; test niżej
# pilnuje tylko, żeby rozpoznanie nie przestało łapać znanych jobów.
_PAID_SECRET = re.compile(
    r"secrets\.(?:SMOKE_)?(?:OPENAI_API_KEY|ANTHROPIC_API_KEY|ALPHA_VANTAGE_API_KEYS?)\b"
)
EXPECTED_PAID_JOBS = frozenset(
    {
        ("fast_loop_12h.yml", "run-market-agent"),
        ("slow_loop_weekly.yml", "retrain-predictive-model"),
        ("integration_smoke.yml", "live-api-smoke"),
        # TEST-16: cotygodniowy test kształtu czyta Alpha Vantage kluczem SMOKE_*.
        ("integration_smoke.yml", "provider-shapes"),
    }
)


def _uses_paid_api(job: dict[str, Any]) -> bool:
    envs = _step_envs(job)
    return any(env.get("PAID_CALLS_ENABLED") == "true" for env in envs) or any(
        _PAID_SECRET.search(str(value)) for env in envs for value in env.values()
    )


def _paid_jobs() -> list[tuple[str, str]]:
    return [
        (workflow, name)
        for workflow, name in _job_ids()
        if _uses_paid_api(_jobs(_workflow_specs()[workflow])[name])
    ]


def test_paid_job_detection_finds_the_known_paid_jobs() -> None:
    assert set(_paid_jobs()) == EXPECTED_PAID_JOBS


@pytest.mark.parametrize(("workflow", "name"), _paid_jobs())
def test_paid_jobs_never_cancel_a_run_in_progress(workflow: str, name: str) -> None:
    spec = _workflow_specs()[workflow]
    levels = {
        "workflow": _concurrency(spec)[1],
        "job": _concurrency(_jobs(spec)[name])[1],
    }

    cancelling = {level: value for level, value in levels.items() if value not in (None, False)}
    assert not cancelling, (
        f"{workflow}:{name} woła płatne API, a `cancel-in-progress` nie jest false "
        f"({cancelling}) — anulowanie zabiłoby run, który już zapłacił"
    )


# CI-25: bez `version` setup-uv instaluje najnowsze uv przy każdym runie, więc wynik
# `--frozen`, format locka i flagi zależą od daty uruchomienia. Wersja jest jawna w
# każdym kroku i taka sama we wszystkich workflowach. `[tool.uv] required-version` w
# pyproject odpada: zatrzymałoby też lokalne uv każdego dewelopera.
# Dependabot nie podbija parametru `version`: podbijaj ręcznie, wszędzie naraz.
_EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+")
RUNNING_LOCALLY = ROOT / "docs" / "running-locally.md"


def _setup_uv_steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        (workflow, dict(step.get("with") or {}))
        for workflow, name in _job_ids()
        for step in _jobs(_workflow_specs()[workflow])[name].get("steps", [])
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]


def test_setup_uv_pins_uv_version() -> None:
    steps = _setup_uv_steps()

    assert steps, "żaden workflow nie instaluje uv — test niczego nie pilnuje"
    unpinned = [
        workflow
        for workflow, inputs in steps
        if not _EXACT_VERSION.fullmatch(str(inputs.get("version", "")))
    ]
    assert unpinned == [], (
        f'setup-uv bez jawnej `version: "x.y.z"` w {unpinned} — instaluje najnowsze uv'
    )


def test_setup_uv_steps_share_one_uv_version() -> None:
    versions = {str(inputs.get("version", "")) for _, inputs in _setup_uv_steps()}

    assert len(versions) == 1, f"różne wersje uv w workflowach: {sorted(versions)}"
    assert all(_EXACT_VERSION.fullmatch(version) for version in versions), sorted(versions)


def test_docs_name_the_pinned_uv_version() -> None:
    versions = {str(inputs.get("version", "")) for _, inputs in _setup_uv_steps()}

    assert versions
    for version in versions:
        assert _EXACT_VERSION.fullmatch(version)
        assert f"`{version}`" in RUNNING_LOCALLY.read_text(encoding="utf-8"), (
            f"docs/running-locally.md nie podaje przypiętej wersji uv {version}"
        )


# BEZ-20: job z kluczami płatnych API wykonuje kod z uv.lock, więc jego pierwszy krok to
# harden-runner w trybie `audit` (inwentarz hostów, bez blokowania). Przejście na `block`
# jest osobną decyzją właściciela, po zebraniu inwentarza (docs/running-locally.md). Przypięcie
# po SHA z komentarzem wersji pilnuje już test_every_action_is_pinned_by_sha (CI-08).
HARDEN_RUNNER = "step-security/harden-runner"
HARDENED_LOOP_WORKFLOWS = frozenset(
    {"fast_loop_12h.yml", "slow_loop_weekly.yml", "shock_watch_hourly.yml"}
)
EXPECTED_HARDENED_JOBS = frozenset(
    {
        ("fast_loop_12h.yml", "run-market-agent"),
        ("slow_loop_weekly.yml", "retrain-predictive-model"),
        ("shock_watch_hourly.yml", "watch-market-shock"),
    }
)


def _runs_project_code_with_secrets(job: dict[str, Any]) -> bool:
    """Krok `uv run` z sekretem w env — kod z lockfile'a widzi klucze."""
    return any(
        "uv run" in str(step.get("run", ""))
        and any("secrets." in str(value) for value in dict(step.get("env") or {}).values())
        for step in job.get("steps", [])
    )


def _secret_bearing_loop_jobs() -> list[tuple[str, str]]:
    return [
        (workflow, name)
        for workflow, name in _job_ids()
        if workflow in HARDENED_LOOP_WORKFLOWS
        and _runs_project_code_with_secrets(_jobs(_workflow_specs()[workflow])[name])
    ]


def test_secret_bearing_job_detection_finds_the_three_loop_jobs() -> None:
    assert set(_secret_bearing_loop_jobs()) == EXPECTED_HARDENED_JOBS


@pytest.mark.parametrize(("workflow", "name"), sorted(EXPECTED_HARDENED_JOBS))
def test_secret_bearing_jobs_start_with_harden_runner(workflow: str, name: str) -> None:
    steps = _jobs(_workflow_specs()[workflow])[name].get("steps", [])
    first = steps[0] if steps else {}

    assert str(first.get("uses", "")).startswith(f"{HARDEN_RUNNER}@"), (
        f"{workflow}:{name} nie zaczyna się od {HARDEN_RUNNER} — hosty, do których "
        "wychodzi job z kluczami, nie trafiają do inwentarza"
    )
    assert dict(first.get("with") or {}).get("egress-policy") == "audit"


def test_harden_runner_never_blocks_traffic() -> None:
    audited = [
        dict(step.get("with") or {})
        for workflow, name in _job_ids()
        for step in _jobs(_workflow_specs()[workflow])[name].get("steps", [])
        if str(step.get("uses", "")).startswith(f"{HARDEN_RUNNER}@")
    ]

    assert len(audited) >= len(EXPECTED_HARDENED_JOBS), "żaden harden-runner — nic do pilnowania"
    for inputs in audited:
        assert inputs.get("egress-policy") == "audit", inputs
        assert "allowed-endpoints" not in inputs, "allowlista należy do trybu block"


# CI-18: job z entry-pointem (`python main_*.py`) i jego sekretami nie potrzebuje pytesta,
# mypy, ruffa ani testcontainers, więc kod z grupy dev nie trafia do środowiska z kluczami.
# Bramka (ci.yml) i joby, które same uruchamiają pytest (smoke, mutmut TEST-35), synchronizują
# grupę dev, bo jej używają.
_ENTRYPOINT_RUN = re.compile(r"\bpython\s+main_\w+\.py\b")
_RUNS_PYTEST = re.compile(r"\bpytest\b")
EXPECTED_ENTRYPOINT_WORKFLOWS = frozenset(
    {
        "bot_poll.yml",
        "deadman_daily.yml",
        "fast_loop_12h.yml",
        "scorecard_pages.yml",
        "shock_watch_hourly.yml",
        "slow_loop_weekly.yml",
    }
)


def _entrypoint_sync_steps() -> list[tuple[str, str, str]]:
    """(workflow, job, komenda) każdego `uv sync` w jobie, który uruchamia `main_*.py`
    i nie uruchamia pytesta."""
    found: list[tuple[str, str, str]] = []
    for workflow, spec in _workflow_specs().items():
        if workflow == "ci.yml":
            continue
        for name, job in _jobs(spec).items():
            runs = [str(step.get("run", "")) for step in job.get("steps", [])]
            if any(_RUNS_PYTEST.search(run) for run in runs):
                continue
            if any(_ENTRYPOINT_RUN.search(run) for run in runs):
                found.extend((workflow, name, run) for run in runs if "uv sync" in run)
    return found


def test_entrypoint_sync_detection_finds_every_entrypoint_workflow() -> None:
    assert {workflow for workflow, _, _ in _entrypoint_sync_steps()} == (
        EXPECTED_ENTRYPOINT_WORKFLOWS
    )


def test_production_workflows_sync_without_dev_dependencies() -> None:
    with_dev = [
        f"{workflow}:{job}"
        for workflow, job, run in _entrypoint_sync_steps()
        if "--no-dev" not in run.split()
    ]

    assert with_dev == [], (
        f"`uv sync` bez `--no-dev` w jobach z entry-pointem: {with_dev} — grupa dev "
        "(pytest, mypy, testcontainers) nie ma czego szukać w jobie z sekretami"
    )


# BEZ-22: connection string z prawem DDL na produkcji był sekretem repo, czytelnym dla każdego
# workflowu z main. Job migracji biegnie w środowisku `production-db` (wymagane zatwierdzenie
# właściciela, audyt w UI), a sekret żyje wyłącznie w tym środowisku. Środowisko, którego
# jeszcze nie ma po stronie GitHuba, powstaje przy pierwszym runie bez reguł ochrony, a sekret
# repo o tej samej nazwie dalej działa — workflow nie pada przed krokami właściciela.
MIGRATION_ENVIRONMENT = "production-db"
ENVIRONMENT_SECRETS = {"SUPABASE_DB_URL": MIGRATION_ENVIRONMENT}
SECRET_SCRIPTS = ("gh-secrets-check.sh", "gh-secrets-sync.sh")


def _job_environment(job: dict[str, Any]) -> str | None:
    """Nazwa środowiska joba — `environment: x` albo `environment: {name: x}`."""
    declared = job.get("environment")
    if isinstance(declared, dict):
        return str(declared.get("name"))
    return None if declared is None else str(declared)


def _jobs_reading_secret(secret: str) -> list[tuple[str, str]]:
    return [
        (workflow, name)
        for workflow, name in _job_ids()
        if f"secrets.{secret}" in yaml.safe_dump(_jobs(_workflow_specs()[workflow])[name])
    ]


def test_db_migrate_runs_in_protected_environment() -> None:
    migrate = _jobs(_workflow_specs()["db_migrate.yml"])["migrate"]

    assert _job_environment(migrate) == MIGRATION_ENVIRONMENT


@pytest.mark.parametrize(("secret", "environment"), sorted(ENVIRONMENT_SECRETS.items()))
def test_environment_secrets_are_read_only_inside_their_environment(
    secret: str, environment: str
) -> None:
    readers = _jobs_reading_secret(secret)

    assert readers, f"żaden job nie czyta {secret} — mapa ENVIRONMENT_SECRETS jest martwa"
    outside = [
        f"{workflow}:{name}"
        for workflow, name in readers
        if _job_environment(_jobs(_workflow_specs()[workflow])[name]) != environment
    ]
    assert outside == [], f"{secret} czytany poza środowiskiem {environment}: {outside}"


def _script_environment_secrets(script: str) -> dict[str, str]:
    text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    match = re.search(r'^ENVIRONMENT_SECRETS="([^"]*)"$', text, re.MULTILINE)
    assert match, f"scripts/{script} nie deklaruje ENVIRONMENT_SECRETS"
    return dict(entry.split("=", 1) for entry in match.group(1).split())


@pytest.mark.parametrize("script", SECRET_SCRIPTS)
def test_secret_scripts_know_the_environment_secrets(script: str) -> None:
    assert _script_environment_secrets(script) == ENVIRONMENT_SECRETS


FAKE_GH = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_DIR/calls"
case "$1 $2" in
  "auth status") exit 0 ;;
  "repo view") echo "owner/repo" ;;
  "variable list") ;;
  "secret set") cat > /dev/null ;;
  "secret list")
    case " $* " in
      *" --env "*) cat "$FAKE_DIR/environment_secrets" 2>/dev/null || exit 1 ;;
      *) cat "$FAKE_DIR/repo_secrets" ;;
    esac ;;
  *) echo "unexpected gh call: $*" >&2; exit 3 ;;
esac
"""


def _workflow_secret_names() -> set[str]:
    text = "".join(path.read_text(encoding="utf-8") for path in WORKFLOWS_DIR.glob("*.yml"))
    return set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text)) - {"GITHUB_TOKEN"}


def _run_secret_script(
    tmp_path: Path, script: str, *args: str, repo: set[str], environment: set[str] | None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    (tmp_path / "repo_secrets").write_text("".join(f"{n}\n" for n in sorted(repo)), "utf-8")
    if environment is not None:
        names = "".join(f"{n}\n" for n in sorted(environment))
        (tmp_path / "environment_secrets").write_text(names, encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SUPABASE_DB_URL=postgresql://example\nOPENAI_API_KEY=sk-example\n", encoding="utf-8"
    )
    result = subprocess.run(
        [str(ROOT / "scripts" / script), "-R", "owner/repo", "-f", str(env_file), *args],
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DIR": str(tmp_path),
            "GH_TOKEN": "",
            "GITHUB_TOKEN": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls_file = tmp_path / "calls"
    calls = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    return result, calls


def test_secret_sync_pushes_the_migration_url_into_its_environment(tmp_path: Path) -> None:
    result, calls = _run_secret_script(
        tmp_path, "gh-secrets-sync.sh", "--yes", repo=set(), environment=set()
    )

    assert result.returncode == 0, result.stdout + result.stderr
    sets = [call for call in calls if call.startswith("secret set")]
    assert "secret set SUPABASE_DB_URL --env production-db --repo owner/repo" in sets
    assert "secret set OPENAI_API_KEY --repo owner/repo" in sets


def test_secret_check_reads_the_migration_url_from_its_environment(tmp_path: Path) -> None:
    repo = _workflow_secret_names() - set(ENVIRONMENT_SECRETS)

    result, calls = _run_secret_script(
        tmp_path, "gh-secrets-check.sh", repo=repo, environment=set(ENVIRONMENT_SECRETS)
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "secret list --env production-db --repo owner/repo" in calls


def test_secret_check_flags_a_migration_url_left_only_on_the_repo(tmp_path: Path) -> None:
    result, _ = _run_secret_script(
        tmp_path, "gh-secrets-check.sh", repo=_workflow_secret_names(), environment=None
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "SUPABASE_DB_URL" in result.stdout
    assert "production-db" in result.stdout


# BEZ-21: Dependabot podbija wersje, ale nic nie mówiło, że lock niesie znaną podatność.
# osv-scanner skanuje uv.lock przy pushu na main, który zmienia lock (merge.sh i push, PR-ów
# nie ma), oraz co tydzień (nowe advisory na niezmienionym locku). Binarka jest przypięta
# wersją i sumą SHA-256, a wynik trafia do podsumowania joba także przy znaleziskach.
AUDIT_WORKFLOW = "dependency_audit.yml"


def _audit_runs() -> list[str]:
    spec = _workflow_specs()[AUDIT_WORKFLOW]
    return [
        str(step.get("run", "")) for job in _jobs(spec).values() for step in job.get("steps", [])
    ]


def test_dependency_audit_runs_on_lockfile_pushes_to_main_and_weekly() -> None:
    assert AUDIT_WORKFLOW in _workflow_specs(), f"brak .github/workflows/{AUDIT_WORKFLOW}"
    triggers = _triggers(_workflow_specs()[AUDIT_WORKFLOW])

    assert set(triggers) == {"push", "schedule", "workflow_dispatch"}, triggers
    assert triggers["push"] == {"branches": ["main"], "paths": ["uv.lock"]}
    (entry,) = triggers["schedule"]
    _minute, _hour, day, month, weekday = str(entry["cron"]).split()
    assert (day, month) == ("*", "*") and weekday.isdigit(), f"nie co tydzień: {entry['cron']}"


def test_dependency_audit_scans_the_lockfile_with_a_verified_binary() -> None:
    assert AUDIT_WORKFLOW in _workflow_specs(), f"brak .github/workflows/{AUDIT_WORKFLOW}"
    runs = "\n".join(_audit_runs())

    assert re.search(r"osv-scanner\S*\s+scan\s+source\s+--lockfile\s+uv\.lock\b", runs), runs
    assert "sha256sum --check --strict" in runs, "binarka osv-scanner bez sprawdzenia sumy"
    assert '>> "$GITHUB_STEP_SUMMARY"' in runs, "wynik skanu poza podsumowaniem joba"


# CI-23: narzędzie z PyPI uruchamiane przez `uvx` (actionlint, zizmor, shellcheck, mutmut)
# wykonuje kod w jobie tak samo jak akcja, więc ma dokładną wersję (`pkg==x.y.z` albo
# `pkg@x.y.z`) — inaczej każdy run brałby najnowsze wydanie.
_EXACT_PIN = re.compile(r"^[A-Za-z0-9._-]+(?:==|@)\d+(?:\.\d+)+$")


def _uvx_package_specs(command: str) -> list[str]:
    """Pakiety każdego `uvx` w komendzie: `--from X`, `--with X` albo pierwszy argument."""
    tokens = shlex.split(command.replace("\\\n", " "))
    specs: list[str] = []
    for start in (index for index, token in enumerate(tokens) if token == "uvx"):
        rest = tokens[start + 1 :]
        explicit = [
            rest[i + 1] for i, token in enumerate(rest[:-1]) if token in ("--from", "--with")
        ]
        positional = [
            token
            for i, token in enumerate(rest)
            if not token.startswith("-")
            and (i == 0 or rest[i - 1] not in ("--from", "--with", "--python"))
        ]
        specs.extend(explicit if "--from" in rest else [*explicit, *positional[:1]])
    return specs


def test_uvx_spec_reader_sees_from_with_and_bare_specs() -> None:
    assert _uvx_package_specs("uvx zizmor@1.30.1 --offline .") == ["zizmor@1.30.1"]
    assert _uvx_package_specs(
        "uvx --python 3.12 --from mutmut==2.5.1 mutmut run \\\n  --no-progress"
    ) == ["mutmut==2.5.1"]
    assert _uvx_package_specs("uvx --from a==1.0 --with b==2.0 a") == ["a==1.0", "b==2.0"]
    assert _uvx_package_specs("uvx zizmor .") == ["zizmor"]


def test_uvx_tools_in_workflows_are_pinned_to_an_exact_version() -> None:
    specs = [
        (workflow, spec)
        for workflow, name in _job_ids()
        for step in _jobs(_workflow_specs()[workflow])[name].get("steps", [])
        for spec in _uvx_package_specs(str(step.get("run", "")))
    ]

    assert {spec.split("=")[0].split("@")[0] for _, spec in specs} >= {
        "actionlint-py",
        "zizmor",
        "shellcheck-py",
        "mutmut",
    }
    unpinned = [(workflow, spec) for workflow, spec in specs if not _EXACT_PIN.match(spec)]
    assert unpinned == [], f"uvx bez dokładnej wersji: {unpinned}"


# CI-23: zizmor (artipacked) traktuje checkout bez `persist-credentials` jako pozostawiony
# token. Każdy checkout deklaruje więc zamiar jawnie: false, a true tylko tam, gdzie job
# pushuje tym tokenem (`publish-attestation-ledger`).
@pytest.mark.parametrize(("workflow", "name"), _job_ids())
def test_every_checkout_declares_persist_credentials(workflow: str, name: str) -> None:
    job = _jobs(_workflow_specs()[workflow])[name]

    for checkout in _checkouts(job):
        declared = dict(checkout.get("with") or {}).get("persist-credentials")
        assert isinstance(declared, bool), (
            f"{workflow}:{name}: checkout bez jawnego `persist-credentials`"
        )
        assert declared is False or (workflow, name) in _pushing_jobs(), (
            f"{workflow}:{name}: `persist-credentials: true` w jobie, który nie pushuje"
        )
