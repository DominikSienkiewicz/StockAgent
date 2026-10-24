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
# nie biegły w żadnym workflowie. Smoke jest WYŁĄCZNIE ręczny (decyzja właściciela):
# bez crona, push i PR-ów, z osobnymi sekretami `SMOKE_*`, żeby nie zjadał limitów
# kluczy produkcyjnych i nie mailował na adres digestu.
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
    spec = _smoke_spec()
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True))
    steps = [step for job in spec["jobs"].values() for step in job.get("steps", [])]
    markers = [
        marker
        for step in steps
        if (marker := _pytest_marker_expression(str(step.get("run", "")))) is not None
    ]

    assert isinstance(triggers, dict) and set(triggers) == {"workflow_dispatch"}, triggers
    assert markers == ["integration"], markers


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
GROUP_EXEMPT_WRITERS = frozenset({"shock_watch_hourly.yml"})
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
        f"{sorted(workflow_groups & job_groups)}"
    )


def test_hourly_shock_watch_stays_out_of_the_writers_queue() -> None:
    spec = _workflow_specs()["shock_watch_hourly.yml"]
    groups = [_concurrency(spec)[0], *(_concurrency(job)[0] for job in _jobs(spec).values())]

    assert SUPABASE_WRITERS_GROUP not in groups


# CI-10: polityki workflowów były dotrzymane ręcznie i nierówno. Test parsuje każdy plik
# z `.github/workflows/`, więc nowy workflow podlega im od pierwszego commitu.
def _triggers(spec: dict[str, Any]) -> dict[str, Any]:
    """Wyzwalacze workflowu; PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True."""
    triggers = spec.get("on", spec.get(True)) or {}
    return triggers if isinstance(triggers, dict) else {}


def _job_ids() -> list[tuple[str, str]]:
    return [
        (workflow, name)
        for workflow, spec in _workflow_specs().items()
        for name in _jobs(spec)
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
