"""FIN-35: strażnik szwu workflow ↔ `paid_calls_enabled`.

`Settings.paid_calls_enabled` jest domyślnie false, więc lokalny run nie płaci
za LLM, Alpha Vantage ani embeddingi — DI podstawia atrapy. Skutek uboczny:
workflow, który uruchamia płacący entry-point bez jawnego
`PAID_CALLS_ENABLED: "true"`, po cichu przeszedłby na atrapy — cykl zielony,
predykcje neutralne, a mail bez analizy.

Kontrakt (jak `tests/test_db_writes_wiring.py`):
1. każdy `main_*.py` jest sklasyfikowany jako płacący albo darmowy;
2. każdy krok workflow z płacącym entry-pointem ustawia `PAID_CALLS_ENABLED: "true"`;
3. krok z darmowym entry-pointem tej zmiennej nie dostaje;
4. przełącznik nie trafia do commitowanego `config.toml`, a `.env.example`
   pokazuje go wyłączonego.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ENV_KEY = "PAID_CALLS_ENABLED"

# Wołają płatne porty: Fast Loop (LLM, rada, AV, embeddingi) i Slow Loop
# (fundamenty AV, recap, sędzia kalibracji).
PAYING_ENTRYPOINTS = frozenset({"main_agent.py", "main_trainer.py"})
# Nie wołają żadnego płatnego portu (watch i deadman z założenia, CLAUDE.md;
# publiczny scorecard WIA-52: repozytorium i zapis pliku; bot MAIL-48: Bot API i odczyty).
FREE_ENTRYPOINTS = frozenset(
    {
        "main_watch.py",
        "main_deadman.py",
        "main_backtest.py",
        "main_tools.py",
        "main_scorecard.py",
        "main_bot.py",
    }
)

_ENTRYPOINT_RUN = re.compile(r"python\s+(main_\w+\.py)")


def _entrypoint_steps() -> list[tuple[str, str, dict[str, object]]]:
    found: list[tuple[str, str, dict[str, object]]] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in spec.get("jobs", {}).values():
            for step in job.get("steps", []):
                for entrypoint in _ENTRYPOINT_RUN.findall(str(step.get("run", ""))):
                    found.append((path.name, entrypoint, dict(step.get("env") or {})))
    return found


STEPS = _entrypoint_steps()
STEP_IDS = [f"{workflow}:{entrypoint}" for workflow, entrypoint, _ in STEPS]


def test_every_entrypoint_is_classified_exactly_once() -> None:
    entrypoints = {path.name for path in ROOT.glob("main_*.py")}

    assert not PAYING_ENTRYPOINTS & FREE_ENTRYPOINTS
    assert entrypoints == PAYING_ENTRYPOINTS | FREE_ENTRYPOINTS, (
        f"{sorted(entrypoints - PAYING_ENTRYPOINTS - FREE_ENTRYPOINTS)} bez klasyfikacji"
    )


def test_paying_jobs_are_found_in_workflows() -> None:
    assert {entrypoint for _, entrypoint, _ in STEPS} >= PAYING_ENTRYPOINTS


@pytest.mark.parametrize(("workflow", "entrypoint", "env"), STEPS, ids=STEP_IDS)
def test_paid_calls_flag_matches_the_entrypoint(
    workflow: str, entrypoint: str, env: dict[str, object]
) -> None:
    if entrypoint in PAYING_ENTRYPOINTS:
        assert env.get(ENV_KEY) == "true", (
            f'{workflow} uruchamia {entrypoint}, ale nie ustawia {ENV_KEY}: "true" — '
            "bez tego produkcja po cichu jedzie na atrapach"
        )
    else:
        assert ENV_KEY not in env, (
            f"{workflow} uruchamia {entrypoint} (bez płatnych portów), a ustawia {ENV_KEY}"
        )


def test_config_toml_does_not_switch_paid_calls() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))

    assert "paid_calls_enabled" not in config, (
        "config.toml jest wspólny dla lokalnego runu i GHA — włączone w nim płatne "
        "wywołania płaciłyby z każdego lokalnego przebiegu"
    )


def test_env_example_ships_paid_calls_disabled() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert f"{ENV_KEY}=false" in lines
