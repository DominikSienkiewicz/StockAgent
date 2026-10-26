"""BEZ-08: strażnik szwu workflow ↔ `db_writes_enabled`.

`Settings.db_writes_enabled` jest domyślnie false, więc lokalny run nie zapisuje
do wspólnej bazy Supabase. Skutek uboczny: workflow, który uruchamia zapisujący
entry-point bez jawnego `DB_WRITES_ENABLED: "true"`, po cichu przestałby
zapisywać snapshoty, predykcje i rozliczenia — cykl zielony, track record stoi.

Kontrakt:
1. każdy `main_*.py` jest sklasyfikowany jako zapisujący albo niezapisujący
   (nowy entry-point bez klasyfikacji = czerwony test);
2. każdy krok workflow z zapisującym entry-pointem ustawia `DB_WRITES_ENABLED: "true"`;
3. krok z entry-pointem tylko do odczytu tej zmiennej nie dostaje (najmniejsze
   uprawnienia — deadman ma zostać monitorem, nie pisarzem);
4. przełącznik nie trafia do commitowanego `config.toml` (włączyłby zapis każdemu
   lokalnemu runowi), a `.env.example` pokazuje go wyłączonego.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ENV_KEY = "DB_WRITES_ENABLED"

# Zapisują do bazy przez strażnika (`build_repository` i fabryki portów zapisu; bot
# MAIL-48: offset `bot_state` i wyciszenia `symbol_mutes` przez `build_bot_store`).
WRITING_ENTRYPOINTS = frozenset(
    {"main_agent.py", "main_trainer.py", "main_watch.py", "main_tools.py", "main_bot.py"}
)
# Tylko czytają: deadman (probe + ślady życia), backtest (czysta ewaluacja)
# i publiczny scorecard (WIA-52: zamknięte predykcje, zapis tylko pliku strony).
READ_ONLY_ENTRYPOINTS = frozenset({"main_deadman.py", "main_backtest.py", "main_scorecard.py"})
# Joby, które dziś zapisują — chroni przed pustą parametryzacją poniżej.
EXPECTED_WRITING_JOBS = frozenset({"main_agent.py", "main_trainer.py", "main_watch.py"})

_ENTRYPOINT_RUN = re.compile(r"python\s+(main_\w+\.py)")


def _entrypoint_steps() -> list[tuple[str, str, dict[str, object]]]:
    """(plik workflow, entry-point, env kroku) dla każdego kroku z `python main_*.py`."""
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

    assert not WRITING_ENTRYPOINTS & READ_ONLY_ENTRYPOINTS
    unclassified = entrypoints - WRITING_ENTRYPOINTS - READ_ONLY_ENTRYPOINTS
    assert not unclassified, (
        f"{sorted(unclassified)} bez klasyfikacji — dopisz do WRITING_ENTRYPOINTS "
        "albo READ_ONLY_ENTRYPOINTS"
    )
    assert entrypoints == WRITING_ENTRYPOINTS | READ_ONLY_ENTRYPOINTS


def test_writing_jobs_are_found_in_workflows() -> None:
    assert {entrypoint for _, entrypoint, _ in STEPS} >= EXPECTED_WRITING_JOBS


@pytest.mark.parametrize(("workflow", "entrypoint", "env"), STEPS, ids=STEP_IDS)
def test_db_writes_flag_matches_the_entrypoint(
    workflow: str, entrypoint: str, env: dict[str, object]
) -> None:
    if entrypoint in WRITING_ENTRYPOINTS:
        assert env.get(ENV_KEY) == "true", (
            f'{workflow} uruchamia {entrypoint}, ale nie ustawia {ENV_KEY}: "true" — '
            "bez tego strażnik po cichu pomija każdy zapis do bazy"
        )
    else:
        assert ENV_KEY not in env, (
            f"{workflow} uruchamia {entrypoint} (tylko odczyt), a ustawia {ENV_KEY}"
        )


def test_config_toml_does_not_switch_db_writes() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))

    assert "db_writes_enabled" not in config, (
        "config.toml jest wspólny dla lokalnego runu i GHA — zapis włączony w nim "
        "pisałby do produkcyjnej bazy z każdego lokalnego przebiegu"
    )


def test_env_example_ships_db_writes_disabled() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert f"{ENV_KEY}=false" in lines
