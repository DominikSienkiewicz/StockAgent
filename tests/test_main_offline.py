"""DX-04: tryb offline całego cyklu (`main_tools.py offline`) — bez kluczy i bez sieci.

Dwa cykle scenariusza z `data/offline/` idą przez prawdziwy graf (bramka,
reflect, rada, zapis), `RunFastLoopCycle` i `ReportAssembler` na fake'ach:
`InMemoryRepository`, `FixtureMarketData`, `ScriptedLLM` i notifier do plików.
Cykl 1 zapisuje predykcję NVDA, cykl 2 z ceną +3% ją rozlicza, a każdy cykl
kończy się mailem HTML z sekcjami znanymi z golden snapshotu (TEST-02).

Brak ruchu sieciowego sprawdza rejestrator gniazd internetowych i DNS (ponad
blokadę pytest-socket, której wyjątek adapter mógłby połknąć); brak kluczy —
`Settings.from_env`, który w tym trybie wywraca test.
"""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path
from typing import Any

import pytest

import main_tools
from src.config import Settings
from src.infrastructure.llm.scripted_llm import ScriptedLLM
from src.tools.offline_scenario import parse_offline_scenario

ROOT = Path(__file__).parent.parent
OFFLINE_DIR = ROOT / "data" / "offline"
GOLDEN_HTML = ROOT / "tests" / "application" / "snapshots" / "full_cycle.html"
SECRET_SUFFIXES = ("_KEY", "_KEYS", "_TOKEN", "_SECRET", "_URL", "_EMAIL")
# Sekcje, które scenariusz musi pokazać: nowe predykcje, rozliczenie i pominięte.
CORE_SECTIONS = ("🔮 Wygenerowane predykcje", "📊 Zamknięte predykcje", "⏸ Pominięte")


@pytest.fixture
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Każda próba gniazda internetowego albo zapytania DNS ląduje na liście."""
    attempts: list[tuple[Any, ...]] = []
    guarded = socket.socket

    class _RecordingSocket(guarded):  # type: ignore[misc, valid-type]
        def __new__(cls, family: int = -1, *args: Any, **kwargs: Any) -> Any:
            if family in (-1, socket.AF_INET, socket.AF_INET6):
                attempts.append(("socket", family))
                raise OSError("offline: network socket attempted")
            return super().__new__(cls, family, *args, **kwargs)

    def _getaddrinfo(*args: Any, **kwargs: Any) -> Any:
        attempts.append(("getaddrinfo", *args))
        raise OSError("offline: DNS lookup attempted")

    monkeypatch.setattr(socket, "socket", _RecordingSocket)
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    return attempts


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Środowisko bez sekretów, a `Settings.from_env` wywraca test."""
    for name in list(os.environ):
        if name.endswith(SECRET_SUFFIXES):
            monkeypatch.delenv(name)

    def _refuse(*_args: Any, **_kwargs: Any) -> Settings:
        raise AssertionError("offline mode must not build Settings")

    monkeypatch.setattr(Settings, "from_env", _refuse)


def _run(out_dir: Path) -> main_tools.OfflineRun:
    scenario = parse_offline_scenario(
        json.loads((OFFLINE_DIR / "scenario.json").read_text(encoding="utf-8"))
    )
    llm = ScriptedLLM.from_mapping(
        json.loads((OFFLINE_DIR / "llm_script.json").read_text(encoding="utf-8"))
    )
    return main_tools.run_offline(
        scenario,
        llm=llm,
        out_dir=out_dir,
        ml_model_path=OFFLINE_DIR / "model.ubj",
        personas_dir=ROOT / "data" / "council_personas",
    )


def _headings(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", raw).strip() for raw in re.findall(r"<h2[^>]*>(.*?)</h2>", html)]


def test_two_offline_cycles_settle_first_prediction(
    tmp_path: Path, network_attempts: list[tuple[Any, ...]], no_keys: None
) -> None:
    run = _run(tmp_path)

    first, second = run.cycles
    assert first.statuses["NVDA"] == "saved"
    settled = {
        row["symbol"]: row["is_trend_correct"]
        for row in run.repository.get_resolved_predictions(30)
    }
    assert settled["NVDA"] is True
    assert ("NVDA", True) in [(item.symbol, item.is_correct) for item in second.resolved]
    assert "NVDA" in second.html_path.read_text(encoding="utf-8")
    assert network_attempts == []


def test_offline_mail_has_the_sections_of_the_golden_snapshot(
    tmp_path: Path, network_attempts: list[tuple[Any, ...]]
) -> None:
    golden = _headings(GOLDEN_HTML.read_text(encoding="utf-8"))

    second = _run(tmp_path).cycles[1]

    offline = _headings(second.html_path.read_text(encoding="utf-8"))
    assert set(offline) <= set(golden), f"sekcje spoza golden: {set(offline) - set(golden)}"
    for core in CORE_SECTIONS:
        assert any(heading.startswith(core) for heading in offline), core
    assert network_attempts == []


def test_offline_run_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "a").cycles
    second = _run(tmp_path / "b").cycles

    for left, right in zip(first, second, strict=True):
        assert left.subject == right.subject
        assert left.html_path.read_bytes() == right.html_path.read_bytes()
        assert left.text_path.read_bytes() == right.text_path.read_bytes()


def test_offline_subcommand_writes_one_mail_per_cycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    network_attempts: list[tuple[Any, ...]],
    no_keys: None,
) -> None:
    exit_code = main_tools.main(["offline", "--out-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert (tmp_path / "cycle-1" / "report-email.html").is_file()
    assert (tmp_path / "cycle-2" / "report-email.txt").read_text(encoding="utf-8").startswith(
        "Temat: "
    )
    assert "Cykl 1" in out and "Cykl 2" in out
    assert "NVDA ✓" in out
    assert network_attempts == []
