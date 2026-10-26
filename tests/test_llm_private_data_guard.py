"""BEZ-32: pozycje portfela i adresy e-mail nigdy nie trafiają do promptu LLM.

Dziś żaden prompt ich nie czyta (inwentarz: `docs/data-flows.md`), ale nic tego nie
pilnowało, a lead maila zna już wagi pozycji. Strażnik stawia prywatne dane tam,
skąd bierze je produkcja — adresy w `Settings` (odbiorca i nadawca maila, kontakt
nagłówka SEC) i pozycje w adapterze `positions` — buduje porty Fast Loopa prawdziwym
DI (`main_agent.build_agent_graph_deps`), przepuszcza cztery cykle przez graf
(predykcja, diagnoza w reflect, 7 person i przewodniczący) i woła joby Slow Loopa
z LLM (recap tygodnia, sędzia diagnoz). Każdy prompt idzie przez rejestrator nad
skryptem trybu offline (DX-04). Zielony od startu; czerwony przy pierwszej zmianie,
która wstrzyknie portfel albo adres do `CouncilInput`, stanu grafu albo promptu.

Plik nie nazywa się `test_prompt_taint.py` — ta nazwa należy do BEZ-14 (fence pól
nieufnych w promptach)."""

from __future__ import annotations

import dataclasses
import json
import re
import threading
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import main_agent
import main_trainer
from src.application.agent_graph import AgentGraphDeps
from src.application.ports import LLMPort, PortfolioPositionsPort
from src.application.quota_monitor import QuotaMonitor
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.application.use_cases.judge_diagnoses import MAX_JUDGED_PER_RUN
from src.domain.asset import Asset
from src.domain.portfolio import Position
from src.domain.value_objects import Money
from src.infrastructure.adapters.fixture_market_data import FixtureMarketData
from src.infrastructure.adapters.in_memory_repo import InMemoryRepository
from src.infrastructure.adapters.local_stubs import (
    LocalStubNewsAdapter,
    LocalStubSentimentAdapter,
)
from src.infrastructure.llm.scripted_llm import ScriptedLLM
from tests.entrypoint_support import build_settings

ROOT = Path(__file__).parent.parent
LLM_SCRIPT = ROOT / "data" / "offline" / "llm_script.json"

OWNER_EMAIL = "owner-sentinel@example.com"
SENDER_EMAIL = "sender-sentinel@example.com"
SEC_CONTACT_EMAIL = "contact-sentinel@example.com"
QUANTITY_SENTINEL = Decimal("7319.4271")
PURCHASE_PRICE_SENTINEL = Decimal("4271.7319")
PRIVATE_FRAGMENTS = (str(QUANTITY_SENTINEL), str(PURCHASE_PRICE_SENTINEL))
EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

SEED_PRICES = {"NVDA": Decimal("100.00"), "MSFT": Decimal("400.00")}
# Każdy cykl przechodzi bramkę 2% dla obu symboli. Skrypt daje NVDA BULLISH i MSFT
# BEARISH, więc cykle 2–4 rozliczają poprzednie prognozy, w tym chybione (diagnoza),
# a po cyklu 4 tydzień ma 6 rozliczonych predykcji (recap wymaga co najmniej 5).
CYCLE_PRICES = (
    {"NVDA": Decimal("103.00"), "MSFT": Decimal("390.00")},
    {"NVDA": Decimal("106.20"), "MSFT": Decimal("402.00")},
    {"NVDA": Decimal("102.90"), "MSFT": Decimal("389.90")},
    {"NVDA": Decimal("106.10"), "MSFT": Decimal("401.70")},
)
CYCLE_INTERVAL = timedelta(hours=12)


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _RecordingLLM(LLMPort):
    """Zapisuje każdy prompt i odpowiada regułami skryptu trybu offline.
    Lock, bo persony rady wołają port z wątków."""

    def __init__(self, script: ScriptedLLM) -> None:
        self._script = script
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    def _record(self, method: str, prompt: str) -> None:
        with self._lock:
            self.calls.append((method, prompt))

    def analyze(self, prompt: str) -> dict[str, object]:
        self._record("analyze", prompt)
        return self._script.analyze(prompt)

    def analyze_mistake(self, prompt: str) -> str:
        self._record("analyze_mistake", prompt)
        return self._script.analyze_mistake(prompt)


class _SentinelPositions(PortfolioPositionsPort):
    """Adapter pozycji właściciela z rozpoznawalną ilością i ceną zakupu."""

    def __init__(self, **_connection: object) -> None:
        pass

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=symbol, quantity=QUANTITY_SENTINEL, purchase_price=PURCHASE_PRICE_SENTINEL
            )
            for symbol in SEED_PRICES
        ]

    def get_as_of(self) -> datetime | None:
        return datetime.now(UTC)


def _script() -> ScriptedLLM:
    return ScriptedLLM.from_mapping(json.loads(LLM_SCRIPT.read_text(encoding="utf-8")))


def _kind(method: str, prompt: str) -> str:
    if method == "analyze_mistake":
        return "diagnosis"
    if '"final_recommendation"' in prompt:
        return "chairman"
    if '"recommendation"' in prompt:
        return "persona"
    if '"trend_direction"' in prompt:
        return "prediction"
    return "other"


def _leaks(prompt: str) -> list[str]:
    found = EMAIL_ADDRESS.findall(prompt)
    return [*found, *(fragment for fragment in PRIVATE_FRAGMENTS if fragment in prompt)]


def _run_fast_loop(
    production_deps: AgentGraphDeps,
    clock: _Clock,
    cycles: tuple[Mapping[str, Decimal], ...],
) -> None:
    for prices in cycles:
        clock.now += CYCLE_INTERVAL
        # Porty danych z fake'ów trybu offline; LLM, rada i reszta pól z DI produkcji.
        deps = dataclasses.replace(
            production_deps,
            market_port=FixtureMarketData(prices),
            sentiment_port=LocalStubSentimentAdapter(),
            news_port=LocalStubNewsAdapter(),
            quote_port=None,
            embedding_port=None,
            clock=clock,
        )
        use_case = AnalyzeMarketUseCase(deps)
        for symbol in prices:
            use_case.run(symbol, Asset(symbol), now=clock.now)


@pytest.fixture
def private_context(mocker: MockerFixture, mock_external_clients: None) -> None:
    """Pozycje właściciela dostaje każdy, kto zbuduje adapter `positions` (dziś tylko
    raport przez `main_agent.build_positions_port`)."""
    mocker.patch(
        "src.infrastructure.adapters.positions_repo.SupabasePositionsAdapter", _SentinelPositions
    )


@pytest.mark.usefixtures("private_context")
def test_prompts_never_contain_positions_or_emails(mocker: MockerFixture) -> None:
    settings = build_settings(
        symbols=list(SEED_PRICES),
        portfolio_positions_enabled=True,
        digest_to_email=OWNER_EMAIL,
        digest_from_email=SENDER_EMAIL,
        edgar_user_agent=f"StockAgent/1.0 ({SEC_CONTACT_EMAIL})",
        # 0 = rada przy każdym symbolu za bramką (7 person + przewodniczący).
        council_volatility_threshold=Decimal("0"),
    )
    assert main_agent.build_positions_port(settings).get_positions(), "brak pozycji w kontekście"
    fast_loop_llm = _RecordingLLM(_script())
    mocker.patch("main_agent.build_llm_adapter", return_value=fast_loop_llm)
    clock = _Clock(datetime.now(UTC) - timedelta(days=3))
    repository = InMemoryRepository(clock=clock)
    for symbol, price in SEED_PRICES.items():
        repository.save_price_snapshot(symbol, Money(price))

    deps = main_agent.build_agent_graph_deps(settings, repository=repository)
    _run_fast_loop(deps, clock, CYCLE_PRICES)
    recap_llm = _RecordingLLM(_script())
    main_trainer._send_weekly_recap(settings, repository, QuotaMonitor(), recap_llm)
    judge_llm = _RecordingLLM(_script())
    main_trainer._run_diagnosis_judge(
        repository, judge_llm, MAX_JUDGED_PER_RUN, rejects_enabled=False
    )

    kinds = Counter(_kind(method, prompt) for method, prompt in fast_loop_llm.calls)
    # Strażnik bez promptów każdego rodzaju byłby zielony z pustego powodu.
    assert kinds["prediction"] >= 1
    assert kinds["diagnosis"] >= 1
    assert kinds["persona"] >= 7
    assert kinds["chairman"] >= 1
    assert len(recap_llm.calls) == 1
    assert judge_llm.calls
    leaks = {
        f"{_kind(method, prompt)}: {leak}"
        for llm in (fast_loop_llm, recap_llm, judge_llm)
        for method, prompt in llm.calls
        for leak in _leaks(prompt)
    }
    assert leaks == set(), f"prywatne dane w prompcie LLM: {sorted(leaks)}"
