# AGENTS.md — zasady pracy w repo StockAgent

## Stack i narzędzia

- **Python 3.12**, manager zależności: **`uv`** (nie `pip`!). Każda komenda przez `uv run ...`.
- Dodanie zależności: `uv add <pkg>`; w zmianie uwzględnij **oba** `pyproject.toml` + `uv.lock`.
- macOS: XGBoost wymaga `brew install libomp` (jednorazowo).

## Quality gate — przed zgłoszeniem pracy jako gotowej

Każda zmiana w kodzie musi przejść **wszystkie cztery**:

```bash
uv run ruff check .
uv run ruff format --check .   # CI-17: repo jest sformatowane; przed commitem `uv run ruff format <pliki>`
uv run mypy          # strict mode; zakres = [tool.mypy].files (src + main_*.py + cały tests/, TEST-13)
uv run pytest -m "not integration and not containers"   # jak CI; kontenerowe: uv run pytest -m containers (Docker)
```

Nie zgłaszaj pracy jako ukończonej, dopóki te cztery nie są zielone. Commit samego
formatowania dopisz do `.git-blame-ignore-revs`.
CI (job `quality`) uruchamia pytest dodatkowo z `--cov=src/domain --cov-branch --cov-fail-under=98`
(CI-03). Próg tylko rośnie: ten sam numer w `[tool.coverage.report].fail_under` i w
`COVERAGE_RATCHET` (`tests/test_quality_gate_sync.py`).

## Architektura — Hexagonal + DDD

Kierunek zależności jest **jednokierunkowy**, nie wolno go odwracać:

```
domain  ←  application  ←  infrastructure
```

- **`src/domain/`** — czysta logika biznesowa. **Zero importów zewnętrznych** (brak
  `requests`, `pandas`, `openai`, `langgraph` itd.). Tylko stdlib + inne moduły domeny.
- **`src/application/`** — porty (interfejsy ABC w `ports.py`), use cases, graf
  LangGraph, prompty, report builder. Zna `domain`, **nie zna** konkretnych adapterów.
- **`src/infrastructure/`** — adaptery implementujące porty. Jedyne miejsce z I/O.
- **`main_agent.py` / `main_trainer.py`** — DI Container. Jedyne miejsce, gdzie
  konkretne adaptery są łączone z use case'ami.

Nowa integracja zewnętrzna = nowy port w `application/ports.py` + adapter w
`infrastructure/`. Nigdy nie wstrzykuj konkretnej klasy infrastruktury do `application`.

## Konwencje testów

- Każdy port ma testy z mockami; każdy adapter ma testy jednostkowe (mock sieci)
  + opcjonalnie integracyjne.
- Adaptery HTTP: testy mockują **`requests.Session.get` / `requests.Session.post`**
  (adaptery wołają przez `self._session` z `_http.build_session()`, nie przez gołe `requests`).
- Testy integracyjne (prawdziwe API): oznaczone `@pytest.mark.integration`,
  pomijane bez kluczy w env. CI uruchamia testy bez integracji oraz osobny zestaw
  `@pytest.mark.containers`.
- Mocki portów: `Mock(spec=SomePort)`.
- Prawdziwe `Settings` w testach, nie `MagicMock`: `build_settings(**overrides)` z
  `tests/entrypoint_support.py` (rzuca na nieznane pole) albo `Settings(**NO_ENV_FILE, ...)`
  z `tests/settings_kwargs.py` — mypy nie zna `_env_file`, więc bez `# type: ignore`.
  Wspólne fixture'y `settings` i `mock_external_clients` żyją w `tests/conftest.py`.

## Reguły domenowe / FinOps

- **Bramka volatility**: płatne porty (LLM, Alpha Vantage, embeddingi) **nie mogą**
  być wołane, gdy zmiana ceny < `volatility_threshold`. Logika progu żyje w domenie
  (`Asset.evaluate_volatility`) — graf jest tylko wykonawcą.
- **Cold-start**: snapshot ceny (`price_snapshots`) zapisuje węzeł TERMINALNY każdego
  ukończonego cyklu — `save_node` albo `ignore_node` (`_persist_price_snapshot`), nie
  `check_price_node` — żeby następny cykl miał punkt odniesienia. Cykl padły w połowie
  nie zostawia snapshotu, więc retry liczy Δ od ostatniego ukończonego cyklu (#6).
- **Self-Reflection**: `reflect_node` działa PRZED bramką volatility — ocena
  przeszłej predykcji jest niezależna od tego, czy bieżący cykl robi nową prognozę.
- **Resilience**: pojedynczy błąd per-symbol nie wywala cyklu. Entry-pointy per symbol
  (`main_agent`, `main_trainer`, `main_watch`, `main_backtest`) zwracają exit 1 tylko gdy
  **wszystkie** symbole padły (`src/domain/exit_policy.run_exit_code`, NIEZ-38). Niedostarczenie raportu (mail
  Fast Loopa albo recap Slow Loopa, gdy kanał jest skonfigurowany) też zostawia exit 0,
  ale nie jest ciche: `flag_undelivered_report` (`main_agent.py`) drukuje adnotację
  `::error title=StockAgent::Raport nie został dostarczony` i zapisuje alert CRITICAL
  (`source` „Raport”) w `quota_alerts` (NIEZ-12).
- **Błędy portów**: taksonomia `src/application/errors.py` (ARCH-33) — `PortError` →
  `ProviderUnavailable`, `QuotaExhausted`, `MalformedPayload`, `NotFound`. Adapter
  tłumaczy wyjątek dostawcy przez `as_port_error`, zachowując typ dostawcy. Nowy kod
  łapie klasy taksonomii, nie `except Exception` — `tests/test_exception_ratchet.py`
  pilnuje, żeby szerokich handlerów nie przybywało (obniżaj baseline, gdy któryś
  znika). Sesje z `build_session()` mają circuit breaker per host na czas procesu
  (NIEZ-31).

## Sekrety

- **Nigdy** nie commituj `.env`. Do `.env.example` wpisuj wyłącznie placeholdery.
- Konfiguracja niewrażliwa żyje w commitowanym `config.toml`; `.env` i `.env.example`
  zawierają wyłącznie sekrety oraz przełączniki `DB_WRITES_ENABLED` i `PAID_CALLS_ENABLED`
  (DOC-30).
- Klucze API w GitHub Actions: jako Repository Secrets (workflowy czytają `${{ secrets.* }}`).

## Dziennik decyzji (DOC-40)

- Pozycja roadmapy ma numer `R-N`, definiowany wyłącznie w `docs/decisions.md`; decyzje
  architektoniczne to wpisy `D-N` tamże, incydenty — wiersze w `docs/engineering-log.md`.

## Język

- Komentarze w kodzie i treść raportu mailowego: **polski**.
- `README.md`: **angielski**.
- Identyfikatory (klasy, funkcje, zmienne): angielski.

## Znane kompromisy

- `agent_graph.py` ma jeden wąski inline `# type: ignore[arg-type]` przy węźle
  `fetch_fundamentals` (`workflow.add_node`) — typing stubs LangGraph 1.x typują
  węzły jako `_Node[Never]` i nie inferują `StateGraph[AgentState]`. W `[tool.mypy]`
  nie ma per-file override'u. Runtime jest poprawny; nie usuwaj ignore'a bez weryfikacji.

## Local project concepts

Keep new feature proposals, plans, audits, and exploratory notes in local
`docs/concepts/`, which is ignored by Git. Maintained technical documentation
and accepted architecture decisions remain tracked elsewhere under `docs/`.
