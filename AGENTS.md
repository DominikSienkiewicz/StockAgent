# AGENTS.md — zasady pracy w repo StockAgent

## Stack i narzędzia

- **Python 3.12**, manager zależności: **`uv`** (nie `pip`!). Każda komenda przez `uv run ...`.
- Dodanie zależności: `uv add <pkg>`; w zmianie uwzględnij **oba** `pyproject.toml` + `uv.lock`.
- macOS: XGBoost wymaga `brew install libomp` (jednorazowo).

## Quality gate — przed zgłoszeniem pracy jako gotowej

Każda zmiana w kodzie musi przejść **wszystkie trzy**:

```bash
uv run ruff check .
uv run mypy          # strict mode; zakres = [tool.mypy].files (src + main_*.py + ratchet tests/, od tests/domain)
uv run pytest -m "not integration and not containers"   # jak CI; kontenerowe: uv run pytest -m containers (Docker)
```

Nie zgłaszaj pracy jako ukończonej, dopóki te trzy nie są zielone.

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
- **Resilience**: pojedynczy błąd per-symbol nie wywala cyklu. `main_agent.main()`
  zwraca exit 1 tylko gdy **wszystkie** symbole padły.

## Sekrety

- **Nigdy** nie commituj `.env`. Do `.env.example` wpisuj wyłącznie placeholdery.
- Klucze API w GitHub Actions: jako Repository Secrets (workflowy czytają `${{ secrets.* }}`).

## Język

- Komentarze w kodzie i treść raportu mailowego: **polski**.
- `README.md`: **angielski**.
- Identyfikatory (klasy, funkcje, zmienne): angielski.

## Znane kompromisy

- `agent_graph.py` ma jeden wąski inline `# type: ignore[arg-type]` przy węźle
  `fetch_fundamentals` (`workflow.add_node`) — typing stubs LangGraph 1.x typują
  węzły jako `_Node[Never]` i nie inferują `StateGraph[AgentState]`. W `[tool.mypy]`
  nie ma per-file override'u. Runtime jest poprawny; nie usuwaj ignore'a bez weryfikacji.
- `report_builder.py` ma `E501` wyłączone (per-file w ruff) — inline HTML
  templating ma naturalnie długie linie.

## Local project concepts

Keep new feature proposals, plans, audits, and exploratory notes in local
`docs/concepts/`, which is ignored by Git. Maintained technical documentation
and accepted architecture decisions remain tracked elsewhere under `docs/`.
