  # CLAUDE.md — zasady pracy w repo StockAgent

## Stack i narzędzia

- **Python 3.12**, manager zależności: **`uv`** (nie `pip`!). Każda komenda przez `uv run ...`.
- Dodanie zależności: `uv add <pkg>` → commituj **oba** `pyproject.toml` + `uv.lock`.
- macOS: XGBoost wymaga `brew install libomp` (jednorazowo).

## Quality gate — przed zgłoszeniem pracy jako gotowej

Każda zmiana w kodzie musi przejść **wszystkie trzy**:

```bash
uv run ruff check src tests main_agent.py main_trainer.py main_watch.py
uv run mypy src main_agent.py main_trainer.py main_watch.py   # strict mode
uv run pytest
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
- **`main_agent.py` / `main_trainer.py` / `main_watch.py`** — DI Container. Jedyne miejsce,
  gdzie konkretne adaptery są łączone z use case'ami. `main_watch.py` (#11) reużywa
  fabryki z `main_agent` i NIE WOŁA żadnego płatnego portu ani `save_price_snapshot`.

Nowa integracja zewnętrzna = nowy port w `application/ports.py` + adapter w
`infrastructure/`. Nigdy nie wstrzykuj konkretnej klasy infrastruktury do `application`.

## TDD — testy przed implementacją

Pracuj cyklem **Red → Green → Refactor**: najpierw test (który failuje), potem
minimalna implementacja, na końcu porządki. Każdy port ma testy z mockami,
każdy adapter ma testy jednostkowe (mock sieci) + opcjonalnie integracyjne.

## Konwencje testów

- Adaptery HTTP: testy mockują **`requests.Session.get` / `requests.Session.post`**
  (adaptery wołają przez `self._session` z `_http.build_session()`, nie przez gołe `requests`).
- Testy integracyjne (prawdziwe API): oznaczone `@pytest.mark.integration`,
  pomijane bez kluczy w env. CI uruchamia `pytest -m "not integration"`.
- Mocki portów: `Mock(spec=SomePort)`.

## Reguły domenowe / FinOps

- **Bramka volatility**: płatne porty (LLM, Alpha Vantage, embeddingi) **nie mogą**
  być wołane, gdy zmiana ceny < `volatility_threshold`. Logika progu żyje w domenie
  (`Asset.evaluate_volatility`) — graf jest tylko wykonawcą.
- **Cold-start**: `check_price_node` zapisuje snapshot ceny w KAŻDYM cyklu
  (`price_snapshots`), żeby następny cykl miał punkt odniesienia.
- **Self-Reflection**: `reflect_node` działa PRZED bramką volatility — ocena
  przeszłej predykcji jest niezależna od tego, czy bieżący cykl robi nową prognozę.
- **Resilience**: pojedynczy błąd per-symbol nie wywala cyklu. `main_agent.main()`
  zwraca exit 1 tylko gdy **wszystkie** symbole padły.

## Migracje bazy

- Pliki: `supabase/migrations/NNN_nazwa.sql`, numer zero-paddowany do 3 cyfr.
  Ta lokalizacja i konwencja nazw NIE są dowolne — `supabase db push` czyta
  wyłącznie `supabase/migrations/` i parsuje nazwy regexem `^([0-9]+)_(.*)\.sql$`.
  Plik spoza wzorca jest **po cichu POMIJANY** (ostrzeżenie na stderr, nie błąd).
- Zero-padding jest wymagany: CLI aplikuje pliki w kolejności leksykograficznej,
  więc bez niego `10_x.sql` poszłoby PRZED `2_x.sql`.
- Pierwsza migracja nie może nazywać się `<14 cyfr>_init.sql` — CLI pomija taki
  plik dla wstecznej kompatybilności.
- Każda nowa migracja = wpis w `MIGRATION_FILES` + asercja w
  `tests/infrastructure/test_migrations.py` (testy aplikują komplet na prawdziwym
  kontenerze pgvector; osobny test pilnuje, żeby lista nie rozjechała się z katalogiem).
- Aplikacja: ręczny workflow „🗄️ DB Migrate", nigdy z crona Fast Loopa.

## Sekrety

- **Nigdy** nie commituj `.env`. Do `.env.example` wpisuj wyłącznie placeholdery.
- Klucze API w GitHub Actions: jako Repository Secrets (workflowy czytają `${{ secrets.* }}`).

## Synchronizacja `.env` ↔ `.env.example`

Plik `.env.example` to **template dla nowych deploymentów** — musi się rozwijać
razem z `.env`, nie zostawać w tyle. Po każdej zmianie w `.env` zastosuj te
reguły do `.env.example` w tym samym kroku:

- **Sekrety** (API keys, tokeny, hasła, DB credentials, adresy mailowe odbiorców):
  wyłącznie **placeholdery** typu `sk-...`, `re_...`,
  `https://twoj-projekt.supabase.co`, `you@example.com`. Nigdy realnych wartości
  z `.env`.
- **Cała reszta** (konfiguracja niewrażliwa — `SYMBOLS`, `SYMBOLS_ETF`,
  `RISK_SYMBOLS`, `RISK_SYMBOL_TYPES`, `SYMBOLS_UNSUPPORTED_PRICE`,
  `VOLATILITY_THRESHOLD`, `SYMBOL_THROTTLE_SECONDS`, `NBP_ENABLED`,
  `COUNCIL_LLM_MODEL`, `ML_MODEL_PATH`, `NOTIFICATIONS_ENABLED`,
  `LLM_PROVIDER`, `DIGEST_FROM_EMAIL` itd.): **identyczne wartości** jak w `.env`.
  Template pokazuje rzeczywistą produkcyjną konfigurację, żeby nowy deployment
  startował z sensownym defaultem zamiast minimalistycznego szkieletu wymagającego
  dośledzenia z README.

Gdy pole jest **wrażliwe** (sekret) — placeholder w `.env.example`, prawdziwa
wartość w `.env`. Gdy **niewrażliwe** — dokładnie ten sam string w obu plikach.

Sygnał, że robisz to dobrze: `diff .env .env.example` pokazuje **tylko** linie
z sekretami; każda inna różnica to dryf, który trzeba naprawić.

## Język

- Komentarze w kodzie i treść raportu mailowego: **polski**.
- `README.md`: **angielski**.
- Identyfikatory (klasy, funkcje, zmienne): angielski.

## Znane kompromisy

- `agent_graph.py` ma per-file override w `[tool.mypy]` (`disable_error_code = ["arg-type"]`)
  — typing stubs LangGraph 1.x nie inferują `StateGraph[AgentState]` poprawnie.
  Runtime jest poprawny; nie usuwaj override'u bez weryfikacji.
- `report_builder.py` ma `E501` wyłączone (per-file w ruff) — inline HTML
  templating ma naturalnie długie linie.
