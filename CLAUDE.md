  # CLAUDE.md — zasady pracy w repo StockAgent

## Stack i narzędzia

- **Python 3.12**, manager zależności: **`uv`** (nie `pip`!). Każda komenda przez `uv run ...`.
- Dodanie zależności: `uv add <pkg>` → commituj **oba** `pyproject.toml` + `uv.lock`.
- macOS: XGBoost wymaga `brew install libomp` (jednorazowo).

## Quality gate — przed zgłoszeniem pracy jako gotowej

Każda zmiana w kodzie musi przejść **wszystkie trzy**:

```bash
uv run ruff check .
uv run mypy          # strict mode; zakres = [tool.mypy].files (src + main_*.py + ratchet tests/, od tests/domain)
uv run pytest -m "not integration and not containers"   # jak CI; kontenerowe: uv run pytest -m containers (Docker)
```

Nie zgłaszaj pracy jako ukończonej, dopóki te trzy nie są zielone.
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
- **`main_agent.py` / `main_trainer.py` / `main_watch.py` / `main_deadman.py` / `main_tools.py` / `main_scorecard.py`** — DI Container.
  Jedyne miejsce, gdzie konkretne adaptery są łączone z use case'ami. `main_watch.py` (#11)
  i `main_deadman.py` (NIEZ-11) reużywają fabryk z `main_agent` i NIE WOŁAJĄ żadnego
  płatnego portu ani `save_price_snapshot`. `main_scorecard.py` (WIA-52) tylko czyta bazę
  i zapisuje plik strony — bez płatnych portów i bez zapisu do bazy.
  Każde narzędzie jednorazowe to podkomenda `main_tools.py`, które jako jedyne składa dla niego
  `Settings` i adaptery, a `src/tools/` trzyma wyłącznie czyste funkcje — osobny `main_<x>.py`
  dostaje tylko entry-point z własnym workflow (ARCH-39, bez wyjątków).

Nowa integracja zewnętrzna = nowy port w `application/ports.py` + adapter w
`infrastructure/`. Nigdy nie wstrzykuj konkretnej klasy infrastruktury do `application`.

Podłączenie portu do Fast Loopa = **jedno pole** w `AgentGraphDeps`
(`application/agent_graph.py`). Ten sam obiekt biorą `create_agent_graph`
i `AnalyzeMarketUseCase`, więc nie ma już trzech list argumentów do zsynchronizowania.
Węzeł grafu zwraca `AgentState`; nowy klucz stanu = pole w `AgentState`. LangGraph po
cichu gubi klucz spoza schematu — pilnują tego mypy i
`test_node_outputs_use_only_declared_state_keys` (ARCH-10).

Analogicznie nowa sekcja maila = nowy moduł `src/application/report_<nazwa>.py`
(renderery HTML i tekstu + obiekt `ReportSection`), jeden wpis w `REPORT_SECTIONS`
(`application/reporting/registry.py`, kolejność maila) i jedno pole w `ReportContext`
(`application/report_builder.py`), które sekcja deklaruje w `context_fields`. Pole
czytane przez wspólne wejście sekcji (`SectionInput`) albo przez kilka sekcji trafia do
`SHARED_CONTEXT_FIELDS`; `tests/application/test_report_registry.py` pilnuje, że każde
pole ma dokładnie jednego konsumenta (ARCH-12). Pole z odczytów repozytorium wypełnia
`ReportAssembler.assemble` (`application/use_cases/assemble_report.py`) przez
`dataclasses.replace`; pole z adaptera (np. portfel właściciela) dokłada
`main_agent._report_inputs`.

`main_agent.py` zostaje DI rootem Fast Loopa: mapuje `Settings` na opcje use case'ów
(`ReportFlags`, `CycleOptions`, `PublicScorecardOptions`), buduje adaptery i kanały
wysyłki i wstrzykuje hooki świata (`time.sleep`, `redact_secrets`, `symbol_context`).
Logika cyklu i raportu żyje w `src/application/use_cases/`: `run_cycle.py`
(`RunFastLoopCycle` — przebieg symboli z jednym `CycleOptions`), `assemble_report.py`
(`ReportAssembler` — zna tylko `RepositoryPort` i `ReportFlags`, nigdy `Settings`; okno
track recordu czyta raz na dispatch `TrackRecordReads`), `publish_public_scorecard.py`
i `reveal_attestations.py` (wyłącznie na portach). `main_agent.deliver_report`
(composition, bo kanały buduje z `Settings`) składa i rozsyła raport i zwraca
`DeliveryOutcome` (`application/report_delivery.py`) ze statusem każdego kanału
(ARCH-02, ARCH-06, ARCH-07, ARCH-08).

## Jednostki zmian względnych (ARCH-04)

- `Ratio` = ułamek (0.025 = 2,5%), `Percent` = procent (2.5 = 2,5%) — NewType nad
  `Decimal` w `src/domain/value_objects.py`. Sufiks `_pct` oznacza procent (`Percent`);
  ułamek nosi typ `Ratio` i w nowym kodzie nazwę bez `_pct` (np. `fraction`, `delta`).
- Ułamek zmienia się w procent wyłącznie przez `to_percent` — NewType nie chroni
  arytmetyki (`Ratio * 100` to goły `Decimal`). Ręczne `* 100` tylko w formaterach
  prezentacji (`report_formatting.pct`, `format_pct_pl`).
- Stan grafu (`AgentState.delta`, `drift`) i `PriceDelta.fraction` to ułamki;
  `PriceDelta.percentage` to zdeprecjonowany alias ułamka (do usunięcia, ARCH-14).
- Starsze pola `_pct` z ułamkiem mają już typ `Ratio` i czekają na zmianę nazwy —
  rozstrzyga typ, nie nazwa.

## TDD — testy przed implementacją

Pracuj cyklem **Red → Green → Refactor**: najpierw test (który failuje), potem
minimalna implementacja, na końcu porządki. Każdy port ma testy z mockami,
każdy adapter ma testy jednostkowe (mock sieci) + opcjonalnie integracyjne.

## Konwencje testów

- Adaptery HTTP: testy mockują **`requests.Session.get` / `requests.Session.post`**
  (adaptery wołają przez `self._session` z `_http.build_session()`, nie przez gołe `requests`).
- Adapter z sekretem w URL-u albo nagłówku woła `safe_get` / `raise_for_status_redacted`
  z `_http.py`, a jego test sprawdza, że komunikat wyjątku nie zawiera sekretu.
- Testy integracyjne (prawdziwe API): oznaczone `@pytest.mark.integration`,
  pomijane bez kluczy w env. CI uruchamia `pytest -m "not integration and not containers"`,
  a testy kontenerowe osobnym krokiem `pytest -m containers`.
- Mocki portów: `Mock(spec=SomePort)`.

## Reguły domenowe / FinOps

- **Bramka volatility**: płatne porty (LLM, Alpha Vantage, embeddingi) **nie mogą**
  być wołane, gdy zmiana ceny < `volatility_threshold`. Logika progu żyje w domenie
  (`Asset.evaluate_volatility`) — graf jest tylko wykonawcą.
  Wyjątek (FIN-30): Slow Loop (recap tygodniowy, sędzia kalibracji, przyszłe joby
  diagnoz) woła LLM poza bramką, ale tylko w tygodniowym budżecie
  `slow_loop_max_paid_llm_calls_per_week` (config.toml: 15). Przydział liczy domena
  (`slow_loop_paid_budget`), wywołanie ponad przydział odrzuca `BudgetedLLMPort`
  przed dostawcą, a licznikiem jest ten sam `MeteredLLMPort` + `CycleTelemetry`
  co w Fast Loopie.
- **Cold-start**: snapshot ceny (`price_snapshots`) zapisuje węzeł TERMINALNY każdego
  ukończonego cyklu — `save_node` albo `ignore_node` (`_persist_price_snapshot`), nie
  `check_price_node` — żeby następny cykl miał punkt odniesienia. Cykl padły w połowie
  nie zostawia snapshotu, więc retry liczy Δ od ostatniego ukończonego cyklu (#6).
  Jedyny wyjątek (FIN-12): symbol odcięty sufitem płatnych wywołań cyklu
  (`max_paid_llm_calls_per_cycle`, węzeł `budget_exhausted`) kończy jako 'ignored'
  BEZ snapshotu ceny — inaczej następny cykl liczyłby Δ od nowej ceny i ruch ponad
  progiem zniknąłby bez analizy.
- **Self-Reflection**: `reflect_node` działa PRZED bramką volatility — ocena
  przeszłej predykcji jest niezależna od tego, czy bieżący cykl robi nową prognozę.
- **Resilience**: pojedynczy błąd per-symbol nie wywala cyklu. `main_agent.main()`
  zwraca exit 1 tylko gdy **wszystkie** symbole padły. Niedostarczenie raportu (mail
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
- Uprawnienia (od 038): nowa tabela = `ENABLE ROW LEVEL SECURITY` (bez polityk);
  nowa funkcja = `REVOKE EXECUTE ... FROM PUBLIC` i `GRANT EXECUTE ... TO service_role`
  w bloku `DO` z guardem `pg_roles`. Test emulacji ról w `test_migrations.py` pada,
  gdy któregoś brakuje; szczegóły w `docs/data-and-access.md`.

## Sekrety

- **Nigdy** nie commituj `.env`. Do `.env.example` wpisuj wyłącznie placeholdery.
- Klucze API w GitHub Actions: jako Repository Secrets (workflowy czytają `${{ secrets.* }}`).

## Konfiguracja: `config.toml` vs `.env` (DOC-30)

- **Konfiguracja niewrażliwa** (symbole, progi, flagi, modele, budżety, harmonogramy
  eksperymentów) żyje w commitowanym `config.toml` — jedyne źródło prawdy, wersjonowane
  i walidowane przez `Settings` (klucz spoza `Settings` = błąd startu). Nie dopisuj jej do
  `.env` ani do YAML-a workflowów.
- **`.env` i `.env.example`** zawierają wyłącznie sekrety (klucze API, tokeny, URL-e baz,
  adresy odbiorców) oraz przełączniki środowiska `DB_WRITES_ENABLED` i
  `PAID_CALLS_ENABLED`. W `.env.example` sekrety mają tylko placeholdery (`sk-...`,
  `re_...`, `https://twoj-projekt.supabase.co`, `you@example.com`); nowy sekret w `.env`
  = ten sam klucz z placeholderem w `.env.example` w tym samym kroku.
- Sygnał, że robisz to dobrze: `.env` i `.env.example` mają ten sam zestaw kluczy,
  a różnią się wyłącznie wartościami sekretów.

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
