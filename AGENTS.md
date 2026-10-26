# AGENTS.md — zasady pracy w repo StockAgent

Jedyne źródło zasad dla każdego agenta (Codex, Claude Code i innych) i dla ludzi (DOC-29).
`CLAUDE.md` importuje ten plik linią `@AGENTS.md` i dokłada wyłącznie dodatki samego
Claude Code. Nową regułę dopisuj tutaj, nigdy w kopii — dwie pełne wersje już raz się
rozjechały (G58); `tests/test_docs_consistency.py::test_claude_md_delegates_to_agents_md`
pilnuje, że żadna sekcja nie stoi w obu plikach.

## Stack i narzędzia

- **Python 3.12**, manager zależności: **`uv`** (nie `pip`!). Każda komenda przez `uv run ...`.
- Dodanie zależności: `uv add <pkg>` → commituj **oba** `pyproject.toml` + `uv.lock`.
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
Jednorazowo per klon: `uv tool install pre-commit && pre-commit install` — hooki: ruff check,
ruff format --check, `uv lock --check`, gitleaks, check-toml, check-yaml (CI-27); mypy i pytest
zostają w bramce.
Goły `# type: ignore` odrzuca mypy (`ignore-without-code`, CI-15); liczba `# type: ignore[...]`
i `cast(Any, …)` w `src/` i `main_*.py` tylko maleje — obniż `TYPE_IGNORE_BASELINE` /
`CAST_ANY_BASELINE` (`tests/test_quality_gate_sync.py`) w commicie, który usuwa ucieczkę.
Brak stubów wyciszamy tylko per moduł (`[[tool.mypy.overrides]]`: pandas, jsonschema, psycopg2,
docker, testcontainers); nowa biblioteka bez stubów = `types-*` albo wpis w override. Ruff ma
RUF100 (martwe `noqa`).
Workflow z `main_*.py` poza `ci.yml` synchronizuje `uv sync --frozen --no-dev` (CI-18);
narzędzie przez `uvx` w workflowie ma dokładną wersję (CI-23).
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
- **`main_agent.py` / `main_trainer.py` / `main_watch.py` / `main_deadman.py` / `main_tools.py` / `main_scorecard.py` / `main_bot.py`** — DI Container.
  Jedyne miejsce, gdzie konkretne adaptery są łączone z use case'ami. `main_watch.py` (R-11)
  i `main_deadman.py` (NIEZ-11) reużywają fabryk z `main_agent` i NIE WOŁAJĄ żadnego
  płatnego portu ani `save_price_snapshot`; watch zapisuje do bazy wyłącznie stemple swoich
  alarmów (`shock_alerts`, `price_level_alerts`, `prediction_logs.invalidation_alerted_at`).
  `main_scorecard.py` (WIA-52) tylko czyta bazę i zapisuje pliki strony (scorecard,
  `scorecard.json`, strona statusu NIEZ-33) — bez płatnych portów i bez zapisu do bazy.
  `main_bot.py` (MAIL-48) reużywa fabryk z `main_agent`, nie woła żadnego płatnego portu ani
  `save_price_snapshot`; zapisuje do bazy wyłącznie offset `bot_state` i wyciszenia
  `symbol_mutes`, odpowiada tylko czatowi `TELEGRAM_CHAT_ID` i nie pinguje heartbeatu.
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
Repozytorium Fast Loopa to wąski `PredictionStorePort` (Protocol w `ports.py`, ARCH-32) z
metodami wołanymi przez graf i `AnalyzeMarketUseCase`. Nowe wywołanie repozytorium w grafie =
metoda dopisana do Protocolu z sygnaturą identyczną z `RepositoryPort`
(`tests/application/test_prediction_store_port.py`). `LLMAdvisoryCouncil` to orkiestracja w
`src/application/council_service.py` (ARCH-19).

Analogicznie nowa sekcja maila = nowy moduł `src/application/report_<nazwa>.py`
(renderery HTML i tekstu + obiekt `ReportSection`), jeden wpis w `REPORT_SECTIONS`
(`application/reporting/registry.py`, kolejność maila) i jedno pole w `ReportContext`
(`application/report_builder.py`), które sekcja deklaruje w `context_fields`. Pole
czytane przez wspólne wejście sekcji (`SectionInput`) albo przez kilka sekcji trafia do
`SHARED_CONTEXT_FIELDS`; `tests/application/test_report_registry.py` pilnuje, że każde
pole ma dokładnie jednego konsumenta (ARCH-12). Nowa sekcja dopisuje też swój klucz do
`MAIL_ORDER` (ten sam plik testów) i wiersz z kolumną `Key` w tabeli sekcji
`docs/report-email.md` (`tests/test_docs_contract.py`). Kolory tekstu biorą się z `PALETTE`
(`report_formatting.py`) — `tests/application/test_report_a11y.py` liczy kontrast AA każdego
tekstu na jego tle. „Portfel” w nazwach sekcji to realne pozycje (`💼 Twój portfel`), liczby
z równych wag to „watchlista”; etykiety rekomendacji i prefiks tematu są w
`src/domain/polish_labels.py`. Pole z odczytów repozytorium wypełnia
`ReportAssembler.assemble` (`application/use_cases/assemble_report.py`) przez
`dataclasses.replace`; pole z adaptera (np. portfel właściciela) dokłada
`main_agent._report_inputs`. Zwarty mail cichego dnia (MAIL-35) = `COMPACT_REPORT_SECTIONS`;
stopka ze źródłami danych i warunkami jest w każdej postaci maila. `llm_budget_breach` jest w
`SHARED_CONTEXT_FIELDS` — czytają je banner limitów, sekcja kosztu i notka budżetu (FIN-36).
Błąd symbolu w mailu (MAIL-32): `SymbolResult.error_category` (`ErrorCategory`,
`application/errors.py`) i `error_provider` (`infrastructure/error_provider.provider_of`, hook
`error_provider` w `RunFastLoopCycle`); `error_message` jest tylko do logów. Nowy adapter
dostawcy = wpis w `PROVIDER_BY_MODULE`.

`main_agent.py` zostaje DI rootem Fast Loopa: mapuje `Settings` na opcje use case'ów
(`ReportFlags`, `CycleOptions`, `PublicScorecardOptions`), buduje adaptery i kanały
wysyłki i wstrzykuje hooki świata (`time.sleep`, `redact_secrets`, `symbol_context`).
Logika cyklu i raportu żyje w `src/application/use_cases/`: `run_cycle.py`
(`RunFastLoopCycle` — przebieg symboli z jednym `CycleOptions`), `assemble_report.py`
(`ReportAssembler` — zna tylko `RepositoryPort` i `ReportFlags`, nigdy `Settings`; okno
track recordu czyta raz na dispatch `TrackRecordReads`), `publish_public_scorecard.py` (strona
i `scorecard.json`, WIA-81), `publish_status_page.py` (strona statusu i `badge.json` z
`cycle_runs`, NIEZ-33) i `reveal_attestations.py` (`reconcile` na starcie runu stempluje
`revealed_at` tylko revealom obecnym w checkoucie `attestation-log`, `run` po raporcie tylko
publikuje — WIA-15); wszystkie wyłącznie na portach. Commitmenty cyklu zbiera
`CycleCommitmentCollector` (`application/attestation_cycle.py`), a `main_agent` po przebiegu
symboli publikuje jedną linię rootu Merkle'a na cykl (WIA-60). `main_agent.deliver_report`
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
- Stan grafu (`AgentState.delta`, `drift`) i `PriceDelta.fraction` to ułamki; procent do
  prezentacji daje `PriceDelta.percent` (alias `percentage` usunięty, ARCH-14).
- Pole z ułamkiem nosi typ `Ratio` i nazwę `*_ratio` (np. `actual_change_ratio`,
  `return_ratio`, `change_ratio`, `eur_pln_30d_change_ratio`); progi alertów to
  `elevated_threshold` / `critical_threshold` typu `Ratio`. Sufiks `_pct` zostaje tylko przy
  prawdziwym procencie (`price_delta_pct`, `min_pct` / `max_pct`, `improvement_pct`,
  `hit_rate_pct`) i w kontraktach poza kodem: klucze JSON narzędzia LLM `get_macro`
  (`eur_pln_30d_change_pct`, `usd_pln_30d_change_pct`) niosą ułamek, ale ich zmiana to
  zmiana przepisu predykcji (tylko za flagą).

## TDD — testy przed implementacją

Pracuj cyklem **Red → Green → Refactor**: najpierw test (który failuje), potem
minimalna implementacja, na końcu porządki. Każdy port ma testy z mockami,
każdy adapter ma testy jednostkowe (mock sieci) + opcjonalnie integracyjne.
Przed `scripts/merge.sh`, w worktree: `uv run python main_tools.py commit-policy` — każdy `fix:`
na gałęzi zmienia `tests/` albo idzie po commicie `test:` (DOC-27); `fix(ci):` i `fix(docs):`
są zwolnione.

## Konwencje testów

- Adaptery HTTP: testy mockują **`requests.Session.get` / `requests.Session.post`**
  (adaptery wołają przez `self._session` z `_http.build_session()`, nie przez gołe `requests`).
- Adapter z sekretem w URL-u albo nagłówku woła `safe_get` / `raise_for_status_redacted`
  z `_http.py`, a jego test sprawdza, że komunikat wyjątku nie zawiera sekretu.
- Testy integracyjne (prawdziwe API): oznaczone `@pytest.mark.integration`,
  pomijane bez kluczy w env. CI uruchamia `pytest -m "not integration and not containers"`,
  a testy kontenerowe osobnym krokiem `pytest -m containers`.
- Mocki portów: `Mock(spec=SomePort)`.
- Prawdziwe `Settings` w testach, nie `MagicMock`: `build_settings(**overrides)` z
  `tests/entrypoint_support.py` (rzuca na nieznane pole) albo `Settings(**NO_ENV_FILE, ...)`
  z `tests/settings_kwargs.py` — mypy ma plugin `pydantic.mypy` (`init_typed`,
  `init_forbid_extra`) i zna `_env_file`; `NO_ENV_FILE` zostaje wspólną konwencją testów.
  Wspólne fixture'y `settings` i `mock_external_clients` żyją w `tests/conftest.py`.
- Stan bota i wyciszenia to osobne porty `BotStatePort` i `SymbolMutePort` (adapter
  `bot_store.py`, nie `RepositoryPort`); kontrakt w
  `test_supabase_repo_postgrest.py::TestBotStoreContract`.
- Adapter Supabase buduje klienta przez `build_supabase_client` (`adapters/_supabase.py`, limit
  PostgREST 15 s, NIEZ-35); testy patchują `<moduł>.build_supabase_client`. Token Finnhub idzie
  nagłówkiem `X-Finnhub-Token` (`finnhub_auth_headers`, BEZ-15).
- Payloady dostawców: `tests/fixtures/providers/` (TEST-16). Nowy parser adaptera = plik w
  kształcie prawdziwej odpowiedzi (bez kluczy i treści newsów) + test w
  `tests/infrastructure/test_provider_shapes.py`.
- Nowa kolumna, RPC albo metoda `SupabaseRepository` = test kontraktowy w
  `tests/infrastructure/test_supabase_repo_postgrest.py` (marker `containers`, prawdziwy
  PostgREST z JWT, jeden zestaw kontenerów na moduł, TEST-17).
- `InMemoryRepository` (`src/infrastructure/adapters/in_memory_repo.py`) to fake `RepositoryPort`
  dla trybu offline (`main_tools.py offline`, DX-04). Nowa metoda abstrakcyjna `RepositoryPort`
  = implementacja także tam, inaczej padnie `tests/test_main_offline.py`.

## Reguły domenowe / FinOps

- **Bramka volatility**: płatne porty (LLM, Alpha Vantage, embeddingi) **nie mogą**
  być wołane, gdy zmiana ceny < `volatility_threshold`. Logika progu żyje w domenie
  (`Asset.evaluate_volatility`) — graf jest tylko wykonawcą.
  Wyjątek (FIN-30): Slow Loop (recap tygodniowy, diagnoza zbiorcza EXP-17, spłata diagnoz
  WIA-53, sędzia diagnoz EXP-10; kalibracja pewności jest arytmetyczna i nie woła LLM,
  WIA-65) woła LLM poza bramką, ale tylko w tygodniowym budżecie
  `slow_loop_max_paid_llm_calls_per_week` (config.toml: 15). Przydział liczy domena
  (`slow_loop_paid_budget`), wywołanie ponad przydział odrzuca `BudgetedLLMPort`
  przed dostawcą, a licznikiem jest ten sam `MeteredLLMPort` + `CycleTelemetry`
  co w Fast Loopie. Prompt joba Slow Loopa żyje w module jego use case'u, nie w
  `prompts.py` — bajty modułów promptów wchodzą do `prompt_sha256` przepisu.
  Twardy limit USD doby i miesiąca UTC (FIN-25, `daily_llm_budget_usd`,
  `monthly_llm_budget_usd`) czyta `llm_calls` raz przed grafem (fail-open) i ustawia
  `AgentGraphDeps.llm_budget_spent`. Koszt tygodnia (FIN-26) liczy domena
  `summarize_weekly_cost` bez LLM i przed progiem próbki recapu; przy pominiętym recapie Slow
  Loop wysyła sam rachunek tygodnia (za `cost_ledger_enabled` i `cycle_runs_enabled`).
- **Cold-start**: snapshot ceny (`price_snapshots`) zapisuje węzeł TERMINALNY każdego
  ukończonego cyklu — `save_node` albo `ignore_node` (`_persist_price_snapshot`), nie
  `check_price_node` — żeby następny cykl miał punkt odniesienia. Cykl padły w połowie
  nie zostawia snapshotu, więc retry liczy Δ od ostatniego ukończonego cyklu.
  Jedyny wyjątek (FIN-12): symbol odcięty sufitem płatnych wywołań cyklu
  (`max_paid_llm_calls_per_cycle`, węzeł `budget_exhausted`) kończy jako 'ignored'
  BEZ snapshotu ceny — inaczej następny cykl liczyłby Δ od nowej ceny i ruch ponad
  progiem zniknąłby bez analizy. Drugi wyjątek (DAT-15, za `price_outlier_guard_enabled`):
  odczyt ceny, który detektor MAD uznał za anomalię, a `pc` notowania jej nie potwierdził,
  kończy jako 'ignored' BEZ snapshotu i bez rozliczenia; stęchła referencja wyłącza strażnika.
  Wyczerpany limit USD (FIN-25) prowadzi każdy symbol za bramką do `budget_exhausted` — bez
  snapshotu, jak FIN-12. Symbol wyciszony z bota (MAIL-45, za `symbol_mutes_enabled`) nie
  idzie do grafu, a jego snapshot zapisuje `main_agent.snapshot_muted_symbols` z darmowego
  `/quote` — inaczej pierwszy cykl po wyciszeniu liczyłby Δ od ceny sprzed tygodni.
- **Self-Reflection**: `reflect_node` działa PRZED bramką volatility — ocena
  przeszłej predykcji jest niezależna od tego, czy bieżący cykl robi nową prognozę.
  Po podejrzanym splicie (DAT-23, za `split_detection_enabled`) reflect rozlicza ceną
  w jednostkach predykcji (`to_pre_split_price`), a bramka liczy ruch od skorygowanej referencji.
- **Resilience**: pojedynczy błąd per-symbol nie wywala cyklu. Entry-pointy per symbol
  (`main_agent`, `main_trainer`, `main_watch`, `main_backtest`) zwracają exit 1 tylko gdy
  **wszystkie** symbole padły (`src/domain/exit_policy.run_exit_code`, NIEZ-38); częściowa
  porażka to exit 0 i WARNING `PARTIAL_FAILURE_LOG`, w Slow Loopie także sam padnięty
  model wspólny. Niedostarczenie raportu (mail
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
  (NIEZ-31). Adapter HTTP woła `translated_get`, `raise_for_status_translated` i
  `json_payload` z `_http.py`, adapter SDK LLM — `as_port_error(exc, sdk_error_kind(exc))`;
  każda publiczna metoda `SupabaseRepository` tłumaczy `APIError` i błędy httpx dekoratorem
  klasy `_translates_public_methods`. Pętla ponowień i alert kwot adapterów LLM to
  `llm/_retry.py` (`call_with_retry`, `RetryPolicy`) i `llm/_quota.py` (`QuotaEmitter`,
  ARCH-29). Sesja `build_session()` daje żądaniu bez `timeout=` 30 s (TEST-31).
- **Paper ledger (WIA-59)**: `paper_trades` jest append-only — save otwiera pozycję, reflect
  zamyka ją raz. `row_hash = sha256(canonicalize(wejście) + prev_hash)` liczy domena
  (`paper_ledger.py`); baza pilnuje liniowości i jednorazowego zamknięcia. Zmiana
  kanonizacji wpisu = nowa `LEDGER_VERSION`. Błąd zapisu ledgera tylko loguje (`PortError`).
- **Champion/challenger (ML-33)**: prognoza challengera (`ml_challengers`, wagi z Release
  `model-latest`) trafia wyłącznie do `shadow_predictions` i nie może zmienić ceny, promptu,
  rady ani rekordu predykcji. Regułę promocji ma domena (`should_promote`), Slow Loop tylko
  raportuje werdykt — podmiana serwowanych wag należy do właściciela.
- **Trening ML**: jedyną regułą, które wiersze trenują fold walk-forward, jest
  `purge_training_indices` (`domain/backtest.py`, ML-26). Każdy trening niesie
  `TrainingProvenance` (odcisk zbioru, parametry z jawnym `random_state`, wersja XGBoost, SHA
  kodu), zapis wag pisze sidecar `.json`, a `main_tools.py reproduce-training` sprawdza
  odtworzenie (ML-22). Granice foldów i ich wiersze treningu projektuje domena
  (`walk_forward_folds`, `expanding_window_bounds`, ARCH-27); `MLPredictionPort.train` i
  `walk_forward_report` przyjmują gotowe `folds` oraz wiersze cech bez DataFrame (ML-55), a
  `BacktestUseCase` liczy foldy sam.
- **Track Record (WIA-82, WIA-68)**: wiersz reguły referencyjnej to wyłącznie agregat
  `get_reference_outcomes` w prywatnym mailu; scorecard publiczny jest bez niego do zgody
  Finnhub i CoinGecko. PSR/DSR od 30 okresów krzywej, n_trials = `count_recipe_versions` za
  `recipe_version_enabled`.
- **Shadow mode wariantu promptu/modelu (EXP-26)**: wariant (`shadow_prompt_variant`, flaga
  `shadow_prompt_variant_enabled`, migracja 083) wołany tylko dla stałej podpróbki
  (`in_shadow_sample`) za bramką; wynik wyłącznie do `shadow_predictions`
  (`save_shadow_trend_prediction`). Wywołanie liczy się pod `shadow_llm` (poza
  `PAID_LLM_SOURCES`) i idzie tylko w luzie sufitu cyklu (`shadow_call_allowed`,
  `CycleTelemetry.reserve_paid_call`), twardo ≤ 5 na cykl — nigdy nie tnie symbolu produkcji.
  Znaczniki wyniku `LLMPort.analyze` (`LLM_ROUTED_MODEL_KEY`, `LLM_ESCALATION_FAILED_KEY` i
  tożsamość dostawcy) zdejmuje graf przed normalizacją; routing daje `model_used` w kwicie v2
  i flagę jakości `escalation_failed`, a eskalacja frontiera liczy się pod `llm` (FIN-21).
- **Rundy rady (ML-40)**: `council_votes.round` (082) — 1 = niezależny głos, 2 = rewizja
  debaty; trafność person (RPC i odczyty z Pythona od 082) liczy rundę 1.
- **Persony publicznie (MAIL-38)**: nazwiska tylko w mailu, publiczne wyjścia biorą archetyp z
  `data/council_personas/archetypes.toml` (poza hashem przepisu); strażnik
  `tests/test_public_persona_guard.py`.
- **Sędzia diagnoz (EXP-10)**: `diagnosis_judge_enabled` sam tylko ocenia (tryb cienia,
  `diagnosis_judge_score`); odrzucenie (`rejected_diagnosis` i placeholder) wymaga
  `diagnosis_judge_rejects_enabled`, które właściciel włącza po kalibracji na 20 własnych ocenach.
- **Rekor (WIA-73)**: `RekorAttestationPublisher` owija publisher pliku (jedno pole
  `attestation_publisher`); root cyklu idzie jako `hashedrekord` z SHA-256 rootu
  (`root_artifact_sha256`), wpis Rekora trafia do pola `rekor` linii ledgera, awaria daje tylko
  WARNING. `id-token: write` dostaje wyłącznie job, który podpisuje.
- **Archiwum digestów (MAIL-24)**: `DigestArchivePort` i prywatny bucket Supabase Storage
  `digests`; podpisany link (TTL 7 dni) dostaje wyłącznie mail i push właściciela (BEZ-01).

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
- Flaga zależna od migracji ma wpis w `FLAG_REQUIRED_MIGRATION`
  (`src/domain/feature_requirements.py`); Fast Loop, Slow Loop i watch wyłączają na czas
  przebiegu flagi, których migracji baza jeszcze nie ma (`schema_version()`, migracja 071).
  Mapę z kolumną Migration tabeli flag w `docs/configuration.md` pilnuje
  `tests/test_feature_requirements_wiring.py`. Nowa migracja = też wiersz w
  `MIGRATION_PROBES` w `scripts/prod-state-check.sh` (`tests/test_prod_state_probes.py`).
- Nowa tabela albo widok = wiersz w tabeli Relations w `docs/data-and-access.md`
  (`tests/test_data_and_access_relations.py`, DOC-37, bez Dockera). Runbook odtworzenia kopii
  dowodów (`docs/running-locally.md`) wykonuje test kontenerowy `TestEvidenceRestoreDrill`
  (NIEZ-48) — zmiana kroków runbooka = zmiana testu.

## Sekrety

- **Nigdy** nie commituj `.env`. Do `.env.example` wpisuj wyłącznie placeholdery.
- Klucze API w GitHub Actions: jako Repository Secrets (workflowy czytają `${{ secrets.* }}`).
  Wyjątek: `SUPABASE_DB_URL` to sekret środowiska `production-db` (BEZ-22), nie repo;
  `scripts/gh-secrets-*.sh` znają go z `ENVIRONMENT_SECRETS`.
- Pozycje, P/L i adresy e-mail nigdy nie trafiają do promptu ani embeddingu. Inwentarz jest w
  `docs/data-flows.md`, strażnik w `tests/test_llm_private_data_guard.py` (BEZ-32). Nowe pole
  promptu = wiersz w tabeli `docs/data-flows.md`.
- Nowy sekret (w workflowie albo w `.env.example`) = wiersz w tabeli „Secrets inventory and
  rotation” w `SECURITY.md` (rola, konto, rotacja, bez wartości i prefiksu klucza);
  pilnuje tego `tests/test_secrets_inventory.py` (BEZ-24).

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

## Dziennik decyzji (DOC-40)

- Pozycja roadmapy ma numer `R-N`, definiowany wyłącznie w `docs/decisions.md`. W kodzie,
  `config.toml` i docs nie pisz „roadmap #N”; `tests/test_decision_log.py` pilnuje, że każde
  `R-N` ma wiersz.
- Komentarz ani docstring w `src/`, `main_*.py` i `tests/` nie niesie gołego „#N” (DOC-53,
  DOC-55): pozycja roadmapy to `R-N`, znalezisko audytu z 2026-10-15 — jego identyfikator z
  audytu (`G-N`, zapisywany jak `G03`), a odwołanie spoza obu spisów (np. znalezisko pierwszego
  audytu z czerwca 2026) opisujesz słowami. Numer, który nie jest odwołaniem (kolor, przykład
  dawnej numeracji), trafia do `NOT_A_REFERENCE` z powodem. Pilnują
  `tests/test_docs_consistency.py::test_no_bare_hash_reference_in_src_comments` i
  `::test_no_bare_hash_reference_in_entrypoints_and_tests`.
- Nowa decyzja architektoniczna = wpis `D-N` w `docs/decisions.md` (data, status, hashe
  commitów, strażnicy). Nowy incydent = wiersz w `docs/engineering-log.md` (incydent → reguła →
  test).
- Uzasadnienie flagi (dlaczego, historia) = wpis w sekcji `Flag rationale` w
  `docs/decisions.md` z kotwicą `<a id="flag-…"></a>`; `docs/configuration.md` trzyma tylko
  zwięzły opis (co robi, kiedy włączyć, czego wymaga) i link `decisions.md#flag-…`
  (`tests/test_flag_rationale.py`, DOC-19).

## Język

- Komentarze w kodzie i treść raportu mailowego: **polski**.
- `README.md` i `SECURITY.md`: **angielski**.
- Identyfikatory (klasy, funkcje, zmienne): angielski.

## Znane kompromisy

- `agent_graph.py` nie ma `# type: ignore`: każdy węzeł, także `fetch_fundamentals`,
  rejestrujemy lambdą (`lambda s: node(s)`). Goła funkcja węzła w `workflow.add_node` daje
  błąd arg-type stubów LangGraph 1.x (`_Node[Never]`) (ARCH-20).

## Lokalne koncepcje

- Propozycje nowych funkcji, plany, audyty i notatki badawcze trzymaj w lokalnym
  `docs/concepts/`, wykluczonym z Gita. Utrzymywana dokumentacja techniczna i przyjęte
  decyzje architektoniczne zostają śledzone w pozostałej części `docs/`.
