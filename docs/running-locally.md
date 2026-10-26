# Running and shipping it

Stack, prerequisites, local setup, and CI. Tunables live in [configuration.md](configuration.md).

## What you need before you see anything

The fastest look at the email skips the loop: `uv run python main_tools.py render-sample-report` writes the email of a fictional cycle to `build/sample-report/` with no account, key, database or network (DX-01, see [Build commands](#build-commands)). To watch the loop itself (gate, self-reflection, council, save and the email), `uv run python main_tools.py offline` runs two whole cycles on in-memory fakes, equally without accounts (DX-04, see [Offline cycle](#offline-cycle-dx-04)). A real cycle needs accounts first (DX-02):

| Account | What it is for | With the committed `config.toml` |
|---|---|---|
| Supabase (free plan) | The database every entry point reads and writes. The [Supabase CLI](https://supabase.com/docs/guides/cli) applies the 70+ migrations in `supabase/migrations/` (step 3 below) | required |
| Finnhub (free) | The quote behind every cycle | required |
| Alpha Vantage (free, ideally two or more keys) | News and sentiment for each symbol past the volatility gate | required with `PAID_CALLS_ENABLED=true` |
| OpenAI | The advisory council and the news embeddings | required with `PAID_CALLS_ENABLED=true` |
| Anthropic | The main analysis (`llm_provider = "anthropic"`) | required unless `llm_provider = "openai"` |
| Resend | The email (`notifications_enabled = true`) | required to receive the email; without it the cycle runs and logs, but nothing is delivered |

That is six accounts with the committed configuration and five at the least (`llm_provider = "openai"`). Without `PAID_CALLS_ENABLED=true` a local run builds stubs in place of Alpha Vantage, OpenAI and Anthropic ([below](#local-runs-make-no-paid-calls-by-default)), so Supabase and Finnhub are enough to watch the pipeline run, but every prediction is then a neutral stub. Once the accounts exist, expect this:

1. **First run: no signals.** A new deployment has no previous price, so every symbol is a cold start and the email opens with `👋 Dzień 1 — buduję punkt odniesienia` under its own welcome subject ([the report email](report-email.md)).
2. **Second run: the first signals**, only for symbols that moved at least `volatility_threshold` (|Δ| ≥ 2%, crypto 5%) since the first run. A quiet day still has none, and costs nothing.
3. **The cycle after a prediction: its settlement.** A prediction is scored by the next cycle once it is at least `reflection_min_age_hours` (6 h) old, and then appears under `📊 Zamknięte predykcje`.
4. **The first Sunday recap** (`📅 Tydzień StockAgenta`) needs at least 5 predictions closed in the week (`MIN_CLOSED_PREDICTIONS_FOR_RECAP` in `src/domain/weekly_recap.py`); with fewer, no recap email goes out. With `cost_ledger_enabled` and `cycle_runs_enabled` that week still gets the `💸 Koszt tygodnia StockAgenta` email with the cost of the week (FIN-26).
5. **The XGBoost leg** stays on the no-change baseline, labelled cold-start, until the Slow Loop has at least 50 settled predictions and a candidate passes the walk-forward gate ([how it works](how-it-works.md), step 10).

## Tech stack

- **Python 3.12** with `from __future__ import annotations`
- **[uv](https://github.com/astral-sh/uv)** (Astral) — dependency manager. The CI gate syncs with `uv sync --locked`, which fails when `uv.lock` no longer matches `pyproject.toml` (a dependency changed without `uv lock`, CI-11); the production workflows install the committed lock with `--frozen`
- **[LangGraph](https://langchain-ai.github.io/langgraph/) 1.2** — decision graph with `StateGraph[AgentState]`
- **OpenAI Python SDK** v2 — `gpt-5-mini` advisory council + `text-embedding-3-small` embeddings, JSON mode
- **Anthropic Python SDK** (core dependency) — Claude Sonnet 4.6, **default model for the main analysis**
- **XGBoost 3.2** + scikit-learn — sklearn-style API, native `.ubj` (UBJSON) format
- **Supabase** (Postgres 16 + pgvector) — REST via `supabase-py`, secret / service_role key
- **Resend.com** — HTML email, sandbox sender without domain verification
- **Pydantic Settings** v2 — typed env vars from `.env` with validators (CSV → list)
- **requests** + `urllib3.Retry` — shared session with exponential backoff on 429 / 5xx
- **pytest 9** + **pytest-mock** — 6,000+ test functions (`def test_`, counted by `uv run python main_tools.py repo-stats` into [`docs/repo-stats.json`](repo-stats.json), DOC-18) + skipped live-API / Docker-container tests
- **ruff** — lint (E, W, F, I, B, UP, SIM rule sets)
- **mypy** strict mode — `src/` and every entry point fully typed; `tests/` not yet (it joins the gate directory by directory, `tests/domain` so far)
- **GitHub Actions CI** — ruff + mypy + pytest + SonarCloud scan on every push / PR

Architecture: **Hexagonal (Ports & Adapters) + DDD**. Domain (pure Python, zero deps) → Application (ports + use cases + LangGraph) → Infrastructure (adapters for API / DB / LLM / ML / email). Dependency arrow flows one way (`domain ← application ← infrastructure`).

## Prerequisites

- Python 3.12 + `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- macOS only: `brew install libomp` (required by XGBoost)
- **Finnhub** API key (free) — https://finnhub.io/dashboard
- **Alpha Vantage** API key(s), ideally several (free 25 req/day × N) — https://www.alphavantage.co/support/#api-key
- **OpenAI** API key — https://platform.openai.com/api-keys
- **Supabase** project (free tier) — https://supabase.com/dashboard
- **Supabase CLI** (`supabase`) — applies the migrations with `supabase db push` (step 3 below); CI pins version `2.121.0` in `db_migrate.yml` — https://supabase.com/docs/guides/cli
- **`SUPABASE_DB_URL`** — the Postgres connection string of the project (the session pooler, `postgres.<ref>` role), used only by `supabase db push --db-url` and the DB Migrate workflow; the agent itself reads `SUPABASE_URL` and `SUPABASE_KEY` and never needs it
- **Resend** account + API key — https://resend.com (optional, for emails)
- **Anthropic API key — required** (default `LLM_PROVIDER=anthropic`; main analysis runs on Claude)

## Local setup

```bash
git clone <repo-url> StockAgent && cd StockAgent

# 1. Sync deps (creates .venv and installs the locked packages, Anthropic SDK included)
uv sync

# 2. Configuration
cp .env.example .env
# fill in the keys (see the ".env keys" block below)

# 3. Database schema — applied by the Supabase CLI, not by hand
#    Migrations live in supabase/migrations/ and are applied in filename order.
#    A ledger table (supabase_migrations.schema_migrations) records what ran, so
#    re-running is a no-op rather than a rebuild. --db-url skips `login`/`link`,
#    so the only secret is the Postgres connection string (TLS is enforced).
#
#    supabase db push --db-url "$SUPABASE_DB_URL" --dry-run   # show, change nothing
#    supabase db push --db-url "$SUPABASE_DB_URL"              # apply
#
#    In CI: run the "🗄️ DB Migrate (manual)" workflow from the Actions tab. It is
#    deliberately workflow_dispatch-only — a broken migration must not fire on the
#    Fast Loop cron, and 022 rebuilds the ml_feature_store materialized view.
#
#    The migrations are the numbered files of supabase/migrations/ (link under this
#    block); the list is not repeated here because it kept going stale.

# ⚠️ Migration 012 is BREAKING for the ML model: it changes the training target
#    from absolute price to a 12h return, and the code now reconstructs price from
#    that return. A .ubj trained on the old (absolute-price) contract is INCOMPATIBLE.
#    After applying 012, DELETE the old model weights (data/models/price_predictor.ubj
#    if present in your deployment) — the Fast Loop falls back to the cold-start
#    "no change" baseline (= return 0) until the next Slow Loop retrains from scratch.

# ⚠️ Migration 038 turns on Row Level Security (no policies) on every public table
#    and revokes the anon / authenticated grants on tables, views, sequences and
#    functions. Afterwards only the secret key (sb_secret_…) or the legacy
#    service_role key can read or write. Put that key into GitHub Secrets and .env
#    BEFORE you run the migration (rollout order: docs/data-and-access.md).

# 4. Smoke test — the same selection as CI; bare `uv run pytest` would also run the
#    live-API tests, which call paid APIs and send a real email when keys are exported
uv run pytest -m "not integration and not containers"

# 5. Single Fast Loop run — read-only against the shared database unless
#    DB_WRITES_ENABLED=true, and without paid calls unless PAID_CALLS_ENABLED=true
#    (see "Local runs are read-only by default" and "Local runs make no paid calls" below)
uv run python main_agent.py
```

**The migrations.** The schema is the set of numbered files in [`supabase/migrations/`](../supabase/migrations/) (`NNN_name.sql`, zero-padded, applied in filename order by `supabase db push`); the directory is the list, and [data-and-access.md](data-and-access.md) says which table, view and function each one added and who may touch it. Read the `⚠️` notes above before applying `012` and `038`, which change what the running code expects.

Config is split in two: **secrets** live in `.env` (gitignored) / GitHub Secrets; **everything non-secret** (symbols, thresholds, models, providers, throttle, Risk Watch…) lives in committed [`config.toml`](../config.toml) — the single source of truth, read directly by `Settings`. No duplication across `.env` / workflow / repo variables.

`.env` — **secrets only**:

```env
# LLM (main analysis = Anthropic; embeddings + council = OpenAI)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...            # main analysis + weekly recap (llm_provider = anthropic)

# Market data + News/Sentiment
FINNHUB_API_KEY=...
ALPHA_VANTAGE_API_KEYS=key1,key2,key3   # CSV — rotation on rate-limit

# Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=sb_secret_...              # secret key (or the legacy service_role JWT)
SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres  # DB Migrate only

# Email — recipient is a secret; sender + on/off toggle live in config.toml
RESEND_API_KEY=re_...
DIGEST_TO_EMAIL=you@example.com
```

`SUPABASE_KEY` must be the project's secret key (`sb_secret_…`, Supabase dashboard → Project Settings → API Keys) or the legacy `service_role` JWT, because migration `038` turns on Row Level Security and the app relies on that key bypassing it. `Settings` refuses to start with a public key (BEZ-10): a publishable key (`sb_publishable_…`) or a JWT whose `role` claim is `anon` or `authenticated` stops every entry point with a validation error that names the fix and does not print the key. The check reads the `role` claim without verifying the signature and lets any other string through. [Data and access](data-and-access.md) gives the rollout order: key first, then the deploy, then migration `038`.

`Settings` does not require the paid providers' keys (ARCH-05). The factory that builds an adapter asks for its key (`require_openai_api_key`, `require_finnhub_api_key`, `require_alpha_vantage_api_keys`), so a missing key stops only an entry point that would call that provider, with a `ValueError` naming the variable: the Fast Loop needs all three, the Slow Loop needs Alpha Vantage and its LLM key when paid calls are on, Shock Watch needs Finnhub only and the deadman none.

Everything else — `symbols`, `symbols_etf`, `volatility_threshold`, `crypto_symbols`, `council_llm_provider/model`, `risk_symbols`, `nbp_enabled`, `symbol_throttle_seconds`, `notifications_enabled`, `digest_from_email`, … — is edited in [`config.toml`](../config.toml). Any value can still be overridden by an environment variable (precedence: **env → `.env` → `config.toml` → code defaults**), e.g. `SYMBOL_THROTTLE_SECONDS=0` for a quick local run. `.env` is read from the repository root, like `config.toml`, whatever the working directory a job starts in (ARCH-23).

The secrets feed **GitHub Actions Secrets**, and a local run uses the **same Supabase project** as CI and production. That is a choice with a cost, not a parity guarantee: a local run reads production with the production secret key, so the only thing between it and the track record is the write guard described next (DX-09). `config.toml` is committed, so CI gets the same config without any repo variables.

### Local runs are read-only by default

Because a local run and production share one database, every local cycle used to write into the production track record: a price snapshot per symbol, a prediction that overwrote the one production saved in the same hour (the write is an upsert on `symbol, timestamp_hour`), settlements of production predictions, quota alerts and the regime history. `Settings.db_writes_enabled` now defaults to `false`, and with it off `main_agent.py`, `main_trainer.py`, `main_watch.py` and `main_tools.py` are dry runs against the shared database (BEZ-08; `main_deadman.py` only reads anyway, while `main_backtest.py` still refreshes `ml_feature_store` before its folds):

- reads go to Supabase unchanged, so the email, the Track Record and the RAG analogues look like production;
- every write method of the repository is skipped with one `WARNING` per method per run (`DB_WRITES_ENABLED=false — SupabaseRepository.save_price_snapshot pominięty …`), and the next calls of the same method log at `DEBUG` only;
- a saved prediction gets the id `dry-run-<uuid>`, so the council votes and the commitment that depend on it cannot be mistaken for a database row;
- `refresh_feature_store` counts as a write (it refreshes a view in the shared database), so a local `main_trainer.py` trains on the view as production last refreshed it;
- the `fx_rates` cache is read but not filled, and attestation commitments and reveals are not published (a commitment of a `dry-run-…` prediction would be a false public entry);
- `main_tools.py import-positions --apply` prints the diff and then refuses with exit code 1 and "nic nie zapisano"; the dry run without `--apply` is unchanged.

The wrappers live in [`src/infrastructure/adapters/write_guard.py`](../src/infrastructure/adapters/write_guard.py); every abstract method of a guarded port is classified there as a read or a write, and a test fails when a new port method is not. The paid calls have their own switch, below.

To write from a local run on purpose (for example a one-off positions import), set the variable for that one command:

```bash
DB_WRITES_ENABLED=true uv run python main_tools.py import-positions --csv positions.csv --apply
```

The switch is read from the environment or `.env` only. It is deliberately not in `config.toml`: that file is shared by local runs and GitHub Actions, so turning writes on there would turn them on for every local run (a test keeps it out). The workflows that write — Fast Loop, Slow Loop and Shock Watch — set `DB_WRITES_ENABLED: "true"` in the step that runs the entry point; the read-only deadman does not. `tests/test_db_writes_wiring.py` checks both, and fails when a new `main_*.py` is not classified as writing or read-only.

**Why the guard and not a separate database (DX-09).** Full isolation would be a second Supabase project for development or a local `supabase start`. Neither is used; the decision is the write guard alone: `DB_WRITES_ENABLED=false` by default and the `ReadOnlyRepository` wrapper in `write_guard.py`. The reason is the machine. A local Supabase stack runs several containers next to the Docker containers of the container tests on a computer that is already short of memory, and a second cloud project would need its own migrations, keys and seed data kept in step with production. The price of the choice is what the guard cannot remove: a local `.env` holds the production secret key, so a write method that escaped the classification would reach production, which is why `write_guard.py` classifies every abstract method of the port and a test fails when a new one is not classified. Reads stay production-like (the email, the Track Record and the RAG analogues look as they do in production), which is the part of the shared database that is worth keeping.

### Local runs make no paid calls by default

A read-only local run still spent the shared Alpha Vantage quota (25 requests a day per key) and LLM credit without a trace in the database. `Settings.paid_calls_enabled` defaults to `false` (FIN-35), and with it off the entry points build deterministic stubs instead of the paid adapters, which are never constructed:

- **Fast Loop** (`main_agent.py`): the main LLM is `LocalStubLLMAdapter` (a neutral `SIDEWAYS`, confidence 50% analysis whose reasoning says it is a stub, and a fixed mistake diagnosis); Alpha Vantage sentiment and news are `LocalStubSentimentAdapter` (a neutral payload with `degraded_reason: paid_calls_disabled`, so the prediction carries that data-quality flag and the email a DEGRADED badge) and `LocalStubNewsAdapter` (no articles); the council, the embeddings and tool use are off, so there is no council verdict and no RAG analogue; the earnings calendar is the null adapter. The volatility gate, the ML model and the free sources that `config.toml` enables (Finnhub, CoinGecko, NBP, FRED; EDGAR and Reddit are off, see [configuration](configuration.md#feature-flags)) run as usual, and the email hides the cycle-cost section, because the stubbed calls are counted like real ones;
- **Slow Loop** (`main_trainer.py`): the Alpha Vantage fundamentals refresh is skipped, and the weekly recap and the diagnosis backfill get the LLM stub: the recap goes out without a narrative (the calibration score makes no LLM call, WIA-65);
- each run logs one `WARNING`: `PAID_CALLS_ENABLED=false — lokalnie bez płatnych wywołań: …`.

Replaying recorded responses is out of scope; the stubs are fixed values. To make paid calls from a local run on purpose, set the variable for that one command:

```bash
PAID_CALLS_ENABLED=true uv run python main_agent.py
```

`PAID_CALLS_ENABLED=false` together with `DB_WRITES_ENABLED=true` is refused: `main_agent.py` and `main_trainer.py` log an `ERROR` naming both variables and exit with code 2 before they build any adapter, because the stubs' neutral predictions would land in the production track record (code 1 stays "every symbol failed"). The check lives in the entry points, not in `Settings`; Shock Watch and the deadman make no paid call and start as usual. Like `DB_WRITES_ENABLED`, the switch is read from the environment or `.env` only and is kept out of `config.toml` by a test. The workflows that pay — Fast Loop and Slow Loop — set `PAID_CALLS_ENABLED: "true"` in the step that runs the entry point; Shock Watch and the deadman make no paid call and do not set it. `tests/test_paid_calls_wiring.py` checks both and fails when a new `main_*.py` is not classified as paying or free.

### Trial runs of the sections that are off (PORT-15, EXP-09)

Some features are built, rendered only and off in `config.toml`, and the owner judges them on a real cycle before any default changes. The decision for these trials is a **local, read-only run**: `DB_WRITES_ENABLED=false` and `PAID_CALLS_ENABLED=false`, the flags switched on for that one command only, the email written to a file instead of being sent. Nothing is committed, nothing is written to the database, no push goes out and no paid API is called.

**How a flag is overridden for one run.** Every `Settings` field can be set by an environment variable named like the field in capitals (precedence: env → `.env` → `config.toml` → code defaults, see above), so `PORTFOLIO_VAR_ENABLED=true uv run python main_agent.py` turns `portfolio_var_enabled` on for that process and leaves `config.toml` as it is.

**How the email gets to a file.** `main_agent.py` always sends through the channels that `build_notifier` builds from the settings, and has no option to write the email to disk. The offline cycle uses `FileReportNotifier` ([`src/infrastructure/adapters/file_notifier.py`](../src/infrastructure/adapters/file_notifier.py)) for that, and the same notifier can stand in for the real channel: the function below replaces `build_notifier` for the one process, exactly as `tests/test_main_agent.py` does (`TestNyseClosedDay`), and then runs the real `main_agent.main()`. Paste it into the shell once. It also blanks the secrets of every other outgoing channel: `HEARTBEAT_URL` (otherwise the trial would ping the production heartbeat and hide a missed cron), Telegram, Slack and Resend.

```bash
trial_run() {   # usage: TRIAL_REPORT_DIR=<dir> [FLAG_NAME=true ...] trial_run
  DB_WRITES_ENABLED=false PAID_CALLS_ENABLED=false \
  HEARTBEAT_URL= TELEGRAM_BOT_TOKEN= TELEGRAM_CHAT_ID= SLACK_WEBHOOK_URL= RESEND_API_KEY= \
  uv run python - <<'PY'
import os
from pathlib import Path
from unittest.mock import patch

import main_agent
from src.infrastructure.adapters.file_notifier import FileReportNotifier

notifier = FileReportNotifier(Path(os.environ["TRIAL_REPORT_DIR"]))
with patch.object(main_agent, "build_notifier", return_value=notifier):
    raise SystemExit(main_agent.main())
PY
}
```

The run needs the `.env` of the owner (`SUPABASE_URL`, `SUPABASE_KEY`, `FINNHUB_API_KEY`), reads the shared database and the free Finnhub, CoinGecko, NBP and FRED sources as any local run does, and writes `report-email.html` and `report-email.txt` (the latter starts with `Temat:` and the subject) into `TRIAL_REPORT_DIR`. `build/` is git-ignored; the file carries the owner's positions, so keep it out of the repository and out of anything shared. Run it on a NYSE trading day: on a weekend or holiday the Fast Loop analyses only the crypto symbols (FIN-34) and the stock-based sections have little to show. With paid calls off the analysis is the fixed stub (neutral `SIDEWAYS`, 50%) and the email says so, which is fine for judging the layout and the free numbers below, not the quality of a forecast. Combining `PAID_CALLS_ENABLED=false` with `DB_WRITES_ENABLED=true` is refused, so a mistake in the command cannot put stub predictions into the track record.

**PORT-15: the risk sections** (`portfolio_var_enabled`, `stress_test_enabled`, `hedge_effectiveness_enabled`). A baseline with the three flags off and the same data, then the run with them on:

```bash
mkdir -p build/trial-runs
TRIAL_REPORT_DIR=build/trial-runs/port-15-baseline trial_run
TRIAL_REPORT_DIR=build/trial-runs/port-15-risk \
  PORTFOLIO_VAR_ENABLED=true STRESS_TEST_ENABLED=true HEDGE_EFFECTIVENESS_ENABLED=true trial_run
wc -c build/trial-runs/port-15-*/report-email.html
diff build/trial-runs/port-15-baseline/report-email.txt build/trial-runs/port-15-risk/report-email.txt
open build/trial-runs/port-15-risk/report-email.html
```

The flags add to two sections of [the report email](report-email.md): VaR/CVaR and the stress test in `🎯 Portfolio Radar`, and the hedges in `🚨 Risk Watch`. What to judge: the length the three add to the email (`wc -c`, the `diff` of the text versions, the number of sections); whether the numbers are plausible; whether the Radar's VaR/CVaR and stress test, which describe the **watchlist at equal weights** (the text says "watchlisty (równe wagi)"), not what you hold, are worth their space; and whether the three belong in one "portfolio risk" section. With `portfolio_positions_enabled` (on in `config.toml`) and positions loaded, `portfolio_var_enabled` also measures the VaR of the positions you actually hold, but that number is not printed: it only lowers the size band of a signal one step when the 1-day VaR95 is at least 3% (see `portfolio_positions_enabled` in [configuration](configuration.md#feature-flags)), so look in the `diff` for a band that moved. A statistic whose `price_snapshots` history is too short is replaced by a "za mało historii (n/wymagane)" note, which is a data finding, not a bug.

**EXP-09: contagion** (`contagion_enabled`). With the flag on, the prompt carries the verdicts of the same-group tickers from this cycle, so the symbols have to be processed one after another and `symbol_concurrency` (4 in `config.toml`) is ignored. Time the run with and without it:

```bash
mkdir -p build/trial-runs
time TRIAL_REPORT_DIR=build/trial-runs/exp-09-baseline trial_run
time TRIAL_REPORT_DIR=build/trial-runs/exp-09-contagion CONTAGION_ENABLED=true trial_run
```

The difference in wall time is the cost of the sequential path. Two limits of a stub run: the verdicts that contagion hands to the prompt are all the same neutral stub, so the effect of contagion on the decisions cannot be seen (only that the block is built and what the sequential path costs), and the stub makes no LLM or council request, so the real cycle, where every symbol past the gate waits for the model one after another, would be slower than this measurement. The first symbol of each group also sees no peers. Contagion is only as good as the order of processing: reading the peers' verdicts from the previous cycle instead (so the symbols could run in parallel again) is not built; it is the follow-up (ML-43) that this measurement is meant to justify or drop.

### Offline cycle (DX-04)

`uv run python main_tools.py offline` runs whole Fast Loop cycles with no key, no database and no network: the real graph (check price → reflect → fundamentals → volatility gate → sentiment → news → predict → council → save), `RunFastLoopCycle` and `ReportAssembler`, with fakes in place of the outside world:

- `InMemoryRepository` ([`src/infrastructure/adapters/in_memory_repo.py`](../src/infrastructure/adapters/in_memory_repo.py)) implements every method of `RepositoryPort` on dictionaries: a prediction and a price snapshot are upserts on symbol and UTC hour, the settlement queue is oldest-first with the minimum age, a settlement is written once, and the rows are parsed by the same functions as in `SupabaseRepository`. Reads computed in the database (persona weights and rankings, pgvector analogues, the `ml_feature_store` view) return nothing, as the adapter does before their migration, so those email sections hide;
- `FixtureMarketData` serves each cycle's prices from [`data/offline/scenario.json`](../data/offline/scenario.json): start time, interval, symbols and asset classes, thresholds, the reference prices of the "previous cycle" (without them the first cycle would be a cold start with every symbol under the gate) and one price map per cycle;
- `ScriptedLLM` answers the prediction, every council persona, the council chairman and the mistake diagnosis from [`data/offline/llm_script.json`](../data/offline/llm_script.json): the first rule whose `when` markers all appear in the prompt replies, and a prompt no rule matches fails like a malformed answer (the email then shows the neutral fallback or a missing vote, never an invented answer). Every scripted reasoning starts with "(skrypt offline)";
- `FileReportNotifier` writes each cycle's email to `<out-dir>/cycle-N/report-email.html` and `report-email.txt` (the latter starts with `Temat:` and the subject the Fast Loop would use).

The shipped scenario has two cycles 12 hours apart. In cycle 1 NVDA (+3%) and MSFT (−2.5%) pass the 2% gate, only NVDA reaches the 3% council threshold, and AAPL and BTC stay below their gates; in cycle 2 reflect settles the NVDA call as correct and the MSFT call as wrong (with the scripted diagnosis, since that move is above the gate too), and BTC passes the 5% crypto gate. The command prints one line per cycle (saved, skipped, failed, settled, subject, email path) and exits 0, 1 when every symbol of a cycle failed, or 2 when the scenario or the script cannot be read. Options: `--out-dir` (default `build/offline`), `--scenario`, `--llm-script`, and `--ml-model` for XGBoost weights; the default `data/offline/model.ubj` does not exist, so the model runs cold-start (target = current price).

What stays fixed: sentiment and news go through the FIN-35 stubs (`paid_calls_disabled`, so every prediction carries that data-quality flag, the email shows the DEGRADED badge and the graph logs one `WARNING` per saved prediction on stderr); the cycle-cost section is hidden as in any stubbed run; alpha sources, tool use, embeddings and attestation are off; the scenario clock stands still at each cycle's start, so the email reports a 0 s duration and two runs give the same files byte for byte. The mail sections that read the repository follow the production `config.toml` (equity curve, calibration curve, lessons, cycle diff). The command builds no `Settings`, so `config.toml` and `.env` play no part. [`tests/test_main_offline.py`](../tests/test_main_offline.py) runs the scenario under a socket and DNS recorder, checks the settlement, that every email section also exists in the golden snapshot, and that the output is deterministic.

## Build commands

```bash
uv sync                            # install deps (Anthropic SDK included; `--extra anthropic` is a kept alias)
uv run python main_agent.py        # one Fast Loop run (analysis + email); no database writes without DB_WRITES_ENABLED=true
uv run python main_trainer.py      # one Slow Loop run (XGBoost retrain); no database writes without DB_WRITES_ENABLED=true
uv run python main_tools.py --help # one-off tools: one subcommand per tool (see "One-off tools" below)
uv run python main_tools.py import-positions --csv positions.csv          # dry-run: diff of the CSV against the positions table (writes nothing)
DB_WRITES_ENABLED=true uv run python main_tools.py import-positions --csv positions.csv --apply  # replace the positions table with the CSV, as_of = now (refused without DB_WRITES_ENABLED=true)
uv run python main_backtest.py     # offline walk-forward backtest (folds purge the 12h label horizon) over `symbols` and `crypto_symbols` (NIEZ-18): per-symbol out-of-sample RMSE / hit-rate vs baseline, plus per-fold rank IC, sign-strategy return and turnover; logs mean IC, its sd and t-stat next to the fold count, and the sign strategy's mean return per row net of `paper_cost_bps` (`crypto_paper_cost_bps` for crypto) (`net=`) with the mean turnover (`turnover=`, a flip counts 1) (read-only, no model write)
uv run python main_tools.py evaluate           # offline eval: hit-rate + scale-free RMSE (returns) vs the no-change baseline with an "undetermined" verdict at cold-start, directional n_eff, naive baselines and binomial-test verdict (read-only)
uv run python main_tools.py evaluate --days 60 # custom evaluation window
uv run python main_tools.py evaluate --by-recipe  # + directional hit rate per recipe version (needs migration 030)
uv run python main_tools.py evaluate --by-universe  # + directional hit rate per symbol-universe version (from the v2 receipt)
uv run python main_tools.py evaluate --by component  # + ablation: LLM trend vs council recommendation vs XGBoost direction on the same rows (see below)
uv run python main_tools.py evaluate --by regime --by asset_class  # + LLM hit rate per segment; repeatable, also confidence_bucket, alpha_fusion_sign, edge_label
uv run python main_tools.py evaluate --by risk_signal --by council_agreement  # + LLM hit rate with and without each email risk signal (DIVERGENCE needs migration 034) and by council agreement with the LLM (WIA-47)
uv run python main_tools.py evaluate --echo  # + momentum echo test: is the LLM trend just the sign of the move that opened the gate? (needs migration 034)
uv run python main_tools.py evaluate --experiment EXP-02 --days 90  # + A/B arms of an in-loop experiment: treatment − control per metric with a block-bootstrap CI (arms from migration 037 or the v2 receipt)
uv run python main_tools.py evaluate --scores  # + RMSE in returns per asset class and Brier / log loss of the declared LLM confidence (WIA-40; reads symbol and confidence_score, migration 016)
uv run python main_tools.py evaluate --json > run.json  # the same report as JSON for comparing runs (WIA-40); notices about missing columns go to stderr
uv run python main_scorecard.py --days 120  # public scorecard page from the database (read-only, zero paid calls); writes public_scorecard_path only with public_scorecard_enabled, an empty window writes nothing
uv run python main_tools.py evaluate --discrimination  # + does confidence mean anything (WIA-38): AUC of the LLM confidence vs a directional hit and of the council consensus vs a BUY/SELL hit, with n, n_eff, a cycle-bootstrap CI and the hit rate of the top 20% / 50% / 100% by confidence
uv run python main_tools.py explain <prediction_id>  # "why?": RAG analogues, SHAP-lite attributions, effective threshold and council votes of one prediction, from its decision receipt (read-only)
uv run python main_tools.py explain --symbol NVDA --date 2026-08-14  # the same for every prediction of a symbol on one UTC day
uv run python main_tools.py persona-ablation --days 90  # does persona weighting help out of sample (WIA-46): council verdicts with the weights known at each prediction vs without weights, with a sign test (needs migration 045)
uv run python main_tools.py council-ablation --days 90  # marginal contribution of each council persona, leave-one-out (WIA-43): change in hit rate and Brier of the consensus strength without the persona, with a block-bootstrap CI; personas whose removal does not hurt are candidates for removal (needs migration 045; ignores interactions, so read it with the redundancy line of the weekly recap)
uv run python main_tools.py cost-envelope  # estimated min/max paid calls and USD of one Fast Loop cycle for config.toml, plus the worst week at the FIN-12 ceiling against fast_loop_weekly_budget_usd (offline, zero paid calls)
uv run python main_tools.py render-sample-report  # the report email of a fixed, fictional cycle into build/sample-report/ (HTML + plain text); no keys, no database, no network, zero paid calls (`--out-dir` to change the directory)
uv run python main_tools.py offline  # two whole Fast Loop cycles on fakes (in-memory repository, fixture prices, scripted LLM) through the real graph and report; one email per cycle into build/offline/cycle-N/; no keys, no database, no network (DX-04, see "Offline cycle" above)
uv run python main_tools.py render-graph-diagram  # docs/graph.mmd and docs/graph-without-fundamentals.mmd, the Fast Loop graph drawn by LangGraph from create_agent_graph (DOC-17); rerun after a graph change, a test compares nodes and edges
uv run python main_tools.py repo-stats  # docs/repo-stats.json: test functions (AST), migrations, ABC ports, adapter modules, domain modules and a stdlib-only check of the domain; numbers quoted in the docs are written as "N+" and a test fails when one is above the count or more than 25% below it (DOC-18)
uv run python main_tools.py verify-attestation --commitments /tmp/attestation-log/commitments.jsonl --reveals /tmp/attestation-log/reveals.jsonl  # check the commit-reveal ledger offline (WIA-26) and the Rekor entry of every cycle root that has one (WIA-73; `--rekor-repository`, `--rekor-workflow` for a fork): OK and exit 0, or FAIL and exit 1; no keys, no database, no network (see configuration.md, Verify it yourself)
uv run python main_tools.py reproduce-training  # does the sidecar's fingerprint and seed give the saved booster (ML-22): rebuilds the shared model's sample read-only, keeps the sidecar's date range, compares the dataset sha256, refits and compares the tree dump; REPRODUCED and exit 0, otherwise exit 1 (2 without the sidecar or the weights; `--sidecar` to point at another file)
uv run python main_tools.py junit-summary test-results/unit.xml  # Markdown table of tests, passed, skipped, failures and errors summed from JUnit reports (CI-16, CI appends it to the job summary); a missing file is a `no report` row; `--mutation` reads `mutmut junitxml` reports (TEST-35); no keys, no network
uv run pytest -m "not integration and not containers" # the quality gate, as in CI
uv run pytest                                   # everything, including live-API tests (real keys, paid calls, a real email)
uv run pytest -m containers                    # Docker: Postgres schema tests and the SupabaseRepository contract tests on PostgREST (TEST-17)
uv run pytest -m integration                   # live API tests (requires real keys)
uv run pytest tests/application/test_report_builder_snapshot.py --update-golden  # rewrite the email golden snapshots after an intended email change
uv run pytest tests/application/test_prompt_golden.py --update-golden  # rewrite the prompt golden files after an intended prompt change
uv run ruff check .                # lint (same command as CI)
uv run ruff format --check .       # formatting (same command as CI, CI-17); without --check it rewrites the files
uv run mypy                        # type check (strict) on [tool.mypy].files: src + every main_*.py + tests/domain (tests/ joins directory by directory)
uvx --from actionlint-py==1.7.12.25 --with shellcheck-py==0.11.0.1 actionlint  # workflow syntax + shellcheck of every run block (CI-23, job lint-infra)
uvx zizmor@1.30.1 .github/workflows  # workflow security audit (CI-23); offline without GH_TOKEN, CI adds the online audits
uvx --from shellcheck-py==0.11.0.1 shellcheck scripts/*.sh  # the shell helpers (CI-23)
```

Every evaluation number (the headline, `--by` segments and `--echo`) skips the placeholder rows saved after a failed main LLM call (`llm_analysis_failed`, the neutral "SIDEWAYS 50%"), with the same filter the email and the Track Record use: they are not predictions.

`--by` segments the offline evaluation (WIA-55). `component` scores, on the same settled rows, the directional hit rate of the LLM trend (the stored `is_trend_correct`), the council's `final_recommendation` (BUY → up, SELL → down, HOLD is no bet; rows without a verdict are left out) and the XGBoost direction (target above or below the price; a target equal to the price is the cold-start "no change" baseline of a model without weights, so no bet). The context segments split the LLM hit rate by `regime` (untagged rows form `UNKNOWN`), `asset_class`, `confidence_bucket` (0.00-0.60, 0.60-0.70, 0.70-0.80, 0.80-1.00, plus a separate `0.50 (domyślna)` bucket for exactly 0.5, the value saved when the LLM gave no confidence, so the default does not mix with genuinely low confidence; a genuine 0.5 lands there too, since the database cannot tell them apart), `alpha_fusion_sign` and `edge_label` (the `evaluate_edge` label of the stored `edge_sigma`; no value is `NO_SIGNAL`). Every segment prints its number of directional bets `n`, `n_eff`, a Wilson 95% interval on `n_eff`, the best naive "always BULLISH / BEARISH" baseline on the same rows and the binomial p-value against it, with Benjamini-Hochberg FDR (q = 0.10) across the segments of one `--by`; a segment with fewer than 10 bets is flagged "za mało danych". A segment's column is read only when it is asked for, so the default run works on any database, and each one needs its own migration: `council_verdict` `005`, `confidence_score` `016`, `regime` `017`, `alpha_fusion_score` `022`, `edge_sigma` `025`. `prediction_logs` stores no asset class, so `asset_class` classifies each symbol from the **current** `crypto_symbols` and `symbols_etf`: a ticker moved between lists changes segment retroactively. Read-only, zero paid calls; nothing of it reaches the email.

`--echo` is the momentum echo test (EXP-01). The gate only lets through moves of at least the threshold, and the LLM sees that same move in its prompt, so a trend that mostly repeats (or reverses) its sign is a call a free rule makes at no cost. On the settled predictions with a `gate_delta` (migration `034`, history backfilled from `price_snapshots`; see `gate_delta_enabled` in [configuration](configuration.md)) it prints (a) the share of echo (`predicted_trend` = sign of the gate delta) and of reversal (= the opposite sign); (b) on the same rows (a directional LLM call, a non-zero delta, a known move), the hit rate of the LLM and of the free baselines "sign of the move" and "opposite sign", with `n` and `n_eff`; (c) the LLM hit rate only where it disagrees with momentum. The comparison is paired: where the LLM and a baseline make the same call they hit or miss together, so only the rows where they disagree decide. There one side's hit is the other's miss, so each comparison is a sign test of the LLM's wins against 0.5 on `n_eff` (one bet per correlated day; a zero move is a tie and drops out): LLM vs "sign of the move" on the reversals, LLM vs "opposite sign" on the echoes, both corrected with Benjamini-Hochberg FDR (q = 0.10). Fewer than 10 compared rows are flagged "za mało danych". Before the migration the run prints "kolumna gate_delta niedostępna — uruchom migrację 034" and the rest of the report instead of failing. The result lives only in `evaluate`, not in the weekly recap or the email (owner's decision).

`--experiment <id>` compares the arms of an experiment defined in `config.toml` (EXP-05, see [Experiments](configuration.md#experiments)). Only the settled rows that carry an arm of that experiment count: from `prediction_logs.experiment_arms` (migration `037`) or, before the migration or with `experiment_arms_enabled` off, from the optional `experiment_arms` key of the v2 decision receipt; a row with both uses the column. For every metric of the experiment's prompt part it prints the treatment minus the control (percentage points for rates, raw difference for Brier scores, where lower is better), each arm's mean with `n` and `n_eff`, and a 95% interval from a block bootstrap that resamples whole UTC days, so both arms of one day move together and a market-wide day does not pass for a difference between arms. Fewer than 20 days give no interval ("za mało cykli"), and fewer than 10 rows in an arm are flagged "za mało danych". The primary metric is marked ★. An id missing from `config.toml` exits with code 1 before reading the database. Experiments run only while the Fast Loop runs, and the council fires only above `council_volatility_threshold`, so a council experiment needs weeks of cycles.

**One-off tools (ARCH-39).** [`main_tools.py`](../main_tools.py) is the composition root of the one-off CLI tools: every tool is one argparse subcommand (`uv run python main_tools.py <command> --help`), only `main_tools.py` builds the `Settings` and the adapters a tool needs, and the tool's logic (parsing, validation, diffs, metrics, formatting) stays in pure functions under `src/tools/`. A separate `main_<x>.py` is reserved for entry points with their own workflow (Fast Loop, Slow Loop, shock watch, deadman, backtest). `[tool.mypy].files` lists every `main_*.py`, so the tools are type-checked like the other entry points, and `tests/test_architecture.py` fails when a module under `src/tools/` imports `src.config`, an adapter or an LLM client. The offline evaluation moved here without an alias: `python -m src.tools.evaluate` is now `python main_tools.py evaluate`, with the same options.

**Explaining a prediction (WIA-56).** `explain` reads `prediction_logs.decision_receipts` (migration `020`), which until now only the graph wrote: the RAG analogues with their `id` and similarity, the SHAP-lite attributions, the effective volatility threshold after the regime and earnings multipliers, the prediction recipe and experiment arms of a v2 receipt, and the council votes from `council_votes`. Pass a prediction id, or `--symbol X --date YYYY-MM-DD` for every prediction of that symbol on one UTC day. An unknown `schema_version` prints a message instead of guessing the content, and a prediction saved without `receipts_enabled` says it has no receipt. Receipts exist only for predictions saved with `receipts_enabled` (in practice 2026-07-10 to 2026-08-17), and a v1 receipt carries no provenance. Read-only, zero paid calls, exit 0.

**Persona weighting ablation (WIA-46).** `persona_weighting_enabled` weighs each council persona by its hit rate over the last 90 days, counted from now (RPC `015`). Judging that switch on past predictions with those weights would use outcomes the prediction could not have known. `persona-ablation` reads the raw settled votes of the `persona_vote_outcomes` RPC (**migration `045`**) and replays the council verdict of every prediction of the last `--days` twice: with the weights known at the moment of the prediction (`persona_accuracy_weights_as_of` in [`src/domain/persona_track_record.py`](../src/domain/persona_track_record.py): only votes settled strictly before it, from the `--window-days` before it) and without weights, both through `derive_consensus`, the rule the Fast Loop uses. A vote without a settlement stamp, or with the lower bound `settled_at = timestamp` that the backfill of migration `029` left, counts as known 12 hours after its prediction (the forecast horizon), never earlier. Each verdict is scored with the persona rule of WIA-51 (HOLD inside the flat band of the asset class). The tool prints the hit rate with and without weights, how many verdicts only one version got right, and an exact sign test on those (one-sided: weighting is better). Only a result below 0.05 says the flag helps out of sample; otherwise the last line says to keep `persona_weighting_enabled = false`. Read-only, zero paid calls; without migration `045` it exits `1` with a message that names it.

**Importing positions (PORT-18).** `import-positions` replaces the hand-maintained `positions` table with a CSV file, so the STALE badge stops being the default state and positions get a real purchase date. The file is UTF-8, comma-separated, with a header row; one row is one lot, and several lots of one symbol (bought on different dates) are allowed:

```csv
symbol,quantity,avg_cost,currency,purchase_date
NVDA,10,118.50,USD,2025-03-01
SAP.DE,4,182.10,EUR,
```

`symbol` is the ticker as in `config.toml` (upper-cased), `quantity` and `avg_cost` are positive numbers with a dot as the decimal separator (`avg_cost` per unit, in `currency`), `currency` is a three-letter ISO 4217 code, and `purchase_date` is an optional column and value in ISO `YYYY-MM-DD`, not in the future; empty means unknown, and the report then says the purchase-date rate is unknown instead of guessing one. Every row is validated before the database is touched, and all errors are printed with their line numbers. A file without a single position is an error, not a request to empty the table. Without `--apply` the command prints the diff against the current table (`+` added, `-` removed, `=` unchanged lots) and writes nothing; reading the table is strict, so a database error stops it instead of showing everything as new. With `--apply` it inserts the file's lots with `as_of` = now and then deletes the rows that existed before, so a failed insert leaves the old portfolio in place. `--apply` needs `DB_WRITES_ENABLED=true` (see [Local runs are read-only by default](#local-runs-are-read-only-by-default)); without it the command prints the diff, says that nothing was written and exits with code 1. Writing `currency` and `purchase_date` needs migration `035` (see [configuration](configuration.md)); without it `--apply` fails with `PGRST204` before it deletes anything, so the old portfolio stays. The import makes no paid calls.

**Email golden snapshot (TEST-02).** `tests/application/test_report_builder_snapshot.py` renders three synthetic cycles with a frozen clock and fixed data: a full cycle (saved stocks with council, news and provenance badges, crypto, skipped and unsupported tickers, a source error, plus Risk Watch, your portfolio, Track Record, the model scorecard, cycle cost and the run footer), a quiet day where the volatility gate skips everything, and the day-1 cold start. It compares the HTML and the plain-text email character for character with `tests/application/snapshots/<scenario>.html` and `.txt`, and on a difference prints a unified diff. The HTML is stored one tag per line so a change reads as an ordinary diff; that layout is reversible and the test checks it, so the comparison stays exact. Nothing in it depends on the machine's time zone, locale or hash seed. When the email changes on purpose, rewrite the snapshots with the `--update-golden` command above (the only thing that writes them), review `git diff tests/application/snapshots/` and commit them together with the change; never edit them by hand to turn the test green.

**Prompt golden files (TEST-10).** `tests/application/test_prompt_golden.py` renders every prompt of the paid path — the prediction, the council investor, its revision round, the chairman, the mistake diagnosis and the weekly recap narrative — from fixed synthetic input and compares each one byte for byte with `tests/golden/prompts/<name>.txt` (UTF-8, `\n` line endings), printing a unified diff on a difference. The input is invented on purpose (no real headlines) and includes a prompt-injection attempt, so the golden file also shows what the untrusted-data fence leaves of it. A diff there is exactly what the model will see after the change. When a prompt changes on purpose, rewrite the files with the `--update-golden` command above, review `git diff tests/golden/prompts/` and commit them with the change. Where an option cannot be added to the pytest call (an IDE run configuration), `UPDATE_GOLDEN=1` is an alias of `--update-golden` for both the prompt and the email snapshots; it is ignored when `CI` is set, so a runner never rewrites a golden file on its own (TEST-10).

**Schema tests in any order (TEST-08).** In `tests/infrastructure/test_migrations.py` the module-scoped `pg_conn` fixture is the bare container: the schema appears only when `TestMigrations::test_migrations_apply_without_errors` runs, so the schema tests on `pg_conn` depend on test order and fail when selected alone with `-k`. A new test that only needs the finished schema takes `migrated_pg_conn` instead: a separate database in the same container (not a second container) that applies every migration in `MIGRATION_FILES` itself, so the test passes alone and in any order. It is shared by the whole module, so a test that inserts rows uses a unique symbol or the rolled-back transaction helper. Tests of applying the migrations themselves stay on `pg_conn`, and a state before a given migration comes from `fresh_pg`.

**Provider-shaped fixtures (TEST-16).** [`tests/fixtures/providers/`](../tests/fixtures/providers/) holds trimmed, key-free payloads in the shape of the real Finnhub, Alpha Vantage (news and the `Note` limit), CoinGecko, EDGAR (tickers map, submissions with the `xslF345X05/` prefix, Form 4 XML), FRED and NBP responses. [`tests/infrastructure/test_provider_shapes.py`](../tests/infrastructure/test_provider_shapes.py) parses each through the public adapter API and derives the expected values from the file, so a re-recorded fixture keeps the tests valid; its `integration` class (`uv run pytest -m integration tests/infrastructure/test_provider_shapes.py`, one Alpha Vantage request, the rest free) compares fresh responses with the fixture shapes: keys and JSON types, extra provider fields allowed. The `provider-shapes` job of `integration_smoke.yml` runs it every Monday. The fixtures are built from the shapes in the adapters and their tests, not yet recorded; re-record them from real responses with the keys and news content removed. A new file in the directory must be listed in that test as live-checked or not checkable, or `test_every_fixture_file_is_classified` fails.

**Unit tests have no network (TEST-11).** pytest runs with pytest-socket's `--disable-socket --allow-unix-socket` (`addopts` in `pyproject.toml`): a test that opens an internet socket or resolves a host name fails with `SocketBlockedError` instead of reaching a paid API or sending a real email with the keys exported in your shell. Tests marked `integration` or `containers` get the network back: `tests/conftest.py` adds pytest-socket's `enable_socket` marker to them at collection, before module-scoped fixtures such as the Postgres container start. Unix sockets stay open for the Docker client. A new test that needs the network must carry one of those two markers.

**No Docker is a failure in CI, a skip locally (CI-06).** The container fixtures (`pg_conn`, `fresh_pg` in [`tests/infrastructure/test_migrations.py`](../tests/infrastructure/test_migrations.py)) share one factory, `_pgvector_connection()`. Without a Docker daemon it skips the container tests, so a laptop without Docker Desktop still runs `uv run pytest -m containers` to the end; with `REQUIRE_DOCKER=1`, which the `containers` step of [`ci.yml`](../.github/workflows/ci.yml) sets, a missing daemon fails the run instead, because a green step that ran zero container tests proves nothing. Use `REQUIRE_DOCKER=1 uv run pytest -m containers` locally when you want the same guarantee. **Property tests (TEST-12).** `tests/domain/test_*_properties.py` check the invariants of the risk math with Hypothesis on 200 generated inputs each (the `ci` profile, no deadline, registered in `tests/conftest.py`; `HYPOTHESIS_PROFILE=default` restores the library defaults): VaR grows with the confidence level, CVaR is never below VaR, Pearson is symmetric and within [-1, 1], the equity-curve drawdown stays within [0, 1], sizing never suggests the full band without a finite hit rate, calibration stays within [0, 1], the implied edge is either a finite sigma or `NO_SIGNAL`, and the volatility gate is symmetric and monotonic in the move and in the threshold. Four of them failed when they were written and the domain now holds them: a period loss is capped at the whole capital (a short against a +150% move is ruin, equity 0, not negative equity), a NaN or infinite hit rate counts as no track record, a confidence outside [0, 1] (rows saved before `clamp_unit`) is clipped before scoring calibration, and a NaN or infinite IV, horizon or model move gives `NO_SIGNAL` with no `edge_sigma` instead of `ALIGNED`. Failing examples are kept in the git-ignored `.hypothesis/` directory and replayed first. Hypothesis is pinned to one exact release in the dev group (`hypothesis==6.168.5`, published 2026-10-05), chosen because it had been public for at least two weeks when it was pinned; raise it deliberately, with `uv lock`, never by a floating lower bound.

## Branch workflow

Work happens on a feature branch in its own worktree under `.worktrees/`, never on `main`. Three helpers in [`scripts/`](../scripts/) cover the cycle:

```bash
scripts/worktree-new.sh <type> <description>   # branch <type>/<description>-YYYY-MM-DD from origin/main in .worktrees/<description>-YYYY-MM-DD; prints the path
# … work and commit in that worktree …
scripts/worktree-status.sh                     # every worktree with ahead/behind, uncommitted files and a state: ready, dirty, empty or detached
scripts/merge.sh <branch> "<Conventional Commit message>" --dry-run   # validate and print the plan, change nothing
scripts/merge.sh <branch> "<Conventional Commit message>"             # merge --no-ff into main, remove the worktree, delete the branch
```

`<type>` is a Conventional Commit type (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`, …) and `<description>` is kebab-case. The merge message is required. `merge.sh` refuses before merging when the primary or the source worktree has uncommitted work, when `origin/main` is ahead, or when the source is a protected branch.

These three files are canonical copies of the Platform templates (`Platform/templates/<name>`, named on the `CANON:` line of each header). Do not edit them here: a change goes into the template and is copied back byte for byte. [`tests/test_workflow_scripts.py`](../tests/test_workflow_scripts.py) checks that each one exists, is executable, names its template and parses in bash.

`merge.sh` removes regenerable build output with the worktree (`.venv/`, `.pytest_cache/`, `__pycache__/` and similar), but the caches of the lint gate, `.mypy_cache/` and `.ruff_cache/`, are not on that list and block the merge. Check the list it prints (or `git status --short --untracked-files=all --ignored=matching` in the worktree), and only when every entry is disposable, rerun with `--discard-worktree-changes`. That flag also deletes any other ignored file in the worktree, such as a local `.env`.

## GitHub Actions

Eleven workflows in [`.github/workflows/`](../.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](../.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest, plus a parallel **SonarCloud** scan job (`pytest --cov` → coverage upload). The pytest step of the quality job measures branch coverage of `src/domain` and fails below a floor (CI-03): `--cov=src/domain --cov-branch --cov-fail-under=98`, the same number as `fail_under` in `[tool.coverage.report]` of `pyproject.toml`; the first measurement was 98.49%, and the floor only rises (`COVERAGE_RATCHET` in [`tests/test_quality_gate_sync.py`](../tests/test_quality_gate_sync.py)), so deleting a domain test turns the step red. The SonarCloud job passes `--cov-fail-under=0` because it measures all of `src/` and the entry points, which `fail_under` was not set for. Relative Markdown links are checked by [`tests/test_markdown_links.py`](../tests/test_markdown_links.py) inside the pytest gate, so a dead link fails locally too, not only in CI. A manual run (*Run workflow*) also starts the `lowest-direct` job (CI-12): `uv sync --resolution lowest-direct` installs the lowest version each direct dependency allows (dev tools included) and runs the unit tests on it, so a lower bound in `pyproject.toml` is tested, not just declared; a red job means raise that bound. [`tests/test_dependency_bounds.py`](../tests/test_dependency_bounds.py) keeps every core lower bound in the same major version as `uv.lock` |
| [`fast_loop_12h.yml`](../.github/workflows/fast_loop_12h.yml) | `30 5 * * 1-5` | Mon–Fri 07:30 / 06:30 | Analysis + email report, plus manual `workflow_dispatch`. The schedule was paused from 2026-08-17 to 2026-10-23 and resumed through the [resume checklist](resume-checklist.md); while it is active, `tests/test_workflow_env_wiring.py::test_enabled_fast_loop_schedule_requires_deadman` checks the deadman cron, the cost ceiling, the weekly budget, the liveness gap and the write and paid-call switches. When the cycle wrote the public scorecard (`public_scorecard_enabled`), the agent job uploads only its directory as the Pages artifact and a separate `deploy-scorecard` job, the only one holding `pages: write` and `id-token: write`, deploys it (WIA-16). |
| [`slow_loop_weekly.yml`](../.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining; uploads `data/models/price_predictor.ubj` as the asset of the GitHub Release `model-latest` (CI-13) only when a candidate beats the naive baseline, and never commits or pushes to `main`. The release notes and the job summary carry the sha256 of the uploaded file. The Fast Loop downloads that asset before the agent step; without the release or the asset it runs on the cold-start baseline with a warning. The ML-30 direction classifier's weights (`price_predictor_direction.ubj`), when its gate passed in the run, go to the same Release as a challenger, which the Fast Loop downloads too and reads only with `shadow_predictions_enabled` (ML-33). Runs share the `slow-loop` concurrency group and queue instead of cancelling each other. With `reference_signal_enabled` (migration `042`) it also runs the free weekly reference signal (WIA-24) |
| [`shock_watch_hourly.yml`](../.github/workflows/shock_watch_hourly.yml) | `7 * * * *` | every hour at :07 | Free shock alerts (`main_watch.py`) — no LLM, no paid API. An hourly 24/7 cron (BTC/ETH trade around the clock), resumed on 2026-10-23 after a pause from 2026-10-16, and independent of the Fast Loop: since ALR-02 it reads the day's change from the free quote first (Finnhub `dp`, CoinGecko 24 h change) and uses a Fast Loop snapshot only as a fallback, where a reference older than 30 h suppresses the alert. A run where no daily change was fresh and every snapshot is stale logs a `Shock watch ślepy` WARNING, a run where every symbol fails (e.g. the database is down) exits 1, and without a push channel (`TELEGRAM_*` or `SLACK_WEBHOOK_URL`) it logs an error and neither sends nor records a shock, so the daily debounce stays free. The owner uncomments its schedule; the Fast Loop cron can stay off. |
| [`deadman_daily.yml`](../.github/workflows/deadman_daily.yml) | `17 9 * * *` | 11:17 (summer) | Free liveness check (`main_deadman.py`) — reads the newest `price_snapshots` and `model_scorecards` rows and mails through Resend when a job has been silent longer than `liveness_*_max_gap_h` in `config.toml` (`0` = deliberately paused, e.g. the Fast Loop; once resumed, the Fast Loop is counted in NYSE trading days, so weekends and holidays raise no alert). One run a day means at most one alert a day. Each run first reads one `price_snapshots` row (read-only, NIEZ-28): the Supabase project is on the Free plan, which pauses a project after 7 days without activity, so a database that does not answer gets its own mail (`baza Supabase nie odpowiada`); this monitors a pause, it does not prevent one. A database that answers the probe but fails a liveness read exits 1, which triggers the failure mail. |
| [`bot_poll.yml`](../.github/workflows/bot_poll.yml) | `*/15 * * * *` | every 15 minutes | The owner's Telegram bot (`main_bot.py`, MAIL-48) — polls `getUpdates` from the offset in `bot_state`, answers only the `TELEGRAM_CHAT_ID` chat (`/why`, `/last`, `/pos`, `/watch`, `/mute`, `/unmute`, `/help`), no LLM, no paid API. Without the `TELEGRAM_*` secrets or with `telegram_bot_enabled = false` the run stops before installing dependencies. Outside the `supabase-writers` queue; a pause comments out the schedule like the other crons. |
| [`scorecard_pages.yml`](../.github/workflows/scorecard_pages.yml) | _disabled_ (manual only) | — | Free public scorecard (`main_scorecard.py`, WIA-52) — renders the aggregates-only page from the settled predictions already in the database, no LLM, no paid API, read-only repository. The `days` input overrides `track_record_days` (an empty window writes no page). Uploads only `dirname(public_scorecard_path)` as the Pages artifact when the page exists and deploys it with `actions/deploy-pages` in a separate job with the `github-pages` environment, the only job holding `pages: write` and `id-token: write`. Designed for a weekly cron (`0 6 * * 1`), commented out during the pause; nothing is written while `public_scorecard_enabled` is off. |
| [`integration_smoke.yml`](../.github/workflows/integration_smoke.yml) | `41 6 * * 1` (provider shapes only; full smoke `workflow_dispatch` only) | Monday 08:41 / 07:41 | Live-API smoke (CI-05): the manual `live-api-smoke` job runs `pytest -m integration`, the tests no other workflow runs — OpenAI chat and embeddings, Anthropic, Alpha Vantage (`TestAlphaVantageLive`: the answer must carry `feed` or be a soft limit the DAT-06 classifier recognises, so a changed response shape fails instead of reading as an empty feed), Finnhub, EDGAR and one real email through Resend. Each run makes 3 paid LLM calls, so that job has no cron and no push or PR trigger, and the workflow reads only its own `SMOKE_*` secrets; a missing secret skips that test. The weekly `provider-shapes` job (TEST-16) runs only [`tests/infrastructure/test_provider_shapes.py`](../tests/infrastructure/test_provider_shapes.py): it fetches fresh Finnhub, Alpha Vantage (one request), CoinGecko, EDGAR, FRED and NBP responses and fails when a key or JSON type of the payloads in [`tests/fixtures/providers/`](../tests/fixtures/providers/) is gone; a failure or timeout mails `SMOKE_DIGEST_TO_EMAIL`. [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) and [`tests/test_integration_smoke_schedule.py`](../tests/test_integration_smoke_schedule.py) guard both jobs. |
| [`dependency_audit.yml`](../.github/workflows/dependency_audit.yml) | `23 6 * * 1`, plus every push to `main` that changes `uv.lock` | Monday 08:23 / 07:23 | Known-vulnerability scan of `uv.lock` (BEZ-21) with [osv-scanner](https://github.com/google/osv-scanner): the release binary is pinned by version and SHA-256, the Markdown table of findings lands in the job summary, and a finding turns the job red (exit 1). No secrets. Dependabot bumps versions; this says when the lock carries a known advisory, also for an unchanged lock. |
| [`mutation_testing.yml`](../.github/workflows/mutation_testing.yml) | — (`workflow_dispatch` only) | — | Mutation testing of the domain (TEST-35) with mutmut 2.5.1 on `asset` (the volatility gate), `regime`, `sizing`, `var` and `finops`, one matrix job per module that runs the `tests/domain` files importing it (`tests/test_mutation_workflow.py` keeps the lists complete). Each job appends mutants, killed, survived, timeouts and the score to the job summary (`main_tools.py junit-summary --mutation`) and uploads `mutmut-<module>.xml`, which carries the diff of every surviving mutant. No threshold yet: surviving mutants never fail the job, only a mutmut crash does; the owner sets the threshold after the first measurement. mutmut 3 cannot run here: it strips the `src.` prefix from module names and rejects `src.*` modules, while this repository imports `src.domain.…`. |
| [`db_migrate.yml`](../.github/workflows/db_migrate.yml) | — (`workflow_dispatch` only) | — | Applies `supabase/migrations/` via a pinned Supabase CLI (`2.121.0`, CI-09). Deliberately manual: a broken migration must never fire on a cron. The job runs in the `production-db` environment, which holds `SUPABASE_DB_URL` and waits for the owner's approval (BEZ-22). |

Fast Loop, Slow Loop, Shock Watch, Deadman, Public Scorecard and DB Migrate each end every job with a step that runs only when the job failed or was cancelled (a timeout included) and mails the run link through Resend — [`scripts/notify-failure.sh`](../scripts/notify-failure.sh), plain bash with the same `RESEND_API_KEY` / `DIGEST_TO_EMAIL` secrets as the digest, so it still fires when dependency installation is what broke. GitHub's own failure mail reaches only whoever last edited the cron. The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

**When a job exits 1 (NIEZ-38).** The four per-symbol entry points share one rule, [`run_exit_code`](../src/domain/exit_policy.py): exit 1 only when **every** symbol of the run failed; a partial failure exits 0, logs one `WARNING` (`częściowa porażka — padło N/M`) and shows the counts in the run summary. A red job from a single symbol teaches you to ignore red, and a green job after a total outage hides it; the failure mail and the heartbeat `/fail` ping follow the exit code, so they fire on the total outage only. [`tests/test_exit_code_contract.py`](../tests/test_exit_code_contract.py) drives each entry point's `main()` with 1 of 3 and 3 of 3 symbols failing.

| Entry point | Exit 1 when | Partial failure (exit 0 + `WARNING`) | Other codes |
|---|---|---|---|
| `main_agent.py` (Fast Loop) | every eligible symbol failed | some symbols failed; an undelivered report is flagged, not failed (NIEZ-12) | 2 — paid calls off with writes on (FIN-35) |
| `main_trainer.py` (Slow Loop) | the feature store refresh failed, or every symbol's evaluation failed | some symbols failed, or only the pooled model failed (the previous weights stay) | 2 — as the Fast Loop |
| `main_watch.py` (Shock Watch) | every watched symbol failed (e.g. the database is down) | some symbols failed | — |
| `main_backtest.py` | every symbol's backtest failed | some symbols failed | — |

**Pinned actions and tools (CI-08, CI-09).** Every `uses:` points at a full commit SHA with the version in a comment (`actions/checkout@11d5960a… # v4.4.0`), because a tag such as `@v4` can be moved to any commit by the action's owner, also under a job with `contents: write`. [`tests/test_workflow_env_wiring.py::test_every_action_is_pinned_by_sha`](../tests/test_workflow_env_wiring.py) fails on a tag, a short SHA or a missing `# vX.Y.Z` comment; look a tag up with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>`. DB Migrate installs a fixed Supabase CLI (`version: 2.121.0`), never `latest`, since that binary runs DDL on production; when you bump it, check that the CLI still parses migration names with `^([0-9]+)_(.*)\.sql$`. `test_supabase_cli_version_is_pinned` and `test_no_workflow_installs_a_tool_at_latest` guard both. [Dependabot](../.github/dependabot.yml) (CI-07) keeps the pins and the lock current: once a week it opens one grouped pull request for the GitHub Actions (SHA and version comment together) and one for the `uv` dependencies, instead of a pull request per package. Dependabot pull requests get no repository secrets, so the SonarCloud job skips itself there while ruff, mypy and pytest run as usual. The Supabase CLI `version:` input is not a dependency Dependabot tracks, so it is bumped by hand.

**uv is pinned too (CI-25).** Every `astral-sh/setup-uv` step sets `version: "0.9.22"` (uv `0.9.22`) instead of installing the newest release, because the result of `--frozen`/`--locked`, the lock format and the flags would otherwise depend on the day a run happens. Dependabot does not touch that input either: bump it in every workflow in one commit, and the number in this paragraph with it. `test_setup_uv_pins_uv_version`, `test_setup_uv_steps_share_one_uv_version` and `test_docs_name_the_pinned_uv_version` in [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) fail on a missing, a differing or an undocumented pin.

**Type escapes (CI-15).** `[tool.mypy]` enables `ignore-without-code`, so a bare `# type: ignore` (which silences every error on its line) fails the mypy step; an ignore must name its error code. The number of `# type: ignore[...]` comments and of `cast(Any, ...)` calls in `src/` and `main_*.py` only goes down: `TYPE_IGNORE_BASELINE` and `CAST_ANY_BASELINE` in [`tests/test_quality_gate_sync.py`](../tests/test_quality_gate_sync.py) fail on the next one, and a change that removes an escape lowers the number in the same commit.

**Test counts (CI-16).** Both pytest steps of the `quality` job write a JUnit report (`--junitxml=test-results/unit.xml` and `test-results/containers.xml`), and the next step, which also runs after a red pytest, appends `main_tools.py junit-summary` of both to the job summary: tests, passed, skipped, failures and errors per step and in total, counted from what actually ran rather than from `def test_`. Link that run in a commit or a document instead of copying numbers. [`tests/test_ci_contract.py`](../tests/test_ci_contract.py) fails when a pytest step stops writing its report or the summary step stops reading it.

**Infrastructure lint (CI-23).** The `lint-infra` job of `ci.yml` runs three static checks over what holds the secrets: [actionlint](https://github.com/rhysd/actionlint) (workflow syntax, expressions, and shellcheck of every `run:` block), [zizmor](https://docs.zizmor.sh/) over `.github/workflows` (template injection, excessive permissions, unpinned actions, credentials persisted by a checkout; with the job's read-only token also the online audits: impostor commits under a pinned SHA, known-vulnerable actions, a version comment that does not match its SHA) and shellcheck over `scripts/*.sh`. The tools come from PyPI through `uvx` at exact versions (`actionlint-py==1.7.12.25`, `zizmor@1.30.1`, `shellcheck-py==0.11.0.1`), and `test_uvx_tools_in_workflows_are_pinned_to_an_exact_version` in [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) fails on a `uvx` without one. A finding is fixed, not silenced. The same commands run locally from the Build commands block above.

**Workflow policy (CI-10, NIEZ-27, FIN-18).** [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) parses every file in `.github/workflows/`, so a new workflow is held to it from its first commit: top-level `permissions:`, `timeout-minutes` on every job, a workflow-level `concurrency:` group on every workflow with a cron or a database write, `persist-credentials: false` on every checkout of a job that does not `git push` (only `publish-attestation-ledger` pushes, so only its ledger checkout keeps the token, and says so with an explicit `persist-credentials: true`, CI-23), and no `cancel-in-progress: true` for a job that calls a paid API. A job counts as paid when it gets `PAID_CALLS_ENABLED` or an OpenAI, Anthropic or Alpha Vantage secret, so the list needs no upkeep. The Fast Loop therefore queues the next run instead of cancelling the one in progress (FIN-18): a cancelled run has already paid for its LLM calls and would also abort the attestation ledger push. Every job that runs a `main_*.py` entry point outside `ci.yml` installs with `uv sync --frozen --no-dev` (CI-18): pytest, mypy, ruff and testcontainers never land in a job that holds the keys; only the jobs that run pytest themselves (the gate in `ci.yml`, the live-API smoke and mutation testing) sync the dev group.

The jobs that write to the database (Fast Loop `run-market-agent`, Slow Loop `retrain-predictive-model`, DB Migrate `migrate`) also wait in one repository-wide queue, `concurrency.group: supabase-writers` with `cancel-in-progress: false` (NIEZ-27), so a manual migration cannot land in the middle of a cycle or of the feature-store refresh. The group is declared on the **job**, next to each workflow's own group (`fast-loop`, `slow-loop`, `db-migrate`): GitHub holds a workflow group for the whole run and a job group only while that job runs, so the ledger push and the Pages deploy do not block the next writer, and a job group never has the name of a workflow group (that would make a run wait for itself; a test pins it). GitHub keeps one running and one pending item per group and cancels an older pending one when a newer arrives, so a scheduled run waiting behind a manual migration can be replaced by a second manual dispatch; the deadman notices a Fast Loop or Slow Loop that never ran. The hourly Shock Watch is deliberately outside the queue: its hourly pending job would keep displacing the daily Fast Loop. It writes only `shock_alerts` (debounced by a unique key), so a migration that rebuilds objects the watch reads can still fail one hourly run.

**Egress audit (BEZ-20).** The job that runs the agent holds every paid key in its environment and executes the code of ~150 packages from `uv.lock`, so the first step of `run-market-agent` (Fast Loop), `retrain-predictive-model` (Slow Loop) and `watch-market-shock` (Shock Watch) is [`step-security/harden-runner`](https://github.com/step-security/harden-runner), pinned by full SHA like every action (`v2.22.1`), with `egress-policy: audit` and nothing else. Audit only records which hosts the runner talks to; it blocks nothing, so a host missing from the list below can never fail a run. Switching to `block` with an allowlist is a separate decision to take once the inventory is complete; [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) (`test_secret_bearing_jobs_start_with_harden_runner`, `test_harden_runner_never_blocks_traffic`) fails when one of the three jobs does not start with it or when any harden-runner step leaves `audit`. The trade-off: the action sees the traffic of the job, and the run page of this public repository links to the StepSecurity insights page with the destinations of that run, so only host names, never a key or a payload, end up there.

To read the result, open a run in the Actions tab, follow the *Harden-Runner* link in the job summary and compare the outbound destinations with the hosts the code is expected to call:

| Host | Why |
|---|---|
| `api.anthropic.com`, `api.openai.com` | main analysis (Anthropic), embeddings and council (OpenAI); Fast Loop, and the Slow Loop only inside the weekly paid budget |
| `finnhub.io` | quotes, news, fundamentals, analyst recommendations |
| `www.alphavantage.co` | sentiment, news and the fundamentals refresh |
| `*.supabase.co` | PostgREST and Storage (the digest archive) |
| `api.resend.com` | the digest, the recap and the failure mail of every job |
| `api.coingecko.com` | crypto prices and 24 h change |
| `api.nbp.pl` | EUR/PLN and USD/PLN rates |
| `api.stlouisfed.org` | FRED yield curve and macro series |
| `www.sec.gov`, `data.sec.gov` | EDGAR filings (`insider_flow_enabled`, `corporate_events_enabled`; both off today, so none expected yet) |
| `www.reddit.com` | Reddit social velocity (`social_velocity_enabled`; off today, so none expected yet) |
| `api.telegram.org`, `hooks.slack.com` | push channels, only when their secrets are set |
| `pypi.org`, `files.pythonhosted.org` | `uv sync` |
| `github.com`, `api.github.com`, `objects.githubusercontent.com` | checkout, the `astral-sh/setup-uv` download of the uv binary, `gh release download` / upload of `model-latest`; `astral.sh` only if a step ever calls the uv installer instead |
| the host inside the `HEARTBEAT_URL` secret | the external heartbeat (NIEZ-13); the name is not in the repository |

Beyond that list expect the runner's own infrastructure (the Actions cache and artifact services under `*.actions.githubusercontent.com`, the Ubuntu package mirror for the Slow Loop's `apt-get install age`, `ghcr.io` or `registry.npmjs.org` only if an action pulls them) and `www.deribit.com`, which only appears once `deribit_dvol_enabled` is on. Shock Watch should show nothing but the free sources, the database, the push channels, the heartbeat and (on a failed run) Resend: an OpenAI, Anthropic or Alpha Vantage host there would be a finding, since BEZ-12 gives that job placeholders for those keys. Anything else is a candidate for the allowlist only after you know which dependency calls it.

**External heartbeat (NIEZ-13).** A watchdog on GitHub Actions dies with it, and GitHub disables the crons of a repository after 60 days without activity (the 08-18 → 10-15 gap lasted 58 days), so the liveness signal comes from outside. Fast Loop, Slow Loop and Shock Watch ping the URL in the `HEARTBEAT_URL` secret once, at the end of the run, through `HeartbeatPort` ([`src/infrastructure/adapters/http_heartbeat.py`](../src/infrastructure/adapters/http_heartbeat.py)): a run that exits 0 sends `GET <URL>`, any other exit code (1, or 2 for the FIN-35 refusal) and a crash send `GET <URL>/fail`, the healthchecks.io convention. A missed ping inside the check's period is the alarm, raised by the external service. The URL is a secret (the check's UUID sits in its path): it is a `SECRET_SETTINGS_FIELDS` member (outside `config_hash`, redacted in the logs), and the adapter's exception carries only the error type and the HTTP status. A failed ping is logged and never changes the exit code. An empty or missing secret builds `NullHeartbeat`, so no request is made. One URL serves all three jobs, so the check's period must cover the longest gap between runs — today the weekly Slow Loop (7 days plus a grace period) while the Fast Loop and Shock Watch crons are paused. To enable it, create a check, add its ping URL as the `HEARTBEAT_URL` repository secret and `scripts/gh-secrets-check.sh` stops reporting it missing.

**Evidence backup (NIEZ-32).** The track record is the main evidence of the project, and the Supabase Free plan has no point-in-time recovery: a lost project or a migration with a `DROP` would erase it. Every Slow Loop run therefore exports the evidence tables — `prediction_logs`, `council_votes` and `model_scorecards` — through `ExportEvidenceUseCase` ([`src/application/use_cases/export_evidence.py`](../src/application/use_cases/export_evidence.py)): a paginated read of an **allowlist** of columns (`EVIDENCE_COLUMNS`), one CSV per table (JSONB as JSON) plus `manifest.json` with the row counts and columns. The allowlist leaves out the embedding vectors, the commitment salt (it must not leak before the reveal; the hash stays), the news text and every other column a later migration adds until it is listed; the positions and subscriber tables are never read, and e-mail addresses inside texts are replaced with `[e-mail usunięty]`. A listed column that the database does not have yet (its migration not applied) is dropped from the read with a `WARNING`, so the copy of the rest still happens. The archive is encrypted **before it touches the disk**: `AgeEncryptedArchive` ([`src/infrastructure/adapters/age_archive.py`](../src/infrastructure/adapters/age_archive.py)) builds the tar.gz in memory, pipes it to `age --encrypt --recipient <BACKUP_AGE_RECIPIENT>` and writes only output that starts with the age header, as `data/backup/evidence-<ISO year>-W<week>.tar.gz.age` (git-ignored); the run logs its path, sha256 and size. The workflow installs `age` only when the secret is set (an apt failure does not block training) and uploads `data/backup/*.age` — nothing else — as a workflow artifact with 90-day retention. The repository is public, so anyone signed in to GitHub can download the artifact, which is why only the ciphertext is uploaded; without `BACKUP_AGE_RECIPIENT`, or with a value that is not an age public key (the `.env.example` placeholder), the step is skipped with a `WARNING` and nothing is written or uploaded — there is no unencrypted fallback. To enable it, run `age-keygen -o backup-key.txt` on your machine, keep that file outside the repository, and add the `public key: age1…` line's key as the `BACKUP_AGE_RECIPIENT` repository secret. To restore it, follow [Restoring the evidence backup](#restoring-the-evidence-backup-niez-48). Zero paid calls; a failed export is logged and never changes the training exit code.

**Run summary (NIEZ-22).** Fast Loop, Slow Loop and Shock Watch append a short markdown summary to the run page in the Actions UI (the file named by `GITHUB_STEP_SUMMARY`), so the outcome is visible even when the email did not go out. The repository is public and so is the run page, which is why the summary carries **aggregates only** — the same invariant as the public scorecard: no watchlist symbol, email address, error text, reasoning, price, amount or position. The Fast Loop lists the duration, the number of symbols per status and per skip reason, the failures per exception class (the class name, never the message), the paid calls per source, the cycle cost (from tokens, an estimate, or both) or `n/a` on a stub run, the quota alerts per severity, whether the report email was delivered, not delivered or not configured, and the HTTP requests per host (NIEZ-24): requests, `2xx`, `4xx`, `429`, `5xx` and urllib3 retries for every host the `requests`-based adapters called (Finnhub, Alpha Vantage, SEC, CoinGecko, NBP, FRED, Reddit, Resend, Telegram, Slack, the heartbeat). The counter is a response hook that `build_session()` in [`src/infrastructure/adapters/_http.py`](../src/infrastructure/adapters/_http.py) puts on every session, so it covers all adapters without changes in their code; `main_agent` installs one `HttpCallMeter` ([`src/application/http_metering.py`](../src/application/http_metering.py)) for the whole run with `metered_http` and binds it to the cycle's `CycleTelemetry`. It keys by host only — never the path or the query string, which carry API keys — and the same numbers go to one `INFO` line at the end of the run. SDKs on httpx (OpenAI, Anthropic, supabase-py) do not pass through it; the LLM side is the `llm_calls` cost ledger. Shock Watch writes how many references it checked, how many were stale and how many alerts it sent; the Slow Loop writes the per-symbol evaluation statuses as counts, the pooled model's status and sample count, and whether the feature store refresh failed. The renderers are pure functions in [`src/application/report_ops.py`](../src/application/report_ops.py); only the entry points write the file, and only when the variable is set, so a local run writes nothing. A failed render or write is logged and never changes the exit code.

**A report that did not arrive (NIEZ-12).** When the Fast Loop had an email channel and the report did not go out, or the Slow Loop's recap could not be built or sent, the job prints the annotation `::error title=StockAgent::Raport nie został dostarczony`, which the Actions UI shows on the run, and stores a CRITICAL alert (`source` "Raport") in `quota_alerts`, so the next email carries it as a banner and, with `realtime_alerts_enabled`, it is pushed. The exit code stays 0: an undelivered report is not a failed cycle (the Fast Loop still exits 1 only when every symbol failed). A run with no email channel configured, and a recap skipped for too few closed predictions, have nothing to deliver and print nothing.

**Repository secrets** (the only things CI needs beyond the committed `config.toml`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEYS`, `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY`, `DIGEST_TO_EMAIL`, `FRED_API_KEY`. Database migrations need one more, used only by the manual DB Migrate workflow and stored as a secret of the `production-db` **environment**, not of the repository (BEZ-22): `SUPABASE_DB_URL` — the **session pooler connection string** (`postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`, Supabase dashboard → Connect → Session pooler), not the PostgREST `SUPABASE_URL`. The two are different things and only the former can run DDL. Do not use the direct connection (`db.<ref>.supabase.co`): it is IPv6-only unless the paid IPv4 add-on is on, and GitHub-hosted runners have no IPv6, so DB Migrate cannot reach it. The transaction pooler (port 6543) is not suited to migrations either. URL-encode the password's special characters (`@` → `%40`, `#` → `%23`, …). The `migrate` job runs in that environment, so only it can read the one secret with DDL rights on production, and the environment's required reviewer (the owner) approves every run, with an audit trail in the Actions UI. Set it up once: create the environment with yourself as the required reviewer (Settings → Environments → `production-db`), then `gh secret set SUPABASE_DB_URL --env production-db` and `gh secret delete SUPABASE_DB_URL` for the old repository copy. Until then the workflow does not break: GitHub creates a missing environment on the first run without protection rules, and the repository secret of the same name still serves the job. `scripts/gh-secrets-check.sh` and `scripts/gh-secrets-sync.sh` read and write that secret in the environment (`ENVIRONMENT_SECRETS` at the top of both), and the check flags a copy left on the repository. `ANTHROPIC_API_KEY` is required (main analysis runs on Claude); `DIGEST_TO_EMAIL` is a **secret** (was a variable — move it). `FRED_API_KEY` is **optional** — only needed when `yield_curve_enabled = true` (the FRED yield-curve alpha source). `SONAR_TOKEN` is **optional** — it enables the SonarCloud scan job; without it that job is skipped, so CI never breaks before Sonar is configured. `HEARTBEAT_URL` is **optional** — the external heartbeat below; without it no ping is sent. `BACKUP_AGE_RECIPIENT` is **optional** — the age public key of the weekly encrypted evidence backup (NIEZ-32, above); without it the Slow Loop makes no backup. The manual Integration smoke reads its own optional secrets, never the production keys: `SMOKE_OPENAI_API_KEY`, `SMOKE_ANTHROPIC_API_KEY`, `SMOKE_FINNHUB_API_KEY`, `SMOKE_ALPHA_VANTAGE_API_KEY` (one key, not a list, so a smoke run does not spend the Fast Loop's daily Alpha Vantage quota), `SMOKE_FRED_API_KEY` (the FRED shape check), `SMOKE_RESEND_API_KEY` and `SMOKE_DIGEST_TO_EMAIL` (the sender is Resend's sandbox `onboarding@resend.dev`, which only delivers to the Resend account's own address). Until they are set, `scripts/gh-secrets-check.sh` lists them as missing and the smoke skips the matching tests.

**SonarCloud — one-time setup.** [`ci.yml`](../.github/workflows/ci.yml) carries a `sonarcloud` job that runs `pytest --cov` over `src/` and every `main_*.py` entry point (CI-04: both `sonar.sources` and the `--cov` list name each entry point, and [`tests/test_quality_gate_sync.py`](../tests/test_quality_gate_sync.py) fails when a new one is missing; the coverage this reveals is not hidden with exclusions) and uploads code + coverage to [SonarCloud](https://sonarcloud.io/project/overview?id=DominikSienkiewicz_StockAgent) (keys live in [`sonar-project.properties`](../sonar-project.properties)). To enable it: (1) create a **`SONAR_TOKEN`** repository secret (SonarCloud → *My Account → Security*); (2) in SonarCloud, **disable *Automatic Analysis*** (*Administration → Analysis Method*) — CI-based and automatic analysis cannot both run, and only the CI scan uploads the coverage report the "Sonar way" gate (≥ 80% on new code) needs.

### Restoring the evidence backup (NIEZ-48)

The weekly evidence backup goes back into a fresh Supabase project in five steps. The restore
drill `TestEvidenceRestoreDrill` in `tests/infrastructure/test_migrations.py` (container test)
builds an export with `ExportEvidenceUseCase`, applies every migration to an empty database,
and runs the load order and the SQL block of this section as written.

1. Download the `evidence-backup-<run id>` artifact of a Slow Loop run and decrypt it with the
   private key: `age --decrypt -i backup-key.txt evidence-2026-W43.tar.gz.age | tar xz`. You get
   `prediction_logs.csv`, `council_votes.csv`, `model_scorecards.csv` and `manifest.json`.
2. Create the new Supabase project, put its session pooler string into `SUPABASE_DB_URL` (the
   repository secret and your shell), and apply every migration to the empty schema: the
   **🗄️ DB Migrate** workflow, or `supabase db push --db-url "$SUPABASE_DB_URL"`.
3. Load the files in this order, because `council_votes` references `prediction_logs`. The
   column list is each file's header line, so a column the export does not carry keeps its
   default:

```bash
for table in prediction_logs council_votes model_scorecards; do
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 \
    -c "\copy $table ($(head -1 "$table.csv")) FROM '$table.csv' WITH (FORMAT csv, HEADER true)"
done
```

4. Move the id sequences past the restored ids — `COPY` does not advance them, so the next
   council vote or scorecard would hit a duplicate key — and rebuild the feature store
   (`psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1`, then paste):

```sql
SELECT setval(pg_get_serial_sequence('council_votes', 'id'), COALESCE(max(id), 0) + 1, false)
FROM council_votes;
SELECT setval(pg_get_serial_sequence('model_scorecards', 'id'), COALESCE(max(id), 0) + 1, false)
FROM model_scorecards;
SELECT refresh_ml_feature_store();
```

5. Compare `SELECT count(*)` of each table with `tables.<name>.rows` in `manifest.json`, then
   point the `SUPABASE_URL` and `SUPABASE_KEY` secrets (and `.env`) at the new project.

What does not come back, by design of the export allowlist:

- **Commitment salts.** A revealed prediction still verifies: its reveal in the
  `attestation-log` branch carries the salt, and the restored `commitment_hash` matches it. An
  open prediction of the lost project can never be revealed — its salt existed only there.
- **Columns outside `EVIDENCE_COLUMNS`** restore as `NULL` or their default, among them the
  stale-quote mark (`stale_quote`), so a settlement on a stale quote counts in the feature store
  again, and the entry and invalidation prices.
- **Embeddings** — the RAG memory starts empty and fills with new predictions.
- **Every other table.** Price snapshots are gone, so each symbol's first cycle is a cold start;
  re-import the positions with `main_tools.py import-positions --apply` and re-enter the
  subscribers by hand.

### Managing the secrets (helper scripts)

Two `gh`-CLI helpers in [`scripts/`](../scripts/) keep the repo's GitHub Actions secrets in sync with your local `.env`. Both derive the required list **live** from `.github/workflows/*.yml` (every `secrets.*` / `vars.*` reference, minus the auto-injected `GITHUB_TOKEN`), so they never drift from the workflows:

```bash
# Audit: which required secrets/variables are set on the repo, which are missing.
# Read-only — never changes anything. Exit 0 = all present, 1 = something missing.
scripts/gh-secrets-check.sh

# Push: set the secrets/variables on the repo from .env. Prints a masked plan and
# asks for confirmation; values go to gh over stdin (never as args). --dry-run to preview.
scripts/gh-secrets-sync.sh [--dry-run] [-y]
```

Both accept `-R owner/repo` (default: auto-detected from the git remote) and `-f <env-file>` (default: `<repo>/.env`). `.env` keys that no workflow references are reported but never pushed. `FRED_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` and `SLACK_WEBHOOK_URL` *are* referenced (Fast Loop, Shock Watch, and the Slow Loop for the model probe's push, NIEZ-25), so add them to `.env` and the sync pushes them; without them the yield-curve vote and the push channels stay silent. Auth uses your `gh auth login` token, or a `GH_TOKEN` from `.env` if it is non-empty (a fine-grained PAT needs `Secrets: Read and write`). Any non-empty value takes precedence over the `gh` login, so leave `GH_TOKEN=` empty, as `.env.example` ships it, unless you mean to use a PAT: an expired token or a leftover placeholder stops both scripts with "gh is not authenticated".

A third helper reads the **production state** in one pass, without changing anything:

```bash
# GitHub: secret names vs. what workflows read, workflow states and last runs, branch
# protection, Pages, last Slow Loop outcome. Supabase (REST, GET only, the app's
# SUPABASE_URL/SUPABASE_KEY): key role, migrations from 017 on detected from the OpenAPI
# description (or per-object GET probes when the key cannot read it) by the table,
# column or function each one leaves behind, latest prediction/snapshot/fundamentals,
# training samples per symbol.
scripts/prod-state-check.sh > docs/concepts/prod-state-$(date +%F).md
```

It accepts `-R`, `-f`, `--skip-github` and `--skip-supabase`. It is read-only by construction: `gh` is only used to list and view, every Supabase request is `curl --get` with the key passed on stdin (never in argv or the output), and no database function is called; [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) enforces this on the script text. Every migration file has exactly one probe ([`tests/test_prod_state_probes.py`](../tests/test_prod_state_probes.py) compares the list with `supabase/migrations/`), so before you switch on a flag you can see which of its migrations the production database already has. A migration that only redefines a view or adds an index, a constraint, privileges or a function's `search_path` (026, 038, 039, 047, 048, 054, 058, 070 and 084) leaves nothing in the OpenAPI description; the report lists it as n/a, and the Supabase dashboard or `supabase migration list` answers for it. The output names secrets and counts subscribers, so keep it in the git-ignored `docs/concepts/`. Postgres version and the project plan are not exposed over REST and still need the Supabase dashboard.

### Secret redaction

Finnhub, Alpha Vantage and FRED take their key as a query parameter, and `requests` copies the full URL into the text of `HTTPError` and of network errors. That text used to reach the email (a failed symbol's error line), the `quota_alerts` table and the public GitHub Actions log. Every adapter with a key in the URL now calls `safe_get` and `raise_for_status_redacted` from [`src/infrastructure/adapters/_http.py`](../src/infrastructure/adapters/_http.py): they raise the same `requests` exception type with a redacted message and `from None`, so a logged traceback does not chain the original. `redact_secrets` replaces with `***` the `apikey`, `token`, `api_key` and `key` query parameters, authorization headers (`Authorization`, `Bearer …`, `apikey`, `x-api-key`), the Telegram bot path (`/bot<token>/`) and the Slack webhook URL. A failed symbol's error in the email passes through the same function, together with the literal secret values from `Settings` (`main_agent._analyze_symbols` hands them to each symbol), so a key echoed in an unrecognised shape is removed too. Telegram and Slack network failures put only the exception type into the alert and the log. In the tool-use loop (`tool_calling_openai.py`, `tool_calling_anthropic.py`) a tool's exception goes back to the paid model as `{"error": ...}` only after `redact_secrets`, which also gets the literal secret values from `Settings` that `main_agent.build_tool_use_adapter` injects; the adapter does not read `Settings` itself.

Logs get a second pass. Every entry point (`main_agent.py`, `main_trainer.py`, `main_watch.py`, `main_deadman.py`, `main_backtest.py`, `main_scorecard.py`, `main_tools.py`) configures logging only through `configure_logging("<entry point>", settings.secret_values())` from [`src/infrastructure/logging_setup.py`](../src/infrastructure/logging_setup.py), and [`tests/test_entrypoint_logging.py`](../tests/test_entrypoint_logging.py) fails when a `main_*.py` skips it or calls `logging.basicConfig` directly. It works like `basicConfig` and then gives every root handler a `RedactingFormatter` from [`src/infrastructure/logging_redaction.py`](../src/infrastructure/logging_redaction.py). The formatter runs `redact_secrets` on the finished record, traceback included: a logging `Filter` runs before the traceback text exists, so it cannot see it. Besides the patterns above, it replaces the literal value of every field in `SECRET_SETTINGS_FIELDS` (`src/config.py`), each key of `ALPHA_VANTAGE_API_KEYS` separately. Values shorter than 8 characters are not replaced, so a short placeholder or chat id does not wipe ordinary log text.

GitHub masks a secret in the job log only as the whole string, so one key out of the comma-separated `ALPHA_VANTAGE_API_KEYS` would still print in clear. Fast Loop, Slow Loop and Shock Watch therefore run [`scripts/mask-csv-secret.sh`](../scripts/mask-csv-secret.sh) right after checkout, before any Python step. It splits the secret on commas and trims each key the way `Settings` does, then registers `::add-mask::` for each one; the runner does not print that command, and the script echoes nothing else. Shock Watch and Deadman inject only a placeholder for that secret and have no such step. [`tests/test_workflow_env_wiring.py`](../tests/test_workflow_env_wiring.py) fails when a workflow maps `secrets.ALPHA_VANTAGE_API_KEYS` without the masking step ahead of Python.

The free jobs get only the secrets they use (BEZ-12). Shock Watch makes no paid call, so its Python step receives placeholders for `OPENAI_API_KEY` and `ALPHA_VANTAGE_API_KEYS` instead of the real keys, and [`tests/test_shock_watch_env_wiring.py`](../tests/test_shock_watch_env_wiring.py) fails when a real paid key comes back. The Slow Loop grants `contents: write` only to the trainer job, which creates the `model-latest` release and replaces its weights asset with the default `GITHUB_TOKEN` through the `gh` CLI (no third-party action); the workflow itself stays read-only. The Fast Loop downloads the asset with the read-only token of its workflow. To run locally with the shipped weights, download them once: `gh release download model-latest --pattern price_predictor.ubj --dir data/models --clobber`.

### Structured logs (NIEZ-21)

With `symbol_concurrency = 4` the lines of four symbols interleave in one job log, and only some messages name their symbol, so one symbol or one run could not be filtered out of a GitHub Actions log. `configure_logging` therefore adds a `RunContextFilter` to every root handler: each record carries `run_id` (`GITHUB_RUN_ID`, the same id as the email footer and `cycle_runs`; a random uuid4 hex outside GitHub Actions), `run_attempt` (`GITHUB_RUN_ATTEMPT`), `git_sha` (`GITHUB_SHA`), `entry_point` (`main_agent`, `main_trainer`, `main_watch`, `main_deadman`, `main_backtest`, `main_scorecard`, `main_tools`) and `symbol`. The symbol comes from a `ContextVar` that the symbol worker (`RunFastLoopCycle.process_one` in `src/application/use_cases/run_cycle.py`, with the `symbol_context` scope injected by `main_agent`) sets for its own call, in the worker thread of the parallel path too; the council's persona threads start with `contextvars.copy_context()`, so their lines carry the symbol as well. Outside a symbol the field is `null`.

The line format is chosen by the `LOG_FORMAT` environment variable: `json` writes one JSON object per line (`ts`, `level`, `logger`, `message`, the five fields above, and `exc_info` with the traceback in the same line), `text` keeps the human-readable `%(asctime)s %(levelname)s %(name)s — %(message)s`. Without the variable the format is `json` inside GitHub Actions (`GITHUB_ACTIONS=true`) and `text` everywhere else, and an unknown value falls back to `text`. JSON goes only to the handler the entry point creates itself; handlers that were already on the root logger (pytest's, for example) keep their format. Redaction runs on the finished JSON line, the same pass as for text.
