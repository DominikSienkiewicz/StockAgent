# Running and shipping it

Stack, prerequisites, local setup, and CI. Tunables live in [configuration.md](configuration.md).

## Tech stack

- **Python 3.12** with `from __future__ import annotations`
- **[uv](https://github.com/astral-sh/uv)** (Astral) — dependency manager, `uv sync --frozen` in CI
- **[LangGraph](https://langchain-ai.github.io/langgraph/) 1.3** — decision graph with `StateGraph[AgentState]`
- **OpenAI Python SDK** v2 — `gpt-5-mini` advisory council + `text-embedding-3-small` embeddings, JSON mode
- **Anthropic Python SDK** (extra `anthropic`) — Claude Sonnet 4.6, **default model for the main analysis**
- **XGBoost 3.2** + scikit-learn — sklearn-style API, native `.ubj` (UBJSON) format
- **Supabase** (Postgres 16 + pgvector) — REST via `supabase-py`, service_role key
- **Resend.com** — HTML email, sandbox sender without domain verification
- **Pydantic Settings** v2 — typed env vars from `.env` with validators (CSV → list)
- **requests** + `urllib3.Retry` — shared session with exponential backoff on 429 / 5xx
- **pytest 9** + **pytest-mock** — 640+ passing tests + skipped live-API / Docker-container tests
- **ruff** — lint (E, W, F, I, B, UP, SIM rule sets)
- **mypy** strict mode — every file fully typed
- **GitHub Actions CI** — ruff + mypy + pytest + SonarCloud scan on every push / PR

Architecture: **Hexagonal (Ports & Adapters) + DDD**. Domain (pure Python, zero deps) → Application (ports + use cases + LangGraph) → Infrastructure (adapters for API / DB / LLM / ML / email). Dependency arrow flows one way (`domain ← application ← infrastructure`).

## Prerequisites

- Python 3.12 + `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- macOS only: `brew install libomp` (required by XGBoost)
- **Finnhub** API key (free) — https://finnhub.io/dashboard
- **Alpha Vantage** API key(s), ideally several (free 25 req/day × N) — https://www.alphavantage.co/support/#api-key
- **OpenAI** API key — https://platform.openai.com/api-keys
- **Supabase** project (free tier) — https://supabase.com/dashboard
- **Resend** account + API key — https://resend.com (optional, for emails)
- **Anthropic API key — required** (default `LLM_PROVIDER=anthropic`; main analysis runs on Claude)

## Local setup

```bash
git clone <repo-url> StockAgent && cd StockAgent

# 1. Sync deps (creates .venv + installs packages + generates uv.lock)
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
#    The migrations, in order:
#   supabase/migrations/001_init.sql        (prediction_logs + ml_feature_store)
#   supabase/migrations/002_price_snapshots.sql  (price_snapshots — breaks cold-start)
#   supabase/migrations/003_add_embedding.sql    (embedding VECTOR(1536) + pgvector index)
#   supabase/migrations/004_align_ml_feature_store.sql  (7-feature XGBoost contract)
#   supabase/migrations/005_council_verdict.sql          (council verdicts schema)
#   supabase/migrations/006_fundamentals_cache.sql       (fundamentals_cache table with TTL)
#   supabase/migrations/007_council_votes.sql            (per-investor structured audit trail)
#   supabase/migrations/008_data_quality_flags.sql       (data_quality_flags on prediction_logs)
#   supabase/migrations/009_trend_correctness.sql        (is_trend_correct on prediction_logs + backfill)
#   supabase/migrations/010_quota_alerts.sql             (quota_alerts audit trail for the report banner)
#   supabase/migrations/011_match_news_embeddings.sql    (pgvector RPC for RAG retrieval over news embeddings)
#   supabase/migrations/012_ml_feature_store_return_target.sql  (target = 12h RETURN, not absolute price)
#   supabase/migrations/013_idempotency_and_pagination.sql      (timestamp_hour + UNIQUE(symbol,hour) → upsert idempotency)
#   supabase/migrations/014_subscribers.sql              (subscribers table — per-watchlist report fan-out)
#   supabase/migrations/015_persona_accuracy.sql         (persona_accuracy_weights RPC — adaptive council weighting)
#   supabase/migrations/016_confidence_calibration.sql   (confidence_calibration + calibration_insight columns)
#   supabase/migrations/017_vector_memory_regime.sql     (market regime on the pgvector memory)
#   supabase/migrations/018_persona_track_record.sql     (persona_accuracy_stats RPC — hit-rate + vote count for the leaderboard)

# ⚠️ Migration 012 is BREAKING for the ML model: it changes the training target
#    from absolute price to a 12h return, and the code now reconstructs price from
#    that return. A .ubj trained on the old (absolute-price) contract is INCOMPATIBLE.
#    After applying 012, DELETE the old model weights (data/models/price_predictor.ubj
#    if present in your deployment) — the Fast Loop falls back to the cold-start
#    "no change" baseline (= return 0) until the next Slow Loop retrains from scratch.

# 4. Smoke test (expect 650+ tests passing + ~22 skipped live/Docker tests)
uv run pytest

# 5. Single Fast Loop run
uv run python main_agent.py
```

Config is split in two: **secrets** live in `.env` (gitignored) / GitHub Secrets; **everything non-secret** (symbols, thresholds, models, providers, throttle, Risk Watch…) lives in committed [`config.toml`](../config.toml) — the single source of truth, read directly by `Settings`. No duplication across `.env` / workflow / repo variables.

`.env` — **secrets only**:

```env
# LLM (main analysis = Anthropic; embeddings + council = OpenAI)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...            # needs `uv sync --extra anthropic`

# Market data + News/Sentiment
FINNHUB_API_KEY=...
ALPHA_VANTAGE_API_KEYS=key1,key2,key3   # CSV — rotation on rate-limit

# Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...                     # service_role key

# Email — recipient is a secret; sender + on/off toggle live in config.toml
RESEND_API_KEY=re_...
DIGEST_TO_EMAIL=you@example.com
```

Everything else — `symbols`, `symbols_etf`, `volatility_threshold`, `crypto_symbols`, `council_llm_provider/model`, `risk_symbols`, `nbp_enabled`, `symbol_throttle_seconds`, `notifications_enabled`, `digest_from_email`, … — is edited in [`config.toml`](../config.toml). Any value can still be overridden by an environment variable (precedence: **env → `.env` → `config.toml` → code defaults**), e.g. `SYMBOL_THROTTLE_SECONDS=0` for a quick local run.

The secrets feed **GitHub Actions Secrets** — local and CI hit the **same Supabase database**, guaranteeing "works on my machine == works in prod" parity. `config.toml` is committed, so CI gets the same config without any repo variables.

## Build commands

```bash
uv sync                            # install deps
uv sync --extra anthropic          # + Anthropic SDK
uv run python main_agent.py        # one Fast Loop run (analysis + email)
uv run python main_trainer.py      # one Slow Loop run (XGBoost retrain)
uv run python main_backtest.py     # offline walk-forward backtest: per-symbol out-of-sample RMSE / hit-rate vs baseline (read-only, no model write)
uv run python -m src.tools.evaluate            # offline eval: hit-rate + RMSE vs baseline (read-only)
uv run python -m src.tools.evaluate --days 60  # custom evaluation window
uv run pytest                                   # full local suite
uv run pytest -m "not integration and not containers" # unit only
uv run pytest -m containers                    # Docker/Postgres schema tests
uv run pytest -m integration                   # live API tests (requires real keys)
uv run ruff check src tests main_agent.py main_trainer.py main_watch.py   # lint
uv run mypy src                    # type check (strict)
```

## GitHub Actions

Five workflows in [`.github/workflows/`](../.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](../.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest, plus a parallel **SonarCloud** scan job (`pytest --cov` → coverage upload) |
| [`fast_loop_12h.yml`](../.github/workflows/fast_loop_12h.yml) | _disabled_ (manual only) | — | Analysis + email report. **Daily schedule is currently paused** — the cron is commented out, so it runs only via manual `workflow_dispatch`. Re-enable by uncommenting the `schedule` block. |
| [`slow_loop_weekly.yml`](../.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining + commit new weights |
| [`shock_watch_hourly.yml`](../.github/workflows/shock_watch_hourly.yml) | `7 * * * *` | hourly at :07, **24/7** | Free shock alerts (`main_watch.py`) — no LLM, no paid API. Runs outside NYSE hours because BTC/ETH trade around the clock. |
| [`db_migrate.yml`](../.github/workflows/db_migrate.yml) | — (`workflow_dispatch` only) | — | Applies `supabase/migrations/` via the Supabase CLI. Deliberately manual: a broken migration must never fire on a cron. |

The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

**Repository secrets** (the only things CI needs beyond the committed `config.toml`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEYS`, `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY`, `DIGEST_TO_EMAIL`, `FRED_API_KEY`. Database migrations need one more, used only by the manual DB Migrate workflow: `SUPABASE_DB_URL` — the **direct Postgres connection string** (`postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres`), not the PostgREST `SUPABASE_URL`. The two are different things and only the former can run DDL. `ANTHROPIC_API_KEY` is required (main analysis runs on Claude); `DIGEST_TO_EMAIL` is a **secret** (was a variable — move it). `FRED_API_KEY` is **optional** — only needed when `yield_curve_enabled = true` (the FRED yield-curve alpha source). `SONAR_TOKEN` is **optional** — it enables the SonarCloud scan job; without it that job is skipped, so CI never breaks before Sonar is configured.

**SonarCloud — one-time setup.** [`ci.yml`](../.github/workflows/ci.yml) carries a `sonarcloud` job that runs `pytest --cov` and uploads code + coverage to [SonarCloud](https://sonarcloud.io/project/overview?id=DominikSienkiewicz_StockAgent) (keys live in [`sonar-project.properties`](../sonar-project.properties)). To enable it: (1) create a **`SONAR_TOKEN`** repository secret (SonarCloud → *My Account → Security*); (2) in SonarCloud, **disable *Automatic Analysis*** (*Administration → Analysis Method*) — CI-based and automatic analysis cannot both run, and only the CI scan uploads the coverage report the "Sonar way" gate (≥ 80% on new code) needs.

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

Both accept `-R owner/repo` (default: auto-detected from the git remote) and `-f <env-file>` (default: `<repo>/.env`). `.env` keys that no workflow references (e.g. `FRED_API_KEY`, `TELEGRAM_*`, `SLACK_WEBHOOK_URL`) are reported but never pushed. Auth uses your `gh auth login` token, or a `GH_TOKEN` from `.env` if present (a fine-grained PAT needs `Secrets: Read and write`).
