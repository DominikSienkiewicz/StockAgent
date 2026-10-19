# Running and shipping it

Stack, prerequisites, local setup, and CI. Tunables live in [configuration.md](configuration.md).

## Tech stack

- **Python 3.12** with `from __future__ import annotations`
- **[uv](https://github.com/astral-sh/uv)** (Astral) — dependency manager, `uv sync --frozen` in CI
- **[LangGraph](https://langchain-ai.github.io/langgraph/) 1.2** — decision graph with `StateGraph[AgentState]`
- **OpenAI Python SDK** v2 — `gpt-5-mini` advisory council + `text-embedding-3-small` embeddings, JSON mode
- **Anthropic Python SDK** (core dependency) — Claude Sonnet 4.6, **default model for the main analysis**
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
ANTHROPIC_API_KEY=sk-ant-...            # main analysis + weekly recap (llm_provider = anthropic)

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
uv sync                            # install deps (Anthropic SDK included; `--extra anthropic` is a kept alias)
uv run python main_agent.py        # one Fast Loop run (analysis + email)
uv run python main_trainer.py      # one Slow Loop run (XGBoost retrain)
uv run python main_tools.py --help # one-off tools: one subcommand per tool (see "One-off tools" below)
uv run python main_tools.py import-positions --csv positions.csv          # dry-run: diff of the CSV against the positions table (writes nothing)
uv run python main_tools.py import-positions --csv positions.csv --apply  # replace the positions table with the CSV, as_of = now
uv run python main_backtest.py     # offline walk-forward backtest (folds purge the 12h label horizon): per-symbol out-of-sample RMSE / hit-rate vs baseline, plus per-fold rank IC, sign-strategy return and turnover; logs mean IC, its sd and t-stat next to the fold count (read-only, no model write)
uv run python main_tools.py evaluate           # offline eval: hit-rate + RMSE vs baseline, directional n_eff, naive baselines and binomial-test verdict (read-only)
uv run python main_tools.py evaluate --days 60 # custom evaluation window
uv run python main_tools.py evaluate --by-recipe  # + directional hit rate per recipe version (needs migration 030)
uv run python main_tools.py evaluate --by-universe  # + directional hit rate per symbol-universe version (from the v2 receipt)
uv run python main_tools.py evaluate --by component  # + ablation: LLM trend vs council recommendation vs XGBoost direction on the same rows (see below)
uv run python main_tools.py evaluate --by regime --by asset_class  # + LLM hit rate per segment; repeatable, also confidence_bucket, alpha_fusion_sign, edge_label
uv run python main_tools.py evaluate --echo  # + momentum echo test: is the LLM trend just the sign of the move that opened the gate? (needs migration 034)
uv run pytest                                   # full local suite
uv run pytest -m "not integration and not containers" # unit only
uv run pytest -m containers                    # Docker/Postgres schema tests
uv run pytest -m integration                   # live API tests (requires real keys)
uv run pytest tests/application/test_report_builder_snapshot.py --update-golden  # rewrite the email golden snapshots after an intended email change
uv run ruff check .                # lint (same command as CI)
uv run mypy                        # type check (strict) on [tool.mypy].files: src + every main_*.py
```

Every evaluation number (the headline, `--by` segments and `--echo`) skips the placeholder rows saved after a failed main LLM call (`llm_analysis_failed`, the neutral "SIDEWAYS 50%"), with the same filter the email and the Track Record use: they are not predictions.

`--by` segments the offline evaluation (WIA-55). `component` scores, on the same settled rows, the directional hit rate of the LLM trend (the stored `is_trend_correct`), the council's `final_recommendation` (BUY → up, SELL → down, HOLD is no bet; rows without a verdict are left out) and the XGBoost direction (target above or below the price; a target equal to the price is the cold-start "no change" baseline of a model without weights, so no bet). The context segments split the LLM hit rate by `regime` (untagged rows form `UNKNOWN`), `asset_class`, `confidence_bucket` (0.00-0.60, 0.60-0.70, 0.70-0.80, 0.80-1.00, plus a separate `0.50 (domyślna)` bucket for exactly 0.5, the value saved when the LLM gave no confidence, so the default does not mix with genuinely low confidence; a genuine 0.5 lands there too, since the database cannot tell them apart), `alpha_fusion_sign` and `edge_label` (the `evaluate_edge` label of the stored `edge_sigma`; no value is `NO_SIGNAL`). Every segment prints its number of directional bets `n`, `n_eff`, a Wilson 95% interval on `n_eff`, the best naive "always BULLISH / BEARISH" baseline on the same rows and the binomial p-value against it, with Benjamini-Hochberg FDR (q = 0.10) across the segments of one `--by`; a segment with fewer than 10 bets is flagged "za mało danych". A segment's column is read only when it is asked for, so the default run works on any database, and each one needs its own migration: `council_verdict` `005`, `confidence_score` `016`, `regime` `017`, `alpha_fusion_score` `022`, `edge_sigma` `025`. `prediction_logs` stores no asset class, so `asset_class` classifies each symbol from the **current** `crypto_symbols` and `symbols_etf`: a ticker moved between lists changes segment retroactively. Read-only, zero paid calls; nothing of it reaches the email.

`--echo` is the momentum echo test (EXP-01). The gate only lets through moves of at least the threshold, and the LLM sees that same move in its prompt, so a trend that mostly repeats (or reverses) its sign is a call a free rule makes at no cost. On the settled predictions with a `gate_delta` (migration `034`, history backfilled from `price_snapshots`; see `gate_delta_enabled` in [configuration](configuration.md)) it prints (a) the share of echo (`predicted_trend` = sign of the gate delta) and of reversal (= the opposite sign); (b) on the same rows (a directional LLM call, a non-zero delta, a known move), the hit rate of the LLM and of the free baselines "sign of the move" and "opposite sign", with `n` and `n_eff`; (c) the LLM hit rate only where it disagrees with momentum. The comparison is paired: where the LLM and a baseline make the same call they hit or miss together, so only the rows where they disagree decide. There one side's hit is the other's miss, so each comparison is a sign test of the LLM's wins against 0.5 on `n_eff` (one bet per correlated day; a zero move is a tie and drops out): LLM vs "sign of the move" on the reversals, LLM vs "opposite sign" on the echoes, both corrected with Benjamini-Hochberg FDR (q = 0.10). Fewer than 10 compared rows are flagged "za mało danych". Before the migration the run prints "kolumna gate_delta niedostępna — uruchom migrację 034" and the rest of the report instead of failing. The result lives only in `evaluate`, not in the weekly recap or the email (owner's decision).

**One-off tools (ARCH-39).** [`main_tools.py`](../main_tools.py) is the composition root of the one-off CLI tools: every tool is one argparse subcommand (`uv run python main_tools.py <command> --help`), only `main_tools.py` builds the `Settings` and the adapters a tool needs, and the tool's logic (parsing, validation, diffs, metrics, formatting) stays in pure functions under `src/tools/`. A separate `main_<x>.py` is reserved for entry points with their own workflow (Fast Loop, Slow Loop, shock watch, deadman, backtest). `[tool.mypy].files` lists every `main_*.py`, so the tools are type-checked like the other entry points, and `tests/test_architecture.py` fails when a module under `src/tools/` imports `src.config`, an adapter or an LLM client. The offline evaluation moved here without an alias: `python -m src.tools.evaluate` is now `python main_tools.py evaluate`, with the same options.

**Importing positions (PORT-18).** `import-positions` replaces the hand-maintained `positions` table with a CSV file, so the STALE badge stops being the default state and positions get a real purchase date. The file is UTF-8, comma-separated, with a header row; one row is one lot, and several lots of one symbol (bought on different dates) are allowed:

```csv
symbol,quantity,avg_cost,currency,purchase_date
NVDA,10,118.50,USD,2025-03-01
SAP.DE,4,182.10,EUR,
```

`symbol` is the ticker as in `config.toml` (upper-cased), `quantity` and `avg_cost` are positive numbers with a dot as the decimal separator (`avg_cost` per unit, in `currency`), `currency` is a three-letter ISO 4217 code, and `purchase_date` is an optional column and value in ISO `YYYY-MM-DD`, not in the future; empty means unknown, and the report then says the purchase-date rate is unknown instead of guessing one. Every row is validated before the database is touched, and all errors are printed with their line numbers. A file without a single position is an error, not a request to empty the table. Without `--apply` the command prints the diff against the current table (`+` added, `-` removed, `=` unchanged lots) and writes nothing; reading the table is strict, so a database error stops it instead of showing everything as new. With `--apply` it inserts the file's lots with `as_of` = now and then deletes the rows that existed before, so a failed insert leaves the old portfolio in place. Writing `currency` and `purchase_date` needs migration `035` and `positions_currency_enabled = true` (see [configuration](configuration.md)); until then `--apply` refuses a file with a non-USD currency or a purchase date. The import makes no paid calls.

**Email golden snapshot (TEST-02).** `tests/application/test_report_builder_snapshot.py` renders three synthetic cycles with a frozen clock and fixed data: a full cycle (saved stocks with council, news and provenance badges, crypto, skipped and unsupported tickers, a source error, plus Risk Watch, your portfolio, Track Record, the model scorecard, cycle cost and the run footer), a quiet day where the volatility gate skips everything, and the day-1 cold start. It compares the HTML and the plain-text email character for character with `tests/application/snapshots/<scenario>.html` and `.txt`, and on a difference prints a unified diff. The HTML is stored one tag per line so a change reads as an ordinary diff; that layout is reversible and the test checks it, so the comparison stays exact. Nothing in it depends on the machine's time zone, locale or hash seed. When the email changes on purpose, rewrite the snapshots with the `--update-golden` command above (the only thing that writes them), review `git diff tests/application/snapshots/` and commit them together with the change; never edit them by hand to turn the test green.

## GitHub Actions

Five workflows in [`.github/workflows/`](../.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](../.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest, plus a parallel **SonarCloud** scan job (`pytest --cov` → coverage upload) |
| [`fast_loop_12h.yml`](../.github/workflows/fast_loop_12h.yml) | _disabled_ (manual only) | — | Analysis + email report. **Daily schedule is currently paused** — the cron is commented out, so it runs only via manual `workflow_dispatch`. Re-enable by uncommenting the `schedule` block. |
| [`slow_loop_weekly.yml`](../.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining; commits `data/models/price_predictor.ubj` only when a candidate beats the naive baseline |
| [`shock_watch_hourly.yml`](../.github/workflows/shock_watch_hourly.yml) | _disabled_ (manual only) | — | Free shock alerts (`main_watch.py`) — no LLM, no paid API. Designed for an hourly 24/7 cron (`7 * * * *`, BTC/ETH trade around the clock), **paused together with the Fast Loop**: its only reference prices are Fast Loop snapshots, and a reference older than 30 h suppresses every alert. A run where every reference is stale logs a `Shock watch ślepy` WARNING, a run where every symbol fails (e.g. the database is down) exits 1, and without a push channel (`TELEGRAM_*` or `SLACK_WEBHOOK_URL`) it logs an error and neither sends nor records a shock, so the daily debounce stays free. Re-enable both crons together. |
| [`deadman_daily.yml`](../.github/workflows/deadman_daily.yml) | `17 9 * * *` | 11:17 (summer) | Free liveness check (`main_deadman.py`) — reads the newest `price_snapshots` and `model_scorecards` rows and mails through Resend when a job has been silent longer than `liveness_*_max_gap_h` in `config.toml` (`0` = deliberately paused, e.g. the Fast Loop). One run a day means at most one alert a day; an unreadable database exits 1, which triggers the failure mail. |
| [`db_migrate.yml`](../.github/workflows/db_migrate.yml) | — (`workflow_dispatch` only) | — | Applies `supabase/migrations/` via the Supabase CLI. Deliberately manual: a broken migration must never fire on a cron. |

Fast Loop, Slow Loop, Shock Watch and DB Migrate each end with a step that runs only when the job failed or was cancelled (a timeout included) and mails the run link through Resend — [`scripts/notify-failure.sh`](../scripts/notify-failure.sh), plain bash with the same `RESEND_API_KEY` / `DIGEST_TO_EMAIL` secrets as the digest, so it still fires when dependency installation is what broke. GitHub's own failure mail reaches only whoever last edited the cron. The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

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

Both accept `-R owner/repo` (default: auto-detected from the git remote) and `-f <env-file>` (default: `<repo>/.env`). `.env` keys that no workflow references are reported but never pushed. `FRED_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` and `SLACK_WEBHOOK_URL` *are* referenced (Fast Loop and Shock Watch), so add them to `.env` and the sync pushes them; without them the yield-curve vote and the push channels stay silent. Auth uses your `gh auth login` token, or a `GH_TOKEN` from `.env` if present (a fine-grained PAT needs `Secrets: Read and write`).

A third helper reads the **production state** in one pass, without changing anything:

```bash
# GitHub: secret names vs. what workflows read, workflow states and last runs, branch
# protection, Pages, last Slow Loop outcome. Supabase (REST, GET only, the app's
# SUPABASE_URL/SUPABASE_KEY): key role, migrations 017–025 detected from the OpenAPI
# description (or per-object GET probes when the key cannot read it), latest
# prediction/snapshot/fundamentals, training samples per symbol.
scripts/prod-state-check.sh > docs/concepts/prod-state-$(date +%F).md
```

It accepts `-R`, `-f`, `--skip-github` and `--skip-supabase`. It is read-only by construction: `gh` is only used to list and view, every Supabase request is `curl --get` with the key passed on stdin (never in argv or the output), and no database function is called; [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py) enforces this on the script text. The output names secrets and counts subscribers, so keep it in the git-ignored `docs/concepts/`. Postgres version and the project plan are not exposed over REST and still need the Supabase dashboard.
