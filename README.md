# 📈 StockAgent
**Architected & Developed by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/)** *Principal AI Engineer | Full Stack Architect*

Autonomous financial agent that runs once daily (on trading days), monitors a curated portfolio of US stocks and ETFs, **fuses price action with curated financial sentiment** (Alpha Vantage NEWS_SENTIMENT), **detects cross-signal divergences** (LLM ↔ AV agreement < 0.3 = 🚨 fake news / manipulation), **learns from its own mistakes** (Self-Reflection backed by closed predictions in Supabase), and delivers a full Polish-language digest email — with charts, trade signals, accuracy history, and clickable top news straight from your inbox.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph 1.3](https://img.shields.io/badge/LangGraph-1.3-1C3C3C?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6B35?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Postgres+pgvector-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Resend](https://img.shields.io/badge/Resend-Email-000000?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Hexagonal+DDD-orange?style=for-the-badge)

## 🧠 The Vision: Signal over Noise

In the era of AI-driven information overload, a single price tick is a useless signal — what matters is the **covariance of sentiment, news, and historical predictions**. This agent treats the market as a system: once a day it pulls clean numerical data, enriches it through a curated financial filter, models hybridly (LLM for reasoning + XGBoost for quantitative inference), and **cyclically confronts itself with reality** through Self-Reflection. It's not a scraper, and it's not another "GPT predict stocks" — it's a **cognitive filter** designed for an aware decision-maker.

## How it works

```
Finnhub (US prices) ──► check_price ──► reflect ──► fetch_fundamentals ──► [Δ ≥ threshold?]
                         (+ snapshot     (Self-        (AlphaVantage           ├── no ──► ignore ──┐
                          to DB)          Reflection)   OVERVIEW+EARNINGS       │                   │
                                                        slow loop only)         └── yes ──► sentiment │
                                                                                            → news    │
                                                                                            → predict │
                                                                                            → council │  ← 15 investor personas
                                                                                            → save ───┤
                                                                                                      │
                          ┌───────────────────────────────────────────────────────────────────────────┘
                          ▼
                 accuracy_stats + resolved_predictions
                          ▼
                 HTML report (QuickChart) ──► Resend.com email
```

1. **Fetch price + snapshot** — `FinnhubAdapter` pulls the current quote for each US ticker. `check_price_node` saves a **price snapshot in every cycle** (`price_snapshots` table) so the next cycle always has a reference point — this breaks the cold-start deadlock. Free tier is US-only; dotted tickers (`CSPX.L`, `BAS.DE`) get a 403 and are handled gracefully.
2. **Self-Reflection (runs every cycle, before the volatility gate)** — `reflect_node` reads the last unverified prediction. It computes two independent domain metrics and persists both: `accuracy_score` (how close the price landed to the numeric target — feeds XGBoost training) and `is_trend_correct` (whether the directional call matched reality — feeds the report's hit-rate). If the prediction was directionally wrong, the LLM diagnoses why ("I ignored hawkish Fed signals"). The insight is injected into the next prediction's prompt as `<reflection_context>`. Decoupled from the volatility gate so every prediction is scored ~12h later, regardless of the current cycle's volatility.
3. **Fundamentals (slow loop refreshes, fast loop reads cache)** — `fetch_fundamentals_node` runs after `reflect` and before the volatility gate. The `FundamentalsPort` is implemented by `AlphaVantageFundamentalsAdapter` (2 API requests per stock symbol: `OVERVIEW` + `EARNINGS`) wrapped in `CachedFundamentalsAdapter` (decorator pattern). In the **slow loop**, real API calls populate the `fundamentals_cache` Supabase table. In the **fast loop**, a `NullFundamentalsAdapter` delegate skips API calls and reads from cache only. ETF symbols (configured via `SYMBOLS_ETF`) always skip fetching — they have no meaningful per-share EPS/P/E. The domain evaluates a deterministic `ValuationVerdict` (`UNDERVALUED / FAIR / OVERVALUED / UNKNOWN`) based primarily on PEG ratio, with PE/growth qualifiers; ETFs always get `UNKNOWN`. The verdict is surfaced to the Council prompt as a `Valuation snapshot` block and rendered as a dedicated **Wycena fundamentalna** section in the email report.
4. **Volatility gate** — `Asset.evaluate_volatility(delta, threshold)` lives in the pure domain (Hexagonal core). Δ < 2% → `ignore`, no paid APIs touched. **Domain decides, graph executes.**
5. **News + Sentiment** — `AlphaVantageClient` makes one request per ticker, rotating N API keys when one exhausts its 25 req/day quota. A `relevance ≥ 0.5` filter strips noise **before** anything reaches the LLM. Per ticker it returns a multi-feature dict: `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `av_sentiment_label`.
6. **LLM (cross-validation)** — GPT-4o (or Claude when `LLM_PROVIDER=anthropic`) receives pre-computed AV sentiment + headlines + reflection context + fundamentals valuation snapshot. It returns structured JSON: `trend_direction`, `confidence_score`, **`av_agreement`** (whether it agrees with AV — anything below 0.3 flags potential manipulation), `target_price_12h`, `reasoning`.
7. **ML hard prediction (local XGBoost)** — model lives in a `.ubj` file inside the repo (Local-First AI). Consumes 7 features: `price_delta`, `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `llm_trend_signal`, `av_llm_agreement`. On cold start (no trained weights yet) it falls back to a "no change" baseline instead of crashing.
8. **Advisory Council** — after `predict_node`, 15 legendary investor personas (Buffett, Graham, Soros, Lynch, Dalio, Munger, Fisher, Tudor Jones, Gross, Livermore, Wood, Burry, Marks, Druckenmiller, Greenblatt) each independently analyse the same data via parallel LLM calls (one worker per investor). The personas are **data, not code** — one JSON file per investor in [`data/council_personas/`](data/council_personas/), schema `{"name": str, "style": str}`. Adding / removing a council member = adding / removing a file. Loader (`src/infrastructure/persona_loader.py`) validates schema and uniqueness at startup; CLI walidator `uv run python -m src.tools.validate_personas` plus a pre-commit hook catch typos before runtime. A final "chairman" call synthesises a consensus `CouncilVerdict` (BUY/SELL/HOLD + `consensus_strength` + `dissenting_views`). Two volatility gates: the main one (`volatility_threshold`, default 2%) decides whether to run the prediction pipeline at all; the **council-specific gate** (`council_volatility_threshold`, default 3%) further filters out medium-Δ cycles where the 16 LLM calls would mostly return HOLD — set it to `0.0` to disable. Stored as JSONB in `prediction_logs.council_verdict` (legacy blob) **and** as one row per investor in `council_votes` (structured audit trail — query "how did Burry vote on NVDA in the last month" without parsing JSON). Rendered as a styled table in the email report.
9. **Persist** — `SupabaseRepository` writes the full record to `prediction_logs`. The news summary is embedded (OpenAI `text-embedding-3-small` → 1536-dim `pgvector`) for later RAG-style retrieval — graceful when embeddings are unavailable.
10. **Slow Loop (weekly cycle)** — `main_trainer.py` retrains XGBoost on resolved predictions (those with `accuracy_score`), commits the new weights back to the repo (Continual Learning). Also runs a fundamentals refresh step using `AlphaVantageFundamentalsAdapter` to repopulate `fundamentals_cache`.
11. **Deliver** — Polish-language HTML report via Resend with 2 charts (Δ12h + forecast), correlation scatter plot, trade signals sorted by `confidence × |Δ|`, risk signals with severity badges, day-over-day diff, and clickable news headlines. Two sections (council + fundamentals valuation) render via Jinja2 templates in `src/application/templates/` — autoescape on, the rest of the report still uses f-string composition in `report_builder.py` (incremental migration). The council template surfaces domain-level signals via `CouncilVerdict.is_split_decision()` (⚠️ PODZIELONA RADA badge), `has_strong_consensus()` (SILNY KONSENSUS badge), and `vote_distribution()` (BUY/SELL/HOLD count).

All HTTP adapters retry transient failures (429 / 5xx) with exponential backoff, and a single per-symbol error never aborts the whole cycle.

## Sources

| Source | Filter / parameters | Notes |
|---|---|---|
| **Finnhub** (`/quote`) | One request per ticker, 60 req/min free tier | US exchanges only. `.L` / `.DE` / `.NL` tickers → 403. |
| **Alpha Vantage** `NEWS_SENTIMENT` | One request per ticker, `limit=50`, client-side relevance filter ≥ 0.5 | **AND filter on tickers** = a separate request per symbol. Rate limit 25 req/day × N keys = N × 25/day. |
| **Alpha Vantage** `OVERVIEW` + `EARNINGS` | 2 requests per stock symbol per slow-loop refresh | Used by `AlphaVantageFundamentalsAdapter`. Free tier = 25 req/day → max ~12 stock symbols per refresh with a single key. Multi-key rotation via `ALPHA_VANTAGE_API_KEYS` (CSV) already supported. ETF symbols (`SYMBOLS_ETF`) are skipped. |
| **Supabase** (Postgres + pgvector) | `prediction_logs` + `price_snapshots` + `fundamentals_cache` tables + materialized view `ml_feature_store` | Service role key. RPC `refresh_ml_feature_store` is called before training. |
| **OpenAI** `chat.completions` | JSON mode (`response_format={"type":"json_object"}`), temperature 0.2 | Default LLM. Switch to Anthropic Claude via `LLM_PROVIDER=anthropic`. |
| **OpenAI** `embeddings` | `text-embedding-3-small` → 1536-dim vector | Embeds the news summary into `pgvector`. Wired regardless of `LLM_PROVIDER` since `OPENAI_API_KEY` is always required — works with the Anthropic LLM too. |
| **Anthropic** (optional) | Messages API, `claude-sonnet-4-6`, max_tokens 4096 | Adapter strips ```` ```json ... ``` ```` wrappers (Claude has no JSON mode). |
| **QuickChart.io** | URL-based, GET request | Zero setup, no dependency, works in every mail client. |
| **Resend.com** | Sandbox sender `onboarding@resend.dev` | Free tier 100 mails/day. No domain verification required — just a verified recipient. |

## Symbols (default portfolio of 22)

```text
AAPL,AMZN,GOOGL,MSFT,META,NVDA,TSLA,AMD,NET,PLTR,
ORCL,UBER,TSM,ASML,ASMIY,SAP,SIEGY,NVO,VT,QUAL,IHI,VB
```

Sector mix: 🤖 AI / semis (NVDA, AMD, TSM, ASML, ASMIY) · ☁️ cloud (MSFT, ORCL, NET, SAP, PLTR) · 📱 big tech (AAPL, AMZN, GOOGL, META) · 🚗 mobility (TSLA, UBER) · 🏭 industrial / pharma (SIEGY, NVO) · 📊 ETFs (VT, QUAL, IHI, VB).

Configurable via `SYMBOLS` in `.env` (CSV).

## Risk signals (three types)

`detect_risk_signals(results)` flags anomalies and renders them in the report with a colour-coded severity badge:

| Type | Trigger | Meaning |
|---|---|---|
| 🚨 **DIVERGENCE** | `\|delta\| > 2%` & sentiment opposes the price direction | Possible correction / pump / unusual market dynamics |
| 🚨 **AV_LLM_CONFLICT** | `av_agreement < 0.3` | LLM disagrees with AV — potential fake news / manipulation |
| ⚠️ **LOW_SIGNAL** | `news_volume < 3` | Decision built on a tiny news sample — low confidence |

Each signal lands in a dedicated section in the email with a red/yellow accent stripe.

## Trade signals (top BUY / SELL)

`build_trade_signals(results)` sorts saved predictions by **signal strength = `confidence × |expected_change| × 100`**. Top 5 surface in the email:

```
🟢 BUY      TSM   confidence 92%  forecast +5.05%  strength 4.65
🟢 BUY      NVDA  confidence 85%  forecast +2.95%  strength 2.51
🔴 SELL     SAP   confidence 71%  forecast -4.24%  strength 3.01
🟡 OBSERVE  ORCL  confidence 45%  forecast +0.39%  strength 0.18
```

Gives an immediate actionable view without digging through the table.

## Email anatomy

The email is a **17 kB+ structured digest in Polish** (HTML + plain-text fallback):

- 🕒 **NYSE session** — open / premarket / after-hours / weekend (`zoneinfo` America/New_York)
- 📊 **Three summary boxes** — predictions / ignored / errors
- 🎯 **Strongest signals** — top BUY/SELL, colour-coded (priority 1: actionable)
- 🚨 **Warning signals** — divergence + AV/LLM conflict + low signal
- 📊 **Portfolio mood** — average sentiment, most positive/negative, high-confidence count
- 📊 **Closed predictions (24h)** — ✅ Trafiona / ❌ Błędna by trend direction from previous cycles
- 🎯 **Accuracy history** — directional hit-rate over the last 30 days
- 📈 **Δ12h chart** (QuickChart bar chart)
- 🧠 **Self-Reflection** — lessons learned per symbol (purple box)
- 🔮 **Predictions table** — price · Δ12h · trend · forecast (12h) `+X.YZ%` · confidence · sentiment · news count
- 📈 **Forecast chart** (QuickChart bar chart)
- 💡 **Reasoning** + 📰 **Top news** (clickable `<a href>` links to the original articles)
- 📈 **Sentiment vs price correlation** (scatter plot)
- ⏸ **Ignored** (chip list) + ⚠️ **Errors**

All sections are **conditional** — they only render when data exists. The first cold-start cycle produces a concise report; once you have 50+ cycles behind you, accuracy sparklines and history kick in.

## Tech stack

- **Python 3.12** with `from __future__ import annotations`
- **[uv](https://github.com/astral-sh/uv)** (Astral) — dependency manager, `uv sync --frozen` in CI
- **[LangGraph](https://langchain-ai.github.io/langgraph/) 1.3** — decision graph with `StateGraph[AgentState]`
- **OpenAI Python SDK** v2 — GPT-4o (default) with `response_format={"type":"json_object"}`
- **Anthropic Python SDK** (optional extra) — Claude Sonnet 4.6 as an alternative
- **XGBoost 3.2** + scikit-learn — sklearn-style API, native `.ubj` (UBJSON) format
- **Supabase** (Postgres 16 + pgvector) — REST via `supabase-py`, service_role key
- **Resend.com** — HTML email, sandbox sender without domain verification
- **QuickChart.io** — URL-based charts (`<img src>` in HTML), zero dependency
- **Pydantic Settings** v2 — typed env vars from `.env` with validators (CSV → list)
- **requests** + `urllib3.Retry` — shared session with exponential backoff on 429 / 5xx
- **pytest 9** + **pytest-mock** — 376 passing tests + 5 skipped live API tests
- **ruff** — lint (E, W, F, I, B, UP, SIM rule sets)
- **mypy** strict mode — every file fully typed
- **GitHub Actions CI** — ruff + mypy + pytest on every push / PR

Architecture: **Hexagonal (Ports & Adapters) + DDD**. Domain (pure Python, zero deps) → Application (ports + use cases + LangGraph) → Infrastructure (adapters for API / DB / LLM / ML / email). Dependency arrow flows one way (`domain ← application ← infrastructure`).

## Prerequisites

- Python 3.12 + `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- macOS only: `brew install libomp` (required by XGBoost)
- **Finnhub** API key (free) — https://finnhub.io/dashboard
- **Alpha Vantage** API key(s), ideally several (free 25 req/day × N) — https://www.alphavantage.co/support/#api-key
- **OpenAI** API key — https://platform.openai.com/api-keys
- **Supabase** project (free tier) — https://supabase.com/dashboard
- **Resend** account + API key — https://resend.com (optional, for emails)
- *(Optional)* Anthropic API key — required when `LLM_PROVIDER=anthropic`

## Local setup

```bash
git clone <repo-url> StockAgent && cd StockAgent

# 1. Sync deps (creates .venv + installs packages + generates uv.lock)
uv sync

# 2. Configuration
cp .env.example .env
# fill in the keys (see the ".env keys" block below)

# 3. Database schema
# Open Supabase → SQL Editor → paste & run, in order:
#   migrations/001_init.sql        (prediction_logs + ml_feature_store)
#   migrations/002_price_snapshots.sql  (price_snapshots — breaks cold-start)
#   migrations/003_add_embedding.sql    (embedding VECTOR(1536) + pgvector index)
#   migrations/004_align_ml_feature_store.sql  (7-feature XGBoost contract)
#   migrations/005_investor_advisory_board.sql  (council verdicts schema)
#   migrations/006_fundamentals_cache.sql       (fundamentals_cache table with TTL)
#   migrations/007_council_votes.sql            (per-investor structured audit trail)
#   migrations/008_data_quality_flags.sql       (data_quality_flags on prediction_logs)
#   migrations/009_trend_correctness.sql        (is_trend_correct on prediction_logs + backfill)

# 4. Smoke test (expect 453 tests passing + 5 skipped live API tests)
uv run pytest

# 5. Single Fast Loop run
uv run python main_agent.py
```

`.env` keys:

```env
# LLM
LLM_PROVIDER=openai             # or anthropic
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...  # requires `uv sync --extra anthropic`

# Market + News
FINNHUB_API_KEY=
ALPHA_VANTAGE_API_KEYS=key1,key2,key3   # CSV — rotation on rate-limit

# DB
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...                     # service_role key

# Email (optional)
NOTIFICATIONS_ENABLED=true
RESEND_API_KEY=re_...
DIGEST_FROM_EMAIL=onboarding@resend.dev   # sandbox or your own verified domain
DIGEST_TO_EMAIL=you@example.com

# Agent
SYMBOLS=AAPL,AMZN,GOOGL,MSFT,META,NVDA,TSLA,AMD,NET,PLTR,ORCL,UBER,TSM,ASML,ASMIY,SAP,SIEGY,NVO,VT,QUAL,IHI,VB
SYMBOLS_ETF=VT,QUAL,IHI,VB          # CSV of ETF tickers — skip fundamentals fetch
VOLATILITY_THRESHOLD=0.02
ML_MODEL_PATH=data/models/price_predictor.ubj
```

The same env vars feed **GitHub Actions secrets** — local and CI hit the **same Supabase database**, guaranteeing "works on my machine == works in prod" parity.

## Build commands

```bash
uv sync                            # install deps
uv sync --extra anthropic          # + Anthropic SDK
uv run python main_agent.py        # one Fast Loop run (analysis + email)
uv run python main_trainer.py      # one Slow Loop run (XGBoost retrain)
uv run pytest                                   # full local suite
uv run pytest -m "not integration and not containers" # unit only
uv run pytest -m containers                    # Docker/Postgres schema tests
uv run pytest -m integration                   # live API tests (requires real keys)
uv run ruff check src tests        # lint
uv run mypy src                    # type check (strict)
```

## GitHub Actions

Three workflows in [`.github/workflows/`](.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest (unit only) |
| [`fast_loop_12h.yml`](.github/workflows/fast_loop_12h.yml) | `30 5 * * 1-5` | **07:30** (summer) / 06:30 (winter), **Mon–Fri only** | Analysis + email report (skipped on weekends — market closed) |
| [`slow_loop_weekly.yml`](.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining + commit new weights |

The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

**Repository secrets:** `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEYS`, `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY` (optional: `ANTHROPIC_API_KEY`).
**Repository variables:** `NOTIFICATIONS_ENABLED`, `DIGEST_FROM_EMAIL`, `DIGEST_TO_EMAIL`, `SYMBOLS`, `VOLATILITY_THRESHOLD` (all optional with sensible defaults).

## Configuration

All tunable parameters live in [`src/config.py`](src/config.py) as a `Settings` Pydantic model with validators. Override via env vars or `.env`:

| Field | Default | Description |
|---|---|---|
| `llm_provider` | `openai` | `openai` or `anthropic` |
| `council_llm_provider` | `None` | Override LLM provider for the advisory council only (heterogeneous strategy: cheap model for 15-persona council, frontier for main analysis). `None` → reuses `llm_provider`. |
| `council_llm_model` | `None` | Override model name for the council adapter (e.g. `gpt-4o-mini`, `claude-haiku-4-5`). `None` → provider default. |
| `council_personas_dir` | `data/council_personas` | Directory with one JSON file per council member (`{"name": str, "style": str}`). Validate with `uv run python -m src.tools.validate_personas`. |
| `volatility_threshold` | `0.02` | Threshold that triggers full analysis (2%) |
| `council_volatility_threshold` | `0.03` | Extra threshold for the advisory council (15 LLM calls). Below this Δ the council is skipped even if the main gate passed. `0.0` disables. |
| `symbols` | `[AAPL, MSFT, NVDA]` | Monitored tickers (CSV in env — override with the 22-symbol portfolio) |
| `alpha_vantage_api_keys` | `[]` | CSV of keys for rotation on rate-limit |
| `symbols_etf` | `[]` | CSV of tickers classified as ETFs (e.g. `VT,QUAL,IHI,VB`). ETFs skip fundamentals fetching (no meaningful per-share EPS/P/E) and always receive `ValuationVerdict.UNKNOWN`. |
| `ml_model_path` | `data/models/price_predictor.ubj` | XGBoost weights file |
| `notifications_enabled` | `false` | Enables email delivery |
| `digest_from_email` | `onboarding@resend.dev` | Resend sandbox sender |

Plus internal constants in `report_builder.py` (`DIVERGENCE_PRICE_THRESHOLD = 0.02`, `AV_LLM_CONFLICT_THRESHOLD = 0.3`, `LOW_SIGNAL_NEWS_THRESHOLD = 3`, `HIGH_RELEVANCE_BAR = 0.8`) and `xgboost_local.py` (hyperparameters `max_depth=4, eta=0.1, subsample=0.8`).

## Disclaimer

Educational / proof-of-concept project demonstrating expertise in Agentic AI, Hexagonal Architecture, and FinOps. **Generated predictions do not constitute investment advice.** Make financial decisions at your own risk, after consulting a licensed advisor.

## License

[MIT](LICENSE)
