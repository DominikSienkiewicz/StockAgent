# 📈 StockAgent
**Architected & Developed by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/)** *Principal AI Engineer | Full Stack Architect*

Autonomous financial agent that runs every 12h, monitors a curated portfolio of US stocks and ETFs, **fuses price action with curated financial sentiment** (Alpha Vantage NEWS_SENTIMENT), **detects cross-signal divergences** (LLM ↔ AV agreement < 0.3 = 🚨 fake news / manipulation), **learns from its own mistakes** (Self-Reflection backed by closed predictions in Supabase), and delivers a full Polish-language digest email — with charts, trade signals, accuracy history, and clickable top news straight from your inbox.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph 1.3](https://img.shields.io/badge/LangGraph-1.3-1C3C3C?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6B35?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Postgres+pgvector-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Resend](https://img.shields.io/badge/Resend-Email-000000?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Hexagonal+DDD-orange?style=for-the-badge)

## 🧠 The Vision: Signal over Noise

In the era of AI-driven information overload, a single price tick is a useless signal — what matters is the **covariance of sentiment, news, and historical predictions**. This agent treats the market as a system: every 12h it pulls clean numerical data, enriches it through a curated financial filter, models hybridly (LLM for reasoning + XGBoost for quantitative inference), and **cyclically confronts itself with reality** through Self-Reflection. It's not a scraper, and it's not another "GPT predict stocks" — it's a **cognitive filter** designed for an aware decision-maker.

## How it works

```
Finnhub (US prices) ──► check_price ──► reflect ──► [Δ ≥ threshold?]
                         (+ snapshot     (Self-        ├── no ──► ignore ──┐
                          to DB)          Reflection)  │                   │
                                                       └── yes ──► sentiment │
                                                                   → news    │
                                                                   → predict │
                                                                   → save ───┤
                                                                             │
                          ┌──────────────────────────────────────────────────┘
                          ▼
                 accuracy_stats + resolved_predictions
                          ▼
                 HTML report (QuickChart) ──► Resend.com email
```

1. **Fetch price + snapshot** — `FinnhubAdapter` pulls the current quote for each US ticker. `check_price_node` saves a **price snapshot in every cycle** (`price_snapshots` table) so the next cycle always has a reference point — this breaks the cold-start deadlock. Free tier is US-only; dotted tickers (`CSPX.L`, `BAS.DE`) get a 403 and are handled gracefully.
2. **Self-Reflection (runs every cycle, before the volatility gate)** — `reflect_node` reads the last unverified prediction. It computes `accuracy_score` via the domain, persists it, and — if the prediction was wrong — the LLM diagnoses why ("I ignored hawkish Fed signals"). The insight is injected into the next prediction's prompt as `<reflection_context>`. Decoupled from the volatility gate so every prediction is scored ~12h later, regardless of the current cycle's volatility.
3. **Volatility gate** — `Asset.evaluate_volatility(delta, threshold)` lives in the pure domain (Hexagonal core). Δ < 2% → `ignore`, no paid APIs touched. **Domain decides, graph executes.**
4. **News + Sentiment** — `AlphaVantageClient` makes one request per ticker, rotating N API keys when one exhausts its 25 req/day quota. A `relevance ≥ 0.5` filter strips noise **before** anything reaches the LLM. Per ticker it returns a multi-feature dict: `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `av_sentiment_label`.
5. **LLM (cross-validation)** — GPT-4o (or Claude when `LLM_PROVIDER=anthropic`) receives pre-computed AV sentiment + headlines + reflection context. It returns structured JSON: `trend_direction`, `confidence_score`, **`av_agreement`** (whether it agrees with AV — anything below 0.3 flags potential manipulation), `target_price_12h`, `reasoning`.
6. **ML hard prediction (local XGBoost)** — model lives in a `.ubj` file inside the repo (Local-First AI). Consumes 7 features: `price_delta`, `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `llm_trend_signal`, `av_llm_agreement`. On cold start (no trained weights yet) it falls back to a "no change" baseline instead of crashing.
7. **Persist** — `SupabaseRepository` writes the full record to `prediction_logs`. The news summary is embedded (OpenAI `text-embedding-3-small` → 1536-dim `pgvector`) for later RAG-style retrieval — graceful when embeddings are unavailable.
8. **Slow Loop (weekly cycle)** — `main_trainer.py` retrains XGBoost on resolved predictions (those with `accuracy_score`), commits the new weights back to the repo (Continual Learning).
9. **Deliver** — Polish-language HTML report via Resend with 2 charts (Δ12h + forecast), correlation scatter plot, trade signals sorted by `confidence × |Δ|`, risk signals with severity badges, day-over-day diff, and clickable news headlines.

All HTTP adapters retry transient failures (429 / 5xx) with exponential backoff, and a single per-symbol error never aborts the whole cycle.

## Sources

| Source | Filter / parameters | Notes |
|---|---|---|
| **Finnhub** (`/quote`) | One request per ticker, 60 req/min free tier | US exchanges only. `.L` / `.DE` / `.NL` tickers → 403. |
| **Alpha Vantage** `NEWS_SENTIMENT` | One request per ticker, `limit=50`, client-side relevance filter ≥ 0.5 | **AND filter on tickers** = a separate request per symbol. Rate limit 25 req/day × N keys = N × 25/day. |
| **Supabase** (Postgres + pgvector) | `prediction_logs` + `price_snapshots` tables + materialized view `ml_feature_store` | Service role key. RPC `refresh_ml_feature_store` is called before training. |
| **OpenAI** `chat.completions` | JSON mode (`response_format={"type":"json_object"}`), temperature 0.2 | Default LLM. Switch to Anthropic Claude via `LLM_PROVIDER=anthropic`. |
| **OpenAI** `embeddings` | `text-embedding-3-small` → 1536-dim vector | Embeds the news summary into `pgvector`. Disabled when `LLM_PROVIDER=anthropic`. |
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
- 📊 **Closed predictions (24h)** — ✅ correct / ❌ wrong from previous cycles
- 🎯 **Accuracy history** — mean accuracy over the last 30 days
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
- **pytest 9** + **pytest-mock** — 266 tests (mocked requests, supabase, OpenAI)
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

# 4. Smoke test (expect 259 tests passing + 7 skipped integration)
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
uv run pytest                      # 266 tests
uv run pytest -m "not integration" # unit only (what CI runs)
uv run pytest -m integration       # integration only (requires real keys)
uv run ruff check src tests        # lint
uv run mypy src                    # type check (strict)
```

## GitHub Actions

Three workflows in [`.github/workflows/`](.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest (unit only) |
| [`fast_loop_12h.yml`](.github/workflows/fast_loop_12h.yml) | `30 6 * * *` + `0 13 * * *` | **08:30 / 15:00** (summer) / 07:30 / 14:00 (winter) | Analysis + email report |
| [`slow_loop_weekly.yml`](.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining + commit new weights |

The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

**Repository secrets:** `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEYS`, `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY` (optional: `ANTHROPIC_API_KEY`).
**Repository variables:** `NOTIFICATIONS_ENABLED`, `DIGEST_FROM_EMAIL`, `DIGEST_TO_EMAIL`, `SYMBOLS`, `VOLATILITY_THRESHOLD` (all optional with sensible defaults).

## Configuration

All tunable parameters live in [`src/config.py`](src/config.py) as a `Settings` Pydantic model with validators. Override via env vars or `.env`:

| Field | Default | Description |
|---|---|---|
| `llm_provider` | `openai` | `openai` or `anthropic` |
| `volatility_threshold` | `0.02` | Threshold that triggers full analysis (2%) |
| `symbols` | `[AAPL, MSFT, NVDA]` | Monitored tickers (CSV in env — override with the 22-symbol portfolio) |
| `alpha_vantage_api_keys` | `[]` | CSV of keys for rotation on rate-limit |
| `ml_model_path` | `data/models/price_predictor.ubj` | XGBoost weights file |
| `notifications_enabled` | `false` | Enables email delivery |
| `digest_from_email` | `onboarding@resend.dev` | Resend sandbox sender |

Plus internal constants in `report_builder.py` (`DIVERGENCE_PRICE_THRESHOLD = 0.02`, `AV_LLM_CONFLICT_THRESHOLD = 0.3`, `LOW_SIGNAL_NEWS_THRESHOLD = 3`, `HIGH_RELEVANCE_BAR = 0.8`) and `xgboost_local.py` (hyperparameters `max_depth=4, eta=0.1, subsample=0.8`).

## Disclaimer

Educational / proof-of-concept project demonstrating expertise in Agentic AI, Hexagonal Architecture, and FinOps. **Generated predictions do not constitute investment advice.** Make financial decisions at your own risk, after consulting a licensed advisor.

## License

[MIT](LICENSE)
