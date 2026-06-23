# 📈 StockAgent
**Architected & Developed by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/)** *Principal AI Engineer | Full Stack Architect*

Autonomous financial agent that runs once daily (on trading days), monitors a curated portfolio of US stocks and ETFs, **fuses price action with curated financial sentiment** (Alpha Vantage NEWS_SENTIMENT), **detects cross-signal divergences** (LLM ↔ AV agreement < 0.3 = 🚨 fake news / manipulation), **learns from its own mistakes** (Self-Reflection backed by closed predictions in Supabase), and delivers a full Polish-language digest email — with charts, trade signals, accuracy history, and clickable top news straight from your inbox.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph 1.3](https://img.shields.io/badge/LangGraph-1.3-1C3C3C?style=for-the-badge)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Sonnet_4.6-D97757?style=for-the-badge&logo=anthropic&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings_+_council-412991?style=for-the-badge&logo=openai&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6B35?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Postgres+pgvector-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Resend](https://img.shields.io/badge/Resend-Email-000000?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Hexagonal+DDD-orange?style=for-the-badge)
![Quality Gate](https://img.shields.io/sonar/quality_gate/DominikSienkiewicz_StockAgent?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud&label=Quality%20Gate)
![Coverage](https://img.shields.io/sonar/coverage/DominikSienkiewicz_StockAgent?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud)

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
                                                                                            → council │  ← 7 investor personas
                                                                                            → save ───┤
                                                                                                      │
                          ┌───────────────────────────────────────────────────────────────────────────┘
                          ▼
                 accuracy_stats + resolved_predictions
                          ▼
                 HTML report (QuickChart) ──► Resend.com email
```

1. **Fetch price + snapshot** — `FinnhubAdapter` pulls the current quote for each US ticker. `check_price_node` saves a **price snapshot in every cycle** (`price_snapshots` table) so the next cycle always has a reference point — this breaks the cold-start deadlock. Free tier is US-only; dotted tickers (`CSPX.L`, `BAS.DE`) get a 403 and are handled gracefully.
2. **Self-Reflection (runs every cycle, before the volatility gate)** — `reflect_node` reads the last unverified prediction. It computes two independent domain metrics and persists both: `accuracy_score` (how close the price landed to the numeric target — feeds XGBoost training) and `is_trend_correct` (whether the directional call matched reality — feeds the report's hit-rate). If the prediction was directionally wrong, the LLM diagnoses why ("I ignored hawkish Fed signals"). The insight is injected into the next prediction's prompt as `<reflection_context>`. The **cheap bookkeeping** (`accuracy_score` / `is_trend_correct` / persisting the outcome) runs every cycle regardless of volatility, so every prediction is scored on the **next cycle**; the **paid `analyze_mistake` LLM diagnosis** is now gated by the same volatility decision as the prediction pipeline (on a flat, below-threshold cycle it records the outcome with a "diagnosis deferred" note instead of spending an LLM call — a FinOps bound on backward-looking spend). **Prediction horizon, honestly:** the loop runs once per trading day (not every 12h despite the legacy `*_12h` names), so a prediction is resolved ~24h later on weekdays and ~72h later across a weekend. A `reflection_min_age_hours` guard (config, default `0`; production `6`) skips predictions too fresh to score fairly — so an overlapping manual `workflow_dispatch` can't prematurely close a prediction made minutes earlier and pollute the accuracy signal.
3. **Fundamentals (slow loop refreshes, fast loop reads cache)** — `fetch_fundamentals_node` runs after `reflect` and before the volatility gate. The `FundamentalsPort` is implemented by `AlphaVantageFundamentalsAdapter` (2 API requests per stock symbol: `OVERVIEW` + `EARNINGS`) wrapped in `CachedFundamentalsAdapter` (decorator pattern). In the **slow loop**, real API calls populate the `fundamentals_cache` Supabase table. In the **fast loop**, a `NullFundamentalsAdapter` delegate skips API calls and reads from cache only. ETF symbols (configured via `SYMBOLS_ETF`) always skip fetching — they have no meaningful per-share EPS/P/E. The domain evaluates a deterministic `ValuationVerdict` (`UNDERVALUED / FAIR / OVERVALUED / UNKNOWN`) based primarily on PEG ratio, with PE/growth qualifiers; ETFs always get `UNKNOWN`. The verdict is surfaced to the Council prompt as a `Valuation snapshot` block and rendered as a dedicated **Wycena fundamentalna** section in the email report.
4. **Volatility gate** — `Asset.evaluate_volatility(delta, threshold)` lives in the pure domain (Hexagonal core). Δ < 2% → `ignore`, no paid APIs touched. **Domain decides, graph executes.**
5. **News + Sentiment** — `AlphaVantageClient` makes one request per ticker, rotating N API keys when one exhausts its 25 req/day quota. A `relevance ≥ 0.5` filter strips noise **before** anything reaches the LLM. Per ticker it returns a multi-feature dict: `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `av_sentiment_label`.
6. **LLM (cross-validation)** — **Claude Sonnet 4.6** (default; or OpenAI GPT when `LLM_PROVIDER=openai`) receives pre-computed AV sentiment + headlines + reflection context + fundamentals valuation snapshot. It returns structured JSON: `trend_direction`, `confidence_score`, **`av_agreement`** (whether it agrees with AV — anything below 0.3 flags potential manipulation), `target_price_12h`, `reasoning`.
7. **ML hard prediction (local XGBoost)** — model lives in a `.ubj` file inside the repo (Local-First AI). Consumes 7 features: `price_delta`, `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `llm_trend_signal`, `av_llm_agreement`. **The model predicts a 12h _return_, not an absolute price** — `predict_node` reconstructs the target price as `current_price × (1 + return)`. This makes RMSE measure the accuracy of the *move* (not the price level, which used to dominate and let a "echo the last price" model win), and the persistence baseline becomes a clean "return = 0". The `price_delta` feature uses the **same reference in training and inference** — the price of the *previous logged prediction* (view's `LAG(price_at_prediction)` ↔ `RepositoryPort.get_last_prediction_price`), not the per-cycle snapshot that drives the volatility gate (avoids train/serve skew). On cold start (no trained weights yet) it falls back to a "no change" (= return 0) baseline instead of crashing — consistent with the new target.
8. **Advisory Council** — after `predict_node`, 7 legendary investor personas (Buffett, Graham, Lynch, Dalio, Soros, Wood, Marks) each independently analyse the same data via parallel LLM calls (one worker per investor). The personas are **data, not code** — one JSON file per investor in [`data/council_personas/`](data/council_personas/), schema `{"name": str, "style": str}`. Adding / removing a council member = adding / removing a file (bounded by a `COUNCIL_MAX_PERSONAS = 12` cap — extra persona files beyond the cap are dropped with a WARNING, so the per-cycle LLM cost can't silently scale without limit). Loader (`src/infrastructure/persona_loader.py`) validates schema and uniqueness at startup; CLI walidator `uv run python -m src.tools.validate_personas` plus a pre-commit hook catch typos before runtime. A final "chairman" call writes the human-readable `summary` + `dissenting_views`, but the decision scalars — `final_recommendation` and `consensus_strength` — are computed **deterministically in the domain from the actual investor votes** (confidence-weighted), not taken from the chairman's self-reported number; if the chairman call fails, the verdict still reflects the real votes instead of silently defaulting to HOLD. Two volatility gates: the main one (`volatility_threshold`, default 2%) decides whether to run the prediction pipeline at all; the **council-specific gate** (`council_volatility_threshold`, default 3%) further filters out medium-Δ cycles where the 8 LLM calls would mostly return HOLD — set it to `0.0` to disable. Stored as JSONB in `prediction_logs.council_verdict` (legacy blob) **and** as one row per investor in `council_votes` (structured audit trail — query "how did Soros vote on NVDA in the last month" without parsing JSON). Rendered as a styled table in the email report.
9. **Persist + RAG** — `SupabaseRepository` writes the full record to `prediction_logs`. The news summary is embedded (OpenAI `text-embedding-3-small` → 1536-dim `pgvector`) **once per cycle in `predict_node`** and reused at save time. That embedding is **actually consumed**: before the LLM call, `predict_node` runs a pgvector similarity search (`match_news_embeddings` RPC, migration `011`) to pull the most similar *past* situations and their real outcomes (trend, hit/miss, the correction insight) into the prompt as `<similar_past_situations>`. RAG is fully graceful — if the RPC/pgvector is unavailable, retrieval is skipped and the prediction still runs.
10. **Slow Loop (weekly cycle)** — `main_trainer.py` retrains XGBoost on resolved predictions (those with a realized `target_return` in the `ml_feature_store` view), commits the new weights back to the repo (Continual Learning). The view owns the single source of truth for `price_delta` (NaN-guarded) and the `target_return` label — the trainer no longer recomputes them. Validation is **walk-forward** (expanding-window folds, not one noisy 20% split): the ship/skip gate decides on the *average* across folds, and the result reports `candidate_holdout_rmse` (vs the zero-return persistence baseline), `candidate_holdout_directional_hit_rate`, `n_folds`, and per-feature distribution stats for drift visibility — keys are explicitly *candidate* metrics because the shipped model is refit on all data and not re-validated. Also runs a fundamentals refresh step using `AlphaVantageFundamentalsAdapter` to repopulate `fundamentals_cache`.
11. **Deliver** — Polish-language HTML report via Resend with 2 charts (Δ per cycle + forecast), correlation scatter plot, trade signals sorted by `confidence × |Δ|`, risk signals with severity badges, day-over-day diff, and clickable news headlines. Two sections (council + fundamentals valuation) render via Jinja2 templates in `src/application/templates/` — autoescape on, the rest of the report still uses f-string composition in `report_builder.py` (incremental migration). The council template surfaces domain-level signals via `CouncilVerdict.is_split_decision()` (⚠️ PODZIELONA RADA badge), `has_strong_consensus()` (SILNY KONSENSUS badge), and `vote_distribution()` (BUY/SELL/HOLD count).

All HTTP adapters retry transient failures (429 / 5xx) with exponential backoff, and a single per-symbol error never aborts the whole cycle.

## Sources

| Source | Filter / parameters | Notes |
|---|---|---|
| **Finnhub** (`/quote`) | One request per ticker, 60 req/min free tier | US exchanges only. `.L` / `.DE` / `.NL` tickers → 403. |
| **Alpha Vantage** `NEWS_SENTIMENT` | One request per ticker, `limit=50`, client-side relevance filter ≥ 0.5 | **AND filter on tickers** = a separate request per symbol. Rate limit 25 req/day × N keys = N × 25/day. |
| **Alpha Vantage** `OVERVIEW` + `EARNINGS` | 2 requests per stock symbol per slow-loop refresh | Used by `AlphaVantageFundamentalsAdapter`. Free tier = 25 req/day → max ~12 stock symbols per refresh with a single key. Multi-key rotation via `ALPHA_VANTAGE_API_KEYS` (CSV) already supported. ETF symbols (`SYMBOLS_ETF`) are skipped. |
| **Supabase** (Postgres + pgvector) | `prediction_logs` + `price_snapshots` + `fundamentals_cache` tables + materialized view `ml_feature_store` | Service role key. RPC `refresh_ml_feature_store` is called before training. |
| **OpenAI** `chat.completions` | JSON mode (`response_format={"type":"json_object"}`), temperature 0.2 | Advisory council (`gpt-5-mini`) + fallback main LLM. Main analysis defaults to Claude; set `LLM_PROVIDER=openai` to run everything on OpenAI. |
| **OpenAI** `embeddings` | `text-embedding-3-small` → 1536-dim vector | Embeds the news summary into `pgvector`. Wired regardless of `LLM_PROVIDER` since `OPENAI_API_KEY` is always required — works with the Anthropic LLM too. |
| **Anthropic** (default main LLM) | Messages API, `claude-sonnet-4-6`, max_tokens 4096 | Adapter strips ```` ```json ... ``` ```` wrappers (Claude has no JSON mode). |
| **QuickChart.io** | URL-based, GET request | Zero setup, no dependency, works in every mail client. |
| **Resend.com** | Sandbox sender `onboarding@resend.dev` | Free tier 100 mails/day. No domain verification required — just a verified recipient. |
| **NBP** (`/exchangerates/rates/A`) | 30-day window for EUR & USD vs PLN | Free, no API key, no rate limit. Used by `NbpClient` (implements `MacroIndicatorsPort`) to compute 30-day FX-stress for the Risk Watch section. |
| **CoinGecko** (`/simple/price`) | One request per cycle for the configured crypto basket | Free, no API key, no meaningful rate limit. Used by `CoinGeckoAdapter` (implements `MarketDataPort`) for crypto prices. Maps clean tickers (`BTC`, `ETH`) to CoinGecko coin ids (`bitcoin`, `ethereum`). |

## Symbols (default portfolio of 43)

```text
AAPL,AMZN,GOOGL,MSFT,META,NVDA,TSLA,AMD,NET,PLTR,
ORCL,UBER,TSM,ASML,ASMIY,SAP,SIEGY,NVO,
DELL,IBM,MU,QCOM,CRWD,INTC,SNDK,BLK,SSNLF,
TEAM,FROG,SNOW,DDOG,SAIL,OKTA,S,PANW,
VT,QUAL,IHI,VB,EWY,IVV,XDWD.DE,IUSN.DE
```

Sector mix: 🤖 AI / semis (NVDA, AMD, TSM, ASML, ASMIY, MU, QCOM, INTC, SNDK) · ☁️ cloud / software (MSFT, ORCL, NET, SAP, PLTR, IBM, TEAM, FROG, SNOW, DDOG) · 🔐 cybersecurity (CRWD, PANW, OKTA, S, SAIL) · 📱 big tech (AAPL, AMZN, GOOGL, META) · 🖥️ hardware (DELL, SSNLF) · 🚗 mobility (TSLA, UBER) · 🏭 industrial / pharma (SIEGY, NVO) · 💰 financials (BLK) · 📊 ETFs (VT, QUAL, IHI, VB, EWY, IVV, XDWD.DE, IUSN.DE).

Configurable via `SYMBOLS` in `.env` (CSV).

## Crypto basket (separate price source, separate threshold)

Crypto tickers live in their **own pool** — `CRYPTO_SYMBOLS=BTC,ETH` — and are routed to a different price adapter (`CoinGeckoAdapter`, not Finnhub). The graph itself doesn't know — `RoutingMarketDataPort` picks the right adapter per symbol behind the `MarketDataPort` interface.

Why a separate pool:

- **Different volatility profile** — BTC natively moves 3–5% per day. The equity threshold (2%) would make every crypto cycle pass the volatility gate, burning 8 LLM calls (7-persona council + chairman) on routine noise. `CRYPTO_VOLATILITY_THRESHOLD=0.05` (5%) keeps the gate meaningful: full pipeline only runs when the move is actually a signal.
- **Different ticker format on the news side** — Alpha Vantage `NEWS_SENTIMENT` requires the `CRYPTO:` prefix (`CRYPTO:BTC`, `CRYPTO:ETH`). `AlphaVantageClient` translates `BTC ↔ CRYPTO:BTC` internally so the rest of the system stays clean.
- **No fundamentals** — same model as ETFs: `AssetType.CRYPTO` ⇒ `evaluate_valuation` short-circuits to `ValuationVerdict.UNKNOWN`, no AlphaVantage `OVERVIEW` / `EARNINGS` requests wasted.

Everything else (predict, news, sentiment, advisory council, prediction logs, self-reflection) works identically to equities — crypto goes through the same `AnalyzeMarketUseCase`.

## Quota monitoring (no silent exhaustion)

Every adapter that has a paid / metered quota writes a `QuotaAlert` to a shared `QuotaMonitor` when something looks bad:

| Adapter | What it emits |
|---|---|
| `AlphaVantageClient` | `CRITICAL` when all `ALPHA_VANTAGE_API_KEYS` have hit the daily 25 req/day cap — feed is partial. |
| `OpenAIAdapter` | `WARNING` when 429 was retried but eventually succeeded (you're at the TPM edge). `CRITICAL` when retries are exhausted and the call failed. Source field includes the model: `OpenAI (gpt-5-mini)`. |
| `AnthropicAdapter` (default main LLM) | Same retry/backoff as OpenAI on 429 / 5xx / 529 (Overloaded) / connection errors. `WARNING` after a successful retry, `CRITICAL` when retries are exhausted. Source field includes the model: `Anthropic (claude-sonnet-4-6)`. |
| `FinnhubAdapter` | `CRITICAL` on 429 (free tier 60 req/min hit). |
| `ResendNotifier` | `CRITICAL` on 429 (free 100 mails/day) or any other 4xx — email was not delivered. The alert appears in the **next** report once delivery recovers. |

All alerts from the current cycle are persisted to the `quota_alerts` table (migration `010_quota_alerts.sql`). The report builder pulls **the last 24 h** of alerts and renders them as a coloured banner at the **very top** of the e-mail (above the NYSE session line). Severity drives the banner colour and ordering — `CRITICAL` rows come first.

If anything `CRITICAL` is in the current cycle, the e-mail subject is prefixed with `⚠️ [QUOTA] ` so the alert is visible directly in the inbox, before opening.

This means: **no exhausted limit, blocked email, or rate-limited API call can silently disappear into the logs**. If the agent saw it, you see it.

## Risk Watch (separate macro pass)

Beyond the per-symbol prediction loop, the agent runs a **separate Risk Watch use case** (`MonitorMacroRiskUseCase`) over proxy instruments that signal *risk-off* conditions. These tickers are tracked but **never enter the prediction pipeline** — their semantics are inverted (a *rise* in `SH` means S&P 500 is *falling*, which is a warning).

```text
EPOL                                # iShares MSCI Poland — sovereign proxy, decline = capital flight from PL
SH,PSQ,RWM,EUM,SQQQ                 # inverse equity (SPY, QQQ, Dow, EM, 3× QQQ)
TBT                                 # inverse long-Treasuries — proxy for rising yields
GLD                                 # gold — classic safe haven
VIXY,UVXY                           # short-term VIX futures — explicit volatility hedge
```

Each instrument is tagged with a `MacroRiskInstrumentType` via `RISK_SYMBOL_TYPES`. The domain (`MacroRiskSignal.evaluate_alert`) maps price change to one of `NORMAL / ELEVATED / CRITICAL` — for `SOVEREIGN_PROXY` the sign is inverted so a *falling* EPOL triggers the alert.

When `NBP_ENABLED=true`, the `NbpClient` adapter (implements `MacroIndicatorsPort`) pulls a 30-day EUR/PLN and USD/PLN window from `api.nbp.pl` (free, no key). `PolishMacroSnapshot.evaluate_stress_level` raises ELEVATED / CRITICAL when the złoty weakens > 2% / > 5% over the window — early warning for PL fiscal stress visible in FX before it shows up in the rating.

Output lands in the e-mail as a dedicated **🚨 Risk Watch** section (signals table + FX block + overall alert level). A per-symbol fetch failure (e.g. Finnhub 403 on a delisted ticker) is logged and skipped — Risk Watch never breaks the main report.

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
- 📊 **Closed predictions (24h)** — full **post-mortem** per resolved prediction (stocks **and** crypto): the original thesis (*why up/down* — `reasoning_text`), the actual move (`$A → $B (+C%)`), the ✅ Trafiona / ❌ Błędna verdict, and **why** (the `reflect` diagnosis for misses, thesis-confirmed for hits). All from data already in `prediction_logs` — no extra LLM calls.
- 🎯 **Accuracy history** — directional hit-rate over the last 30 days
- 🪙 **Krypto** — dedicated section listing every tracked crypto ticker (price · Δ · trend/forecast for `saved`, "poniżej progu" for `ignored`). Rendered from the domain asset class (`Asset.asset_type`, surfaced as `SymbolResult.asset_class`), **independent of the 5% volatility gate** — so crypto stays visible even on a quiet day instead of vanishing into the "Pominięte" chips.
- 📈 **Δ chart** (QuickChart bar chart, change since last cycle)
- 🧠 **Self-Reflection** — lessons learned per symbol (purple box)
- 🔮 **Predictions table** — price · Δ (per cycle) · trend · forecast (next cycle) `+X.YZ%` · confidence · sentiment · news count. Each row is tagged with its **sector** next to the company name (stocks → Polish sector e.g. `Cyberbezpieczeństwo`, ETFs → `ETF`, crypto → `Krypto`); mapping in `report_formatting.SECTORS`.
- 💡 **Worth a look (sectors in motion)** — peer suggestions computed **from the current cycle only, zero extra API calls**: when a stock sector runs hot (its strongest monitored move `|Δ| ≥ 3%` or mean `|sentiment| ≥ 0.3`), the report suggests notable peers in that sector you don't yet monitor (curated `report_formatting.PEERS`), e.g. *Cyberbezpieczeństwo gorące (CRWD +6.0%, PANW +4.2%) — rozważ: ZS, FTNT, CYBR*. Top 3 sectors, 3 peers each; section hidden when nothing is hot.
- 📈 **Forecast chart** (QuickChart bar chart)
- 💡 **Reasoning** + 📰 **Top news** (clickable `<a href>` links to the original articles)
- 📈 **Sentiment vs price correlation** (scatter plot)
- ⏸ **Ignored** (chip list) + ⚠️ **Errors**

All sections are **conditional** — they only render when data exists. The first cold-start cycle produces a concise report; once you have 50+ cycles behind you, accuracy sparklines and history kick in.

## Tech stack

- **Python 3.12** with `from __future__ import annotations`
- **[uv](https://github.com/astral-sh/uv)** (Astral) — dependency manager, `uv sync --frozen` in CI
- **[LangGraph](https://langchain-ai.github.io/langgraph/) 1.3** — decision graph with `StateGraph[AgentState]`
- **OpenAI Python SDK** v2 — `gpt-5-mini` advisory council + `text-embedding-3-small` embeddings, JSON mode
- **Anthropic Python SDK** (extra `anthropic`) — Claude Sonnet 4.6, **default model for the main analysis**
- **XGBoost 3.2** + scikit-learn — sklearn-style API, native `.ubj` (UBJSON) format
- **Supabase** (Postgres 16 + pgvector) — REST via `supabase-py`, service_role key
- **Resend.com** — HTML email, sandbox sender without domain verification
- **QuickChart.io** — URL-based charts (`<img src>` in HTML), zero dependency
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

# 3. Database schema
# Open Supabase → SQL Editor → paste & run, in order:
#   migrations/001_init.sql        (prediction_logs + ml_feature_store)
#   migrations/002_price_snapshots.sql  (price_snapshots — breaks cold-start)
#   migrations/003_add_embedding.sql    (embedding VECTOR(1536) + pgvector index)
#   migrations/004_align_ml_feature_store.sql  (7-feature XGBoost contract)
#   migrations/005_council_verdict.sql          (council verdicts schema)
#   migrations/006_fundamentals_cache.sql       (fundamentals_cache table with TTL)
#   migrations/007_council_votes.sql            (per-investor structured audit trail)
#   migrations/008_data_quality_flags.sql       (data_quality_flags on prediction_logs)
#   migrations/009_trend_correctness.sql        (is_trend_correct on prediction_logs + backfill)
#   migrations/010_quota_alerts.sql             (quota_alerts audit trail for the report banner)
#   migrations/011_match_news_embeddings.sql    (pgvector RPC for RAG retrieval over news embeddings)
#   migrations/012_ml_feature_store_return_target.sql  (target = 12h RETURN, not absolute price)
#   migrations/013_idempotency_and_pagination.sql      (timestamp_hour + UNIQUE(symbol,hour) → upsert idempotency)

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

Config is split in two: **secrets** live in `.env` (gitignored) / GitHub Secrets; **everything non-secret** (symbols, thresholds, models, providers, throttle, Risk Watch…) lives in committed [`config.toml`](config.toml) — the single source of truth, read directly by `Settings`. No duplication across `.env` / workflow / repo variables.

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

Everything else — `symbols`, `symbols_etf`, `volatility_threshold`, `crypto_symbols`, `council_llm_provider/model`, `risk_symbols`, `nbp_enabled`, `symbol_throttle_seconds`, `notifications_enabled`, `digest_from_email`, … — is edited in [`config.toml`](config.toml). Any value can still be overridden by an environment variable (precedence: **env → `.env` → `config.toml` → code defaults**), e.g. `SYMBOL_THROTTLE_SECONDS=0` for a quick local run.

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
uv run ruff check src tests        # lint
uv run mypy src                    # type check (strict)
```

## GitHub Actions

Three workflows in [`.github/workflows/`](.github/workflows/):

| File | Cron (UTC) | Polish time (CEST / CET) | What it does |
|---|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | — (on push / PR) | — | ruff + mypy + pytest, plus a parallel **SonarCloud** scan job (`pytest --cov` → coverage upload) |
| [`fast_loop_12h.yml`](.github/workflows/fast_loop_12h.yml) | _disabled_ (manual only) | — | Analysis + email report. **Daily schedule is currently paused** — the cron is commented out, so it runs only via manual `workflow_dispatch`. Re-enable by uncommenting the `schedule` block. |
| [`slow_loop_weekly.yml`](.github/workflows/slow_loop_weekly.yml) | `0 3 * * 0` | Sunday 05:00 (summer) | XGBoost retraining + commit new weights |

The loop workflows expose `workflow_dispatch` for manual triggers from the GitHub UI. Their cron is fixed-UTC and **does not follow DST** — when winter time kicks in, the schedule shifts by one hour relative to Polish time. GitHub Actions cron is best-effort — 5-60 min delays are normal.

**Repository secrets** (the only things CI needs beyond the committed `config.toml`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEYS`, `SUPABASE_URL`, `SUPABASE_KEY`, `RESEND_API_KEY`, `DIGEST_TO_EMAIL`. `ANTHROPIC_API_KEY` is required (main analysis runs on Claude); `DIGEST_TO_EMAIL` is a **secret** (was a variable — move it). `FRED_API_KEY` is **optional** — only needed when `yield_curve_enabled = true` (the FRED yield-curve alpha source). `SONAR_TOKEN` is **optional** — it enables the SonarCloud scan job; without it that job is skipped, so CI never breaks before Sonar is configured.

**SonarCloud — one-time setup.** [`ci.yml`](.github/workflows/ci.yml) carries a `sonarcloud` job that runs `pytest --cov` and uploads code + coverage to [SonarCloud](https://sonarcloud.io/project/overview?id=DominikSienkiewicz_StockAgent) (keys live in [`sonar-project.properties`](sonar-project.properties)). To enable it: (1) create a **`SONAR_TOKEN`** repository secret (SonarCloud → *My Account → Security*); (2) in SonarCloud, **disable *Automatic Analysis*** (*Administration → Analysis Method*) — CI-based and automatic analysis cannot both run, and only the CI scan uploads the coverage report the "Sonar way" gate (≥ 80% on new code) needs.

**Optional alpha-data sources** (category "Data"): seven extra signals — SEC EDGAR insider flow, AlphaVantage earnings calendar, Finnhub options/IV, Reddit social velocity, FRED yield-curve, Finnhub analyst consensus, and regime-tagged RAG (vector memory). Each is a port + adapter with a **Null fallback** and is **off by default** in `config.toml` (`insider_flow_enabled`, `earnings_calendar_enabled`, `options_flow_enabled`, `social_velocity_enabled`, `yield_curve_enabled`, `analyst_consensus_enabled`, `vector_memory_enabled`) — so they make **zero network calls** until you flip a flag (and supply its key where needed). Enabled per-symbol signals are aggregated into an "Alpha Signals" report section; `vector_memory_enabled` needs migration `017`.
**Repository variables:** none. All non-secret config lives in committed [`config.toml`](config.toml), read directly by `Settings` — so there is nothing to map in the workflow `env:` and no `vars.*` drift to guard. The workflow injects only the secrets above; the split (secrets mapped, non-secret config kept out) is guarded by [`tests/test_workflow_env_wiring.py`](tests/test_workflow_env_wiring.py).

### Managing the secrets (helper scripts)

Two `gh`-CLI helpers in [`scripts/`](scripts/) keep the repo's GitHub Actions secrets in sync with your local `.env`. Both derive the required list **live** from `.github/workflows/*.yml` (every `secrets.*` / `vars.*` reference, minus the auto-injected `GITHUB_TOKEN`), so they never drift from the workflows:

```bash
# Audit: which required secrets/variables are set on the repo, which are missing.
# Read-only — never changes anything. Exit 0 = all present, 1 = something missing.
scripts/gh-secrets-check.sh

# Push: set the secrets/variables on the repo from .env. Prints a masked plan and
# asks for confirmation; values go to gh over stdin (never as args). --dry-run to preview.
scripts/gh-secrets-sync.sh [--dry-run] [-y]
```

Both accept `-R owner/repo` (default: auto-detected from the git remote) and `-f <env-file>` (default: `<repo>/.env`). `.env` keys that no workflow references (e.g. `FRED_API_KEY`, `TELEGRAM_*`, `SLACK_WEBHOOK_URL`) are reported but never pushed. Auth uses your `gh auth login` token, or a `GH_TOKEN` from `.env` if present (a fine-grained PAT needs `Secrets: Read and write`).

## Configuration

All tunable parameters are a `Settings` Pydantic model in [`src/config.py`](src/config.py) with validators. Non-secret values are set in committed [`config.toml`](config.toml); secrets come from `.env` / GitHub Secrets. Precedence: **env var → `.env` → `config.toml` → the code defaults below** (so any field can be overridden ad-hoc via an env var). The `Default` column is the in-code fallback when a field is absent everywhere; `config.toml` ships the actual production values.

| Field | Default | Description |
|---|---|---|
| `llm_provider` | `openai` | `openai` or `anthropic` (production `config.toml` sets `anthropic`). |
| `council_llm_provider` | `None` | Override LLM provider for the advisory council only (heterogeneous strategy: cheap model for 7-persona council, frontier for main analysis). `None` → reuses `llm_provider`. |
| `council_llm_model` | `None` | Override model name for the council adapter (e.g. `gpt-5-mini`, `claude-haiku-4-5`). `None` → provider default. |
| `council_personas_dir` | `data/council_personas` | Directory with one JSON file per council member (`{"name": str, "style": str}`). Validate with `uv run python -m src.tools.validate_personas`. |
| `volatility_threshold` | `0.02` | Threshold that triggers full analysis (2%) |
| `council_volatility_threshold` | `0.03` | Extra threshold for the advisory council (8 LLM calls: 7 personas + chairman). Below this Δ the council is skipped even if the main gate passed. `0.0` disables. |
| `symbols` | `[AAPL, MSFT, NVDA]` | Monitored tickers (CSV in env — override with the 22-symbol portfolio) |
| `alpha_vantage_api_keys` | `[]` | CSV of keys for rotation on rate-limit |
| `symbols_etf` | `[]` | CSV of tickers classified as ETFs (e.g. `VT,QUAL,IHI,VB`). ETFs skip fundamentals fetching (no meaningful per-share EPS/P/E) and always receive `ValuationVerdict.UNKNOWN`. |
| `risk_symbols` | `[]` | CSV of Risk Watch tickers (inverse / safe-haven / VIX / sovereign-proxy). When empty, the Risk Watch use case is not built. |
| `risk_symbol_types` | `{}` | `SYM:TYPE,SYM:TYPE,...` mapping each `risk_symbols` entry to a `MacroRiskInstrumentType` (`INVERSE_EQUITY` / `INVERSE_TREASURY` / `SAFE_HAVEN` / `VOLATILITY` / `SOVEREIGN_PROXY`). |
| `nbp_enabled` | `false` | When `true`, wires `NbpClient` (`MacroIndicatorsPort`) into Risk Watch — adds the EUR/PLN, USD/PLN 30-day stress block to the e-mail. |
| `crypto_symbols` | `[]` | CSV of crypto tickers (clean form: `BTC,ETH`). Routed via `RoutingMarketDataPort` to `CoinGeckoAdapter`. News goes through Alpha Vantage with the `CRYPTO:` prefix added by `AlphaVantageClient`. Fundamentals always skipped (no per-coin EPS/P/E). |
| `crypto_volatility_threshold` | `0.05` | Separate volatility gate for `AssetType.CRYPTO` — equity threshold (2%) would treat BTC's natural 3-5% daily move as a signal every cycle. 5% keeps the LLM/council budget under control. |
| `symbols_unsupported_price` | `[]` | CSV of tickers the current price adapter cannot fetch (Finnhub free → 403 on EU-listed `.DE` / `.L`). Pre-filtered in `main_agent.main()` — they show up as **ignored**, not **errors**, so the report's error count reflects actual issues. |
| `symbol_throttle_seconds` | `0.0` | Sleep between symbols in the main loop. Set `> 0` to spread LLM calls across the OpenAI **TPM window** (council on OpenAI; tier 1 = 30k tokens / min) — at 30+ symbols with the 7-persona council, bursting hits 429s. Recommended: `2.0`. |
| `council_llm_provider` / `council_llm_model` | `None` | Route the advisory council to a cheaper / faster LLM. With 7 personas × N symbols the council dominates token use — pinning it to a cheap model (`gpt-5-mini`) keeps council cost low while the main analysis runs on Claude Sonnet 4.6. |
| `reflection_min_age_hours` | `6` | Minimum age (hours) a prediction must reach before `reflect_node` scores it. `reflect_node` resolves the **oldest** due (unverified, past-cutoff) prediction each cycle, draining the backlog rather than only ever touching the latest. `0` opts out of the age floor (backtest/legacy). The default `6` guards against an overlapping manual run prematurely closing a fresh prediction against a near-identical price (which would pollute `accuracy_score` and inflate the report hit-rate). |
| `ml_model_path` | `data/models/price_predictor.ubj` | XGBoost weights file |
| `notifications_enabled` | `false` | Enables email delivery |
| `digest_from_email` | `onboarding@resend.dev` | Resend sandbox sender |

Plus internal constants in `report_builder.py` (`DIVERGENCE_PRICE_THRESHOLD = 0.02`, `AV_LLM_CONFLICT_THRESHOLD = 0.3`, `LOW_SIGNAL_NEWS_THRESHOLD = 3`, `HIGH_RELEVANCE_BAR = 0.8`) and `xgboost_local.py` (hyperparameters `max_depth=4, eta=0.1, subsample=0.8`).

## Disclaimer

Educational / proof-of-concept project demonstrating expertise in Agentic AI, Hexagonal Architecture, and FinOps. **Generated predictions do not constitute investment advice.** Make financial decisions at your own risk, after consulting a licensed advisor.

## License

[MIT](LICENSE)
