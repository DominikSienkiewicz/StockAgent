# How it works

The pipeline, the data it runs on, and every signal it emits. Start at [the README](../README.md) for what this project is.

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
                 HTML report (inline charts) ──► Resend.com email
```

1. **Fetch price + snapshot** — `FinnhubAdapter` pulls the current quote for each US ticker. `check_price_node` saves a **price snapshot in every cycle** (`price_snapshots` table) so the next cycle always has a reference point — this breaks the cold-start deadlock. Free tier is US-only; dotted tickers (`CSPX.L`, `BAS.DE`) get a 403 and are handled gracefully.
2. **Self-Reflection (runs every cycle, before the volatility gate)** — `reflect_node` reads the last unverified prediction. It computes two independent domain metrics and persists both: `accuracy_score` (how close the price landed to the numeric target — feeds XGBoost training) and `is_trend_correct` (whether the directional call matched reality — feeds the report's hit-rate). If the prediction was directionally wrong, the LLM diagnoses why ("I ignored hawkish Fed signals"). The insight is injected into the next prediction's prompt as `<reflection_context>`. The **cheap bookkeeping** (`accuracy_score` / `is_trend_correct` / persisting the outcome) runs every cycle regardless of volatility, so every prediction is scored on the **next cycle**; the **paid `analyze_mistake` LLM diagnosis** is now gated by the same volatility decision as the prediction pipeline (on a flat, below-threshold cycle it records the outcome with a "diagnosis deferred" note instead of spending an LLM call — a FinOps bound on backward-looking spend). **Prediction horizon, honestly:** the loop runs at most once per trading day (not every 12h despite the legacy `*_12h` names; with the cron paused, only as often as you dispatch it manually), so a prediction is resolved ~24h later on weekdays and ~72h later across a weekend. A `reflection_min_age_hours` guard (config, default `0`; production `6`) skips predictions too fresh to score fairly — so an overlapping manual `workflow_dispatch` can't prematurely close a prediction made minutes earlier and pollute the accuracy signal.
3. **Fundamentals (slow loop refreshes, fast loop reads cache)** — `fetch_fundamentals_node` runs after `reflect` and before the volatility gate. The `FundamentalsPort` is implemented by `AlphaVantageFundamentalsAdapter` (2 API requests per stock symbol: `OVERVIEW` + `EARNINGS`) wrapped in `CachedFundamentalsAdapter` (decorator pattern). In the **slow loop**, real API calls populate the `fundamentals_cache` Supabase table. In the **fast loop**, a `NullFundamentalsAdapter` delegate skips API calls and reads from cache only. ETF symbols (configured via `SYMBOLS_ETF`) always skip fetching — they have no meaningful per-share EPS/P/E. The domain evaluates a deterministic `ValuationVerdict` (`UNDERVALUED / FAIR / OVERVALUED / UNKNOWN`) based primarily on PEG ratio, with PE/growth qualifiers; ETFs always get `UNKNOWN`. The verdict is surfaced to the Council prompt as a `Valuation snapshot` block and rendered as a dedicated **Wycena fundamentalna** section in the email report.
4. **Volatility gate** — `Asset.evaluate_volatility(delta, threshold)` lives in the pure domain (Hexagonal core). Δ < 2% → `ignore`, no paid APIs touched. **Domain decides, graph executes.**
5. **News + Sentiment** — `AlphaVantageClient` makes one request per ticker, rotating N API keys when one exhausts its 25 req/day quota. A `relevance ≥ 0.5` filter strips noise **before** anything reaches the LLM. Per ticker it returns a multi-feature dict: `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `av_sentiment_label`.
6. **LLM (cross-validation)** — **Claude Sonnet 4.6** (default; or OpenAI GPT when `LLM_PROVIDER=openai`) receives pre-computed AV sentiment + headlines + reflection context + fundamentals valuation snapshot. It returns structured JSON: `trend_direction`, `confidence_score`, **`av_agreement`** (whether it agrees with AV — anything below 0.3 flags potential manipulation), `target_price_12h`, `reasoning`.
7. **ML hard prediction (local XGBoost)** — model lives in a `.ubj` file inside the repo (Local-First AI). Consumes 7 features: `price_delta`, `av_sentiment_score`, `av_relevance_avg`, `news_volume_24h`, `high_relevance_count`, `llm_trend_signal`, `av_llm_agreement`. **The model predicts a 12h _return_, not an absolute price** — `predict_node` reconstructs the target price as `current_price × (1 + return)`. This makes RMSE measure the accuracy of the *move* (not the price level, which used to dominate and let a "echo the last price" model win), and the persistence baseline becomes a clean "return = 0". The `price_delta` feature uses the **same reference in training and inference** — the price of the *previous logged prediction* (view's `LAG(price_at_prediction)` ↔ `RepositoryPort.get_last_prediction_price`), not the per-cycle snapshot that drives the volatility gate (avoids train/serve skew). On cold start (no trained weights yet) it falls back to a "no change" (= return 0) baseline instead of crashing — consistent with the new target.
8. **Advisory Council** — after `predict_node`, 7 legendary investor personas (Buffett, Graham, Lynch, Dalio, Soros, Wood, Marks) each independently analyse the same data via parallel LLM calls (one worker per investor). The personas are **data, not code** — one JSON file per investor in [`data/council_personas/`](../data/council_personas/), schema `{"name": str, "style": str}`. Adding / removing a council member = adding / removing a file (bounded by a `COUNCIL_MAX_PERSONAS = 12` cap — extra persona files beyond the cap are dropped with a WARNING, so the per-cycle LLM cost can't silently scale without limit). Loader (`src/infrastructure/persona_loader.py`) validates schema and uniqueness at startup; CLI walidator `uv run python -m src.tools.validate_personas` plus a pre-commit hook catch typos before runtime. A final "chairman" call writes the human-readable `summary` + `dissenting_views`, but the decision scalars — `final_recommendation` and `consensus_strength` — are computed **deterministically in the domain from the actual investor votes** (confidence-weighted), not taken from the chairman's self-reported number; if the chairman call fails, the verdict still reflects the real votes instead of silently defaulting to HOLD. Two volatility gates: the main one (`volatility_threshold`, default 2%) decides whether to run the prediction pipeline at all; the **council-specific gate** (`council_volatility_threshold`, default 3%) further filters out medium-Δ cycles where the 8 LLM calls would mostly return HOLD — set it to `0.0` to disable. Stored as JSONB in `prediction_logs.council_verdict` (legacy blob) **and** as one row per investor in `council_votes` (structured audit trail — query "how did Soros vote on NVDA in the last month" without parsing JSON). Rendered as a styled table in the email report. That audit trail is also what makes the personas **accountable**: the `persona_accuracy_stats` RPC (migration 018) joins settled predictions back onto each investor's votes, and the report renders a **credibility leaderboard** — "Buffett — 68% (22 głosy)". The ranking rule lives in the domain (`rank_personas`, `src/domain/persona_track_record.py`): a `min_votes = 5` threshold drops personas whose sample is too small to mean anything, so a single lucky call can't top the board. Sample size is always printed next to the percentage. Zero paid calls — it is computed entirely from votes already in the database, and the section suppresses itself when no persona clears the threshold.
9. **Persist + RAG** — `SupabaseRepository` writes the full record to `prediction_logs`. The news summary is embedded (OpenAI `text-embedding-3-small` → 1536-dim `pgvector`) **once per cycle in `predict_node`** and reused at save time. That embedding is **actually consumed**: before the LLM call, `predict_node` runs a pgvector similarity search (`match_news_embeddings` RPC, migration `011`) to pull the most similar *past* situations and their real outcomes (trend, hit/miss, the correction insight) into the prompt as `<similar_past_situations>`. RAG is fully graceful — if the RPC/pgvector is unavailable, retrieval is skipped and the prediction still runs.
10. **Slow Loop (weekly cycle)** — `main_trainer.py` retrains XGBoost on resolved predictions (those with a realized `target_return` in the `ml_feature_store` view), commits the new weights back to the repo (Continual Learning). The view owns the single source of truth for `price_delta` (NaN-guarded) and the `target_return` label — the trainer no longer recomputes them. Validation is **walk-forward** (expanding-window folds, not one noisy 20% split): the ship/skip gate decides on the *average* across folds, and the result reports `candidate_holdout_rmse` (vs the zero-return persistence baseline), `candidate_holdout_directional_hit_rate`, `n_folds`, and per-feature distribution stats for drift visibility — keys are explicitly *candidate* metrics because the shipped model is refit on all data and not re-validated. Also runs a fundamentals refresh step using `AlphaVantageFundamentalsAdapter` to repopulate `fundamentals_cache`.
11. **Deliver** — Polish-language HTML report via Resend. **Every chart is inline HTML** — no `<img>`, no external host. Gmail and Outlook block remote images by default, which used to punch holes through the middle of the report, and the old `quickchart.io` URLs carried the portfolio's deltas and sentiment scores to a third party on every send. Bars are div-based (the `_render_sparkline_html` technique), the scatter is a quadrant table, and each bar keeps a minimum width plus its printed value so a -0.5% move can never look like a -8% one. The report opens with a **5-second lead**: the cycle's strongest signal ("NVDA -4.2%, rada PODZIELONA") rendered above the fold and reused as the email **subject line**, so the digest wins the inbox before it is opened. The lead's ranking lives in the domain (`src/domain/digest_lead.py`) and is deliberately *not* delegated to an LLM: a CRITICAL quota alert always outranks the biggest price move, and additive (not multiplicative) scoring means a symbol without a council verdict still scores. On a brand-new deployment the first email is instead a **"Dzien 1" onboarding section** — "building a baseline for N instruments, first predictions tomorrow" plus the portfolio's sector breakdown — with its own welcome subject, replacing what used to be a wall of "Pominiete". The distinction is load-bearing: `SkipReason` separates a genuine cold start (no previous price) from a symbol filtered by the volatility gate and from a ticker the price adapter cannot serve, so a mass source outage can never be dressed up as a friendly first run. Below it: 2 charts (Δ per cycle + forecast), correlation scatter plot, trade signals sorted by `confidence × |Δ|`, risk signals with severity badges, day-over-day diff, a council credibility leaderboard, and clickable news headlines. Two sections (council + fundamentals valuation) render via Jinja2 templates in `src/application/templates/` — autoescape on, the rest of the report still uses f-string composition in `report_builder.py` (incremental migration). The council template surfaces domain-level signals via `CouncilVerdict.is_split_decision()` (⚠️ PODZIELONA RADA badge), `has_strong_consensus()` (SILNY KONSENSUS badge), and `vote_distribution()` (BUY/SELL/HOLD count).

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

**Different sideways band** — `SIDEWAYS_TOLERANCE` is ±0.5%, a sensible "no move" band for a stock. Against BTC's 3–5% daily volatility it would call almost every day a move, so a `SIDEWAYS` prediction on crypto was scored wrong nearly every time. `CRYPTO_SIDEWAYS_TOLERANCE` (±2.5%) is used instead when the asset is crypto. This is a measurement correction, not a loosened grade.

**Crypto is retrained** — the Slow Loop iterates `symbols + crypto_symbols`, deduplicated. It previously trained only `symbols`, so BTC and ETH never went through a retrain and their model stayed at cold start forever.

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
