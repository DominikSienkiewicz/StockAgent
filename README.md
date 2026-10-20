# 📈 StockAgent
**Designed and maintained by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/), built with AI coding agents:** I set the architecture, write the specifications, define the quality gates enforced in CI and review every change.

**Educational proof of concept — not investment advice.** A Python and LangGraph agent that
analyses a fixed watchlist of US stocks, ETFs and crypto. It combines price action with
curated financial news sentiment, cross-checks the LLM's reading against Alpha Vantage data,
scores every past prediction against the realised price, and writes a Polish-language report
email with charts and its own running hit-rate.

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph 1.2](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=for-the-badge)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Sonnet_4.6-D97757?style=for-the-badge&logo=anthropic&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings_+_council-412991?style=for-the-badge&logo=openai&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6B35?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Postgres+pgvector-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Resend](https://img.shields.io/badge/Resend-Email-000000?style=for-the-badge)
![Architecture](https://img.shields.io/badge/Architecture-Hexagonal+DDD-orange?style=for-the-badge)
![Quality Gate](https://img.shields.io/sonar/quality_gate/DominikSienkiewicz_StockAgent?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud&label=Quality%20Gate)
![Coverage](https://img.shields.io/sonar/coverage/DominikSienkiewicz_StockAgent?server=https%3A%2F%2Fsonarcloud.io&style=for-the-badge&logo=sonarcloud)

## Status

**The daily report run is paused** since 2026-08-17, when commit `02fc397` commented out
its cron; it starts only on manual `workflow_dispatch`. The reason is decision 2: the loop's
cost and its silent failures. Every weekday run paid for LLM, council and Alpha Vantage
calls, and a run that failed or did not start raised no alarm. Waves A to R have since added
a ceiling on the paid calls of a cycle, an explicit weekly budget, a daily deadman, an
external heartbeat and explicit write and paid-call switches in the workflows. The hourly
price-shock check is paused too (`f886da0`), but it can resume on its own: it reads each
symbol's change for the day from the same free quote and uses the daily run's snapshots
only as a fallback. Two jobs still run on schedule: the weekly model retraining, and a free
daily deadman that mails when a job that should run has gone quiet. The retraining has not
shipped weights yet. Its last run (2026-10-11) still trained one model per symbol and
skipped all 45: the best-covered symbol had 17 settled predictions against the 50 required,
288 across all symbols. Training now fits one shared
model on the pooled predictions of every symbol, so the 50-sample minimum applies to that
pooled sample, and weights ship only if the shared candidate also passes the walk-forward
gate against the strongest of three baselines: no change, the mean drift and the LLM
direction alone. Until a run ships them, the XGBoost leg runs on the no-change baseline
and the report and the council prompt label it as cold-start.

**Resume.** The conditions and the owner's steps, in order, are in the
[resume checklist](docs/resume-checklist.md): the secret Supabase key, the pending
migrations and the flags they unlock, the heartbeat secret, the experiment windows, the
deadman's gap for the daily run, and uncommenting the crons last. Once the daily run's cron
is active, `tests/test_workflow_env_wiring.py` fails unless a deadman runs on a cron and the
cost ceiling, the weekly budget, the liveness gap and the write and paid-call switches are in
place. Every run, scheduled or manual, is listed on the
[GitHub Actions workflows page](https://github.com/DominikSienkiewicz/StockAgent/actions).

**Database risk: the Supabase Free plan.** The Supabase project is on the Free plan, which
pauses a project after 7 days without activity. While the daily run and the shock check are
paused, the only job that writes to the database is the weekly retraining, so one late
GitHub cron run can be enough for a pause; until the project is restored in the Supabase
dashboard, every job fails on connect. The daily deadman therefore starts each run with a
read-only query of one `price_snapshots` row through PostgREST, with no paid keys, and mails
when the database does not answer. It is a monitor, not a keepalive: it reports a pause that
has already happened, and it does not guarantee that Supabase will not pause the project.

## 🧠 The Vision: Signal over Noise

In the era of AI-driven information overload, a single price tick is a useless signal — what
matters is the **covariance of sentiment, news, and historical predictions**. This agent
treats the market as a system: it pulls clean numerical data, enriches it through a curated
financial filter, is built to model hybridly (LLM for reasoning + XGBoost for quantitative
inference, once XGBoost has earned its weights against a naive baseline),
and **cyclically confronts itself with reality** through Self-Reflection. It's not a scraper,
and it's not another "GPT predict stocks" — every prediction it makes is recorded, closed
against the realised price, and fed back into the next run.

## One pass, end to end

`price + snapshot` → `self-reflection on the last prediction` → `fundamentals` →
**volatility gate** → `sentiment + news` → `LLM cross-validation` → `local XGBoost (baseline until trained)` →
`7-persona advisory council` → `persist + RAG` → `HTML email`.

The gate is the point: below the threshold no paid API is touched at all. A cycle where
every symbol was filtered reports **zero paid calls**, and the report prints that bill.

## Documentation

| Document | What's in it |
|---|---|
| [How it works](docs/how-it-works.md) | The eleven pipeline steps, every data source, the symbol and crypto pools, quota monitoring, Risk Watch, and the risk / trade signals |
| [The report email](docs/report-email.md) | Section-by-section anatomy of what lands in the inbox |
| [Running it](docs/running-locally.md) | Stack, prerequisites, local setup, database migrations, build commands, GitHub Actions, secret management |
| [Configuration](docs/configuration.md) | Every `Settings` field with its default, plus the feature flags and what each one costs |
| [Data and access](docs/data-and-access.md) | Every table and view with its data class and who reads and writes it, the Supabase keys, retention, and the Row Level Security rollout |
| [Reference signal protocol](docs/reference-protocol-v1.md) | The frozen rule of the free weekly benchmark the paid LLM has to beat (`reference-v1`) |

Architecture: **Hexagonal (Ports & Adapters) + DDD**. Domain (pure Python, zero deps) →
Application (ports + use cases + LangGraph) → Infrastructure (adapters for API / DB / LLM /
ML / email). The dependency arrow flows one way.

## Disclaimer

Educational / proof-of-concept project demonstrating expertise in Agentic AI, Hexagonal
Architecture, and FinOps. **Generated predictions do not constitute investment advice.**
Make financial decisions at your own risk, after consulting a licensed advisor.

**Survivorship bias.** The watchlist was hand-picked (2026-06) with knowledge of how those
companies had already risen, so a long bias on it is not skill. The track record and the
public scorecard show the naive "always BULLISH" baseline on the same predictions for that
reason, and each prediction's recipe records a `universe_version` (a hash of the sorted
watchlist), so adding symbols starts a new segment instead of passing for better accuracy.
Their paper-PnL equity curve is net of trading costs charged on turnover (10 bps, crypto
25 bps) and sits next to two buy&hold benchmarks with the same costs: a basket of the same
symbols in the same windows and IVV, together with a market-relative hit rate (did the
direction get the sign of the return over IVV right).

## License

[MIT](LICENSE)
