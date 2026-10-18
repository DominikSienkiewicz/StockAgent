# 📈 StockAgent
**Designed and maintained by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/), built with AI coding agents:** I set the architecture, write the specifications, define the quality gates enforced in CI and review every change.

**Educational proof of concept — not investment advice.** A Python and LangGraph agent that
analyses a fixed watchlist of US stocks, ETFs and crypto. It combines price action with
curated financial news sentiment, cross-checks the LLM's reading against Alpha Vantage data,
scores every past prediction against the realised price, and writes a Polish-language report
email with charts and its own running hit-rate. The daily report run is **paused** and starts
only on manual `workflow_dispatch`. The hourly price-shock check is paused with it, because
its only reference prices are the daily run's snapshots. Two jobs still run on schedule: the
weekly model retraining, and a free daily deadman that mails when a job that should run has
gone quiet. The retraining has not shipped weights yet. Its last run (2026-10-11) still
trained one model per symbol and skipped all 45: the best-covered symbol had 17 settled
predictions against the 50 required, 288 across all symbols. Training now fits one shared
model on the pooled predictions of every symbol, so the 50-sample minimum applies to that
pooled sample, and weights ship only if the shared candidate also passes the walk-forward
gate against a no-change baseline. Until a run ships them, the XGBoost leg runs on that
baseline and the report and the council prompt label it as cold-start.

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

Architecture: **Hexagonal (Ports & Adapters) + DDD**. Domain (pure Python, zero deps) →
Application (ports + use cases + LangGraph) → Infrastructure (adapters for API / DB / LLM /
ML / email). The dependency arrow flows one way.

## Disclaimer

Educational / proof-of-concept project demonstrating expertise in Agentic AI, Hexagonal
Architecture, and FinOps. **Generated predictions do not constitute investment advice.**
Make financial decisions at your own risk, after consulting a licensed advisor.

## License

[MIT](LICENSE)
