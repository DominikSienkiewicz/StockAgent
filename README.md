# 📈 StockAgent
**Architected & Developed by [Dominik](https://www.linkedin.com/in/dominik-sienkiewicz/)**

Autonomous financial agent for a curated portfolio of US stocks, ETFs and crypto. It fuses
price action with curated financial sentiment, cross-checks an LLM against Alpha Vantage to
flag possible manipulation, scores every past prediction against the realised price, and
delivers a Polish-language digest email with inline charts, trade signals and its own
running hit-rate. The daily schedule is **currently paused** — it runs on manual
`workflow_dispatch`.

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

In the era of AI-driven information overload, a single price tick is a useless signal — what
matters is the **covariance of sentiment, news, and historical predictions**. This agent
treats the market as a system: it pulls clean numerical data, enriches it through a curated
financial filter, models hybridly (LLM for reasoning + XGBoost for quantitative inference),
and **cyclically confronts itself with reality** through Self-Reflection. It's not a scraper,
and it's not another "GPT predict stocks" — every prediction it makes is recorded, closed
against the realised price, and fed back into the next run.

## One pass, end to end

`price + snapshot` → `self-reflection on the last prediction` → `fundamentals` →
**volatility gate** → `sentiment + news` → `LLM cross-validation` → `local XGBoost` →
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
