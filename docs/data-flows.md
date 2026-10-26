# Data flows to LLM providers

What StockAgent sends to a language-model or embedding provider, which code builds it and
where each field comes from. Every flow below is market data, the agent's own earlier
predictions and the council's own earlier votes; none of them carries the owner's private
data.

**Positions, P/L and e-mail addresses never reach a prompt.** No prompt and no embedding
input contains the owner's positions (quantities, purchase prices, purchase dates), P/L,
portfolio weights or any e-mail address (the report recipient and sender, subscribers, the
contact address in the SEC `User-Agent`). Positions are read only by the email report and
the risk layer (`main_agent._build_user_portfolio`, `PortfolioRiskUseCase`); addresses are
read only by the delivery channels. [`tests/test_llm_private_data_guard.py`](../tests/test_llm_private_data_guard.py)
(BEZ-32) puts sentinel positions and addresses where production takes them from (`Settings`
and the `positions` adapter), builds the Fast Loop ports with the real dependency injection,
runs four cycles through the graph (prediction, reflect diagnosis, seven personas and the
chairman) and the Slow Loop recap and diagnosis judge, and fails if any recorded prompt
carries a sentinel or an e-mail address.

Prompts go only to the providers below. The project uses no external tracing (LangSmith,
Langfuse, OpenTelemetry): the owner declined it on 2026-10-24 for the same privacy reason.

## Providers

| Provider | Used for | Chosen by |
|---|---|---|
| OpenAI (chat completions) | Main analysis, council, Slow Loop jobs (default) | `llm_provider = "openai"`, `council_llm_provider`, `llm_fallback_provider` |
| Anthropic (messages) | Any of the above when configured; the frontier model of the routing cascade | `llm_provider = "anthropic"`, `council_llm_provider`, `router_frontier_provider` with `model_routing_enabled`, `llm_fallback_provider` |
| OpenAI (embeddings, `text-embedding-3-small`) | RAG memory, whichever provider analyses | always OpenAI |

[Configuration](configuration.md) describes the provider and model settings. Without
`PAID_CALLS_ENABLED=true` the run uses local stubs and sends no LLM or embedding request.

## Inventory

| Call | Builder | Fields sent | Source |
|---|---|---|---|
| Prediction (Fast Loop `predict`) | `get_prediction_prompt` in [`prompts.py`](../src/application/prompts.py) | Symbol, current price, move since the last cycle, Alpha Vantage sentiment score and label, news volume and high-relevance count, news headlines and summary; the lesson from the last settled prediction; RAG analogues (summary, trend, outcome of earlier predictions); regime label; stances of peer symbols this cycle; the Alpha Fusion summary; fresh 8-K items with `corporate_events_enabled` | Finnhub or CoinGecko price, Alpha Vantage `NEWS_SENTIMENT`, `prediction_logs`, the regime of the cycle, SEC EDGAR |
| Prediction with tools (`tool_use_enabled`) | Same prompt plus the tools of [`tools.py`](../src/application/tools.py) | `get_fundamentals`: trailing and forward P/E, PEG, EPS growth of the symbol; `get_macro`: EUR/PLN and USD/PLN with their 30-day change | Fundamentals cache (`fundamentals_cache`), NBP |
| Shadow prompt variant (EXP-26) | Same prompt with the variant's options | As the prediction | As the prediction |
| Mistake diagnosis (Fast Loop `reflect`, Slow Loop backfill WIA-53) | `get_mistake_diagnosis_prompt` in [`prompts.py`](../src/application/prompts.py); [`backfill_diagnoses.py`](../src/application/use_cases/backfill_diagnoses.py) | Predicted trend, saved headlines and reasoning, entry, target and settlement price, the decision receipt (analogues used), a vindicated dissenter (persona name and recommendation) | `prediction_logs` (with its decision receipt), `council_votes` |
| Batch diagnosis (EXP-17, Slow Loop) | `batch_diagnosis_prompt` in [`batch_diagnosis.py`](../src/application/use_cases/batch_diagnosis.py) | Per miss: symbol, trend, prices, saved headlines and reasoning | `prediction_logs` |
| Diagnosis judge (EXP-10, Slow Loop) | `diagnosis_judge_prompt` in [`judge_diagnoses.py`](../src/application/use_cases/judge_diagnoses.py) | Symbol, trend, entry, target and settlement price, saved headlines and reasoning, the diagnosis under review | `prediction_logs` |
| Weekly recap narrative (Slow Loop) | `_narrative_prompt` in [`weekly_recap.py`](../src/application/use_cases/weekly_recap.py) | Number of settled predictions, best and worst symbol with their verdicts, ECE change, their lessons and the reasoning of vindicated dissenters | `prediction_logs` (with the stored council verdict) |
| Council persona | `investor_prompt` in [`council_prompts.py`](../src/application/council_prompts.py) | Persona name and style; symbol, price, move, sentiment score, headlines, the main analysis' trend and confidence (left out in the EXP-02 blind arm), ML target and whether the model has weights, cached fundamentals and valuation verdict | `data/council_personas/`, the cycle's graph state |
| Council revision (debate mode) | `investor_revision_prompt` in [`council_prompts.py`](../src/application/council_prompts.py) | As the persona, plus the peers' first-round recommendation, confidence and reasoning | The first council round |
| Council chairman | `chairman_prompt` in [`council_prompts.py`](../src/application/council_prompts.py) | As the persona, plus every opinion (persona name, recommendation, confidence, reasoning) | The council round |
| Embedding | `EmbeddingPort.embed` from the `predict` node | The cycle's news summary (RAG query, stored with the prediction) | Alpha Vantage `NEWS_SENTIMENT` |

The orchestration of the council lives in
[`council_service.py`](../src/application/council_service.py). Text that comes from outside
(headlines, lessons, reasoning) is wrapped in data-only fences before it reaches a prompt
(BEZ-06, BEZ-14).

## Retention and training at the providers

Checked on 2026-10-26 against the providers' current terms for API customers:

- **OpenAI API.** Data sent to the API is not used to train OpenAI models unless the
  organisation opts in. Abuse-monitoring logs of chat completions and embeddings are kept for
  up to 30 days. Zero data retention needs OpenAI's prior approval.
  Source: [Data controls in the OpenAI platform](https://developers.openai.com/api/docs/guides/your-data).
- **Anthropic API.** Inputs and outputs of commercial products are not used for training by
  default. They are deleted from Anthropic's backend within 30 days; inputs and outputs
  flagged for a Usage Policy violation may be kept for up to 2 years. Sources:
  [Is my data used for model training?](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)
  and [How long do you store my organization's data?](https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data).

These are the default terms; nothing in the code asks for a zero-data-retention arrangement.
Re-check both pages before adding a provider or a new kind of field to a prompt.
