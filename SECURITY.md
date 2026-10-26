# Security

StockAgent is an educational, single-owner project: one scheduled agent, one database and one
recipient of the report email. This page says how to report a problem, what the system
defends against, which gaps are known and left open, and how its secrets are handled. Start
at [the README](README.md) for what the project is.

## Reporting a vulnerability

Report it privately through GitHub's private vulnerability reporting: the **Security** tab of
the repository, then **Report a vulnerability**. Do not open a public issue or pull request
for it, and do not test against the live workflows or the database. Describe what an attacker
controls, what they reach, and the file or workflow involved. This is a one-person project,
so the response is best effort.

Only the `main` branch is supported; there are no releases.

## Threat model

### What is worth protecting

- **The keys.** Paid API keys (LLM, market data, email) and the Supabase secret key live in
  GitHub repository secrets. The Supabase key can read and write every table, and the
  database URL of the migration workflow can change the schema.
- **The owner's private data.** The portfolio positions, the recipient addresses
  (`subscribers`, the digest recipient) and the email itself, which carries third-party data
  under personal-use licences.
- **The track record.** Predictions are written before the outcome and scored after it;
  their value is that nobody, the agent included, rewrites them afterwards.

### Where untrusted data enters

News headlines and summaries (Alpha Vantage) are written by third parties and are the main
injection path. Every provider payload (Finnhub, CoinGecko, SEC EDGAR, FRED, NBP, Reddit,
Deribit) is untrusted input to a parser, and the LLM's own answer is untrusted input to the
rest of the pipeline. The repository is public, so its GitHub Actions logs are public too.

### Untrusted news → LLM → email

- **Into the prompt.** [`fence_untrusted`](src/application/_prompt_safety.py) normalises
  Unicode (NFKC, format characters removed), strips control characters, backticks, chat-ML
  tokens, pseudo-roles (`system:`), HTML/XML tag openings and forged copies of its own
  markers in any letter case, cuts every item to a length limit, and wraps the result in a
  `DATA-ONLY` fence that tells the model the content is data, not instructions. News, the
  RAG analogs from earlier cycles and the inputs of the Slow Loop jobs all go through it.
  [`tests/application/test_prompt_taint.py`](tests/application/test_prompt_taint.py) injects
  a corpus of payloads into every untrusted field of every prompt builder and fails when its
  canary appears outside a fence.
- **Out of the model.** The answer is parsed as JSON against the prompt's schema: a trend
  outside the enum becomes `SIDEWAYS`, a number that is missing, not a number or out of range
  becomes a default, and each case leaves a data-quality flag on the record. An answer that
  cannot be used becomes a neutral analysis flagged `llm_analysis_failed`, which stays out of
  the track record and of training. Injected text can therefore steer a prediction, but it
  cannot run code, and the prompt holds no key to leak.
- **Into the email.** Every field from a provider, the LLM or an error message is
  HTML-escaped (`escape` and the autoescaping Jinja templates in
  [`src/application/report_formatting.py`](src/application/report_formatting.py) and
  [`src/application/report_templates.py`](src/application/report_templates.py)), and a link
  is rendered only when it is `http` or `https` with a host (`safe_http_url`).
  [`tests/application/test_report_builder.py`](tests/application/test_report_builder.py)
  feeds it LLM, news and error text and rejects other schemes.

### Tool use

The tool-use loop is off in `config.toml` (`tool_use_enabled = false`). When it is on, the
model can call two read-only tools from [`src/application/tools.py`](src/application/tools.py):
`get_fundamentals` (served from the cache, no new paid request) and `get_macro` (free NBP
rates). There are at most `tool_use_max_rounds` rounds (3 in `config.toml`) and then one
forced answer without tools. No tool writes, sends a message or fetches a URL the model
chooses, and a tool's error goes back to the model only after secret redaction.

### Public outputs

The GitHub Actions run summary carries aggregates only: no symbol, address, error text,
price or position. The public scorecard (built, not deployed) carries aggregates of the
agent's own predictions and a test asserts that no symbol, address or amount reaches it.
Persona names appear only in the owner's email; public outputs show an archetype.

### Database

Every table has Row Level Security on without policies, and the anonymous and authenticated
roles have no grants, so only the secret key reads or writes; the agent refuses to start with
a publishable key. New functions are executable by the service role only. Local runs are
read-only and make no paid calls unless switched on explicitly (`DB_WRITES_ENABLED`,
`PAID_CALLS_ENABLED`). The tables, their data classes and who reads and writes them are in
[data and access](docs/data-and-access.md).

### CI and supply chain

Every action is pinned to a commit SHA, every checkout drops its credentials
(`persist-credentials: false`), and every workflow starts from read-only permissions, with
write scopes granted to the single job that needs them. Dependencies are installed from the
lockfile, and the CI gate fails when `uv.lock` does not match `pyproject.toml`.
[Dependabot](.github/dependabot.yml) proposes updates of the actions and the Python
dependencies weekly.

## Known gaps

These are open on purpose or not fixed yet. Each is written down so that a change has to
face it.

1. **The fence is a mitigation, not a guarantee.** A model can still follow an instruction
   inside the fence. The schema bounds what that can change (a prediction in the owner's
   email), not whether it happens.
2. **Stored injection through memory.** The lesson from self-reflection and the advisory
   council's reasonings are written by our own LLM from third-party news, so an injection can
   ride along and come back in a later prompt. With `prompt_taint_fence_enabled` off, as in
   production while the running experiments last, both reach their prompts unfenced;
   `KNOWN_LEAKS_WITHOUT_FLAG` in
   [`tests/application/test_prompt_taint.py`](tests/application/test_prompt_taint.py) lists
   them. The RAG analogs are fenced, but a poisoned summary keeps returning until its row is
   removed; there is no quarantine.
3. **Look-alike letters pass the sanitiser.** NFKC folds full-width and compatibility forms,
   not letters of other scripts, so a pseudo-role spelled with Cyrillic look-alikes stays in
   the text (still inside the fence).
4. **One key opens the whole database.** The Supabase secret key in GitHub secrets can read
   and write every table; there is no per-job role and no short-lived credential. A
   compromised workflow run could therefore alter the track record, which the commit-reveal
   attestation only shows after the fact, and only once it is published.
5. **Redaction works on known shapes.** Logs and the email are redacted for the configured
   secret values and for recognisable key shapes; a secret shorter than 8 characters, or one
   that is not configured and has no known shape, would print in clear in a public log.

## Secrets model

- **Where they live.** Locally in `.env`, which is never committed; in CI as GitHub
  repository secrets. Non-sensitive configuration (symbols, thresholds, flags, models,
  budgets) lives in the committed `config.toml`, and `.env.example` carries the same keys as
  `.env` with placeholders only.
- **Who gets them.** A workflow maps each secret into the environment of the step that uses
  it. The free jobs receive placeholders instead of the paid keys they do not need, and only
  the manual DB Migrate workflow receives the database URL that can change the schema.
- **What keeps them out of the output.** Adapters with a key in the URL raise errors with a
  redacted message, every entry point's log handler redacts each record (traceback
  included), and the comma-separated Alpha Vantage keys are masked one by one in the job
  log. [Secret redaction](docs/running-locally.md#secret-redaction) describes each step.
