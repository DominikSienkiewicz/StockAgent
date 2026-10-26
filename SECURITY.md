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
6. **Rotation is manual.** Nothing rotates a key on its own; the
   [inventory](#secrets-inventory-and-rotation) and the 180-day warning of
   `scripts/gh-secrets-check.sh` make an overdue key visible, not rotated.

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

## Secrets inventory and rotation

Every secret a workflow reads and every key of [`.env.example`](.env.example) has a row here,
without its value; [`tests/test_secrets_inventory.py`](tests/test_secrets_inventory.py) fails
when one is missing, when a row names a secret nothing reads any more, or when this page
carries the shape of a key. What each workflow runs and when is in the
[workflow table](docs/running-locally.md#github-actions).

| Secret | Read by | Role and scope | Account | How to rotate |
|---|---|---|---|---|
| `OPENAI_API_KEY` | Fast Loop, Slow Loop | Embeddings, the advisory council and the fallback main analysis; paid | OpenAI platform, project key | Create a key, sync, delete the old key |
| `ANTHROPIC_API_KEY` | Fast Loop, Slow Loop | The main analysis; paid | Anthropic Console | Create a key, sync, delete the old key |
| `FINNHUB_API_KEY` | Fast Loop, Shock Watch | Stock quotes, analyst consensus, earnings calendar; personal-use plan | Finnhub dashboard | Regenerate the key in the dashboard, sync |
| `ALPHA_VANTAGE_API_KEYS` | Fast Loop, Slow Loop | News, sentiment, fundamentals and earnings; a comma-separated list, each key with its own daily quota, masked one by one in the job log | Alpha Vantage, one free key per address | Request a new key, replace it in the list, sync |
| `FRED_API_KEY` | Fast Loop | Yield curve and macro series; read-only, free | FRED account (Federal Reserve Bank of St. Louis) | Request a key in the account, sync, delete the old key |
| `SUPABASE_URL` | Fast Loop, Slow Loop, Shock Watch, Deadman, Public Scorecard, Telegram bot | Address of the project's API; not a credential on its own | Supabase project | Changes only with a new project |
| `SUPABASE_KEY` | Fast Loop, Slow Loop, Shock Watch, Deadman, Public Scorecard, Telegram bot | Secret key: reads and writes every table past Row Level Security | Supabase project, API keys | Create a new secret key, sync, then delete the old key in the dashboard |
| `SUPABASE_DB_URL` | DB Migrate (manual) | Connection string of the `postgres` role through the session pooler: changes the schema; the application never reads it | Supabase project, database password | Reset the database password, rebuild the connection string, sync |
| `RESEND_API_KEY` | every workflow except CI and the Integration smoke | Sends the report, the alerts and the failure mails | Resend | Create a key, sync, revoke the old key |
| `DIGEST_TO_EMAIL` | every workflow except CI and the Integration smoke | The recipient address; personal data, not a credential | — | Update when the address changes |
| `TELEGRAM_BOT_TOKEN` | Fast Loop, Slow Loop, Shock Watch, Telegram bot | Sends pushes and reads the owner's commands as the bot | Telegram, BotFather | Revoke the token in BotFather (a new one is issued at once), sync |
| `TELEGRAM_CHAT_ID` | Fast Loop, Slow Loop, Shock Watch, Telegram bot | The only chat the bot pushes to and answers; not a credential | Telegram | Update when the chat changes |
| `SLACK_WEBHOOK_URL` | Fast Loop, Slow Loop, Shock Watch | Posts to one Slack channel; the whole URL is the secret | Slack app with an incoming webhook | Regenerate the webhook, sync |
| `HEARTBEAT_URL` | Fast Loop, Slow Loop, Shock Watch | Pings the external heartbeat check; the check's id in the path is the secret | External heartbeat service | Create a new check, sync, delete the old check |
| `BACKUP_AGE_RECIPIENT` | Slow Loop | Public key that encrypts the weekly evidence backup; its private half never enters the repository or GitHub | `age` key pair held by the owner | Generate a new pair, sync the public key, keep the old private key to read older backups |
| `SONAR_TOKEN` | CI (`sonarcloud` job) | Uploads the analysis and coverage; optional, the job is skipped without it | SonarCloud, My Account, Security | Revoke the token, generate a new one, sync |
| `SMOKE_OPENAI_API_KEY`, `SMOKE_ANTHROPIC_API_KEY`, `SMOKE_FINNHUB_API_KEY`, `SMOKE_ALPHA_VANTAGE_API_KEY`, `SMOKE_FRED_API_KEY`, `SMOKE_RESEND_API_KEY`, `SMOKE_DIGEST_TO_EMAIL` | Integration smoke (manual and weekly shape check) | Separate keys for the live-API tests, so a smoke run spends no production quota; optional, a missing key skips its test | The same providers, separate keys | As the production key of the same provider |
| `GH_TOKEN` | no workflow: only the local `scripts/gh-secrets-*.sh` | Optional fine-grained token with read and write access to this repository's secrets; empty means the `gh auth login` session | GitHub, fine-grained personal access token | Set an expiry when creating it; delete it when done |

**Policy.** Rotate every key at least once every 180 days, and at once when it may have
leaked: printed in a log, an email, an alert or a screenshot. Keys have leaked this way
before (incident G23: keys from the query string reached the public job logs), and since
then every error text and log line is redacted. A key that changes in place at the provider
breaks the scheduled runs until the new value is synced, so rotate outside the schedule
windows.

**Procedure.**

1. Create the new value in the account named in the table.
2. Put it into your local `.env`.
3. Push it to the repository secrets with `scripts/gh-secrets-sync.sh` (it shows a masked
   plan first and passes values to `gh` over stdin).
4. Revoke the old value at the provider.
5. Run `scripts/gh-secrets-check.sh` to confirm every secret a workflow reads is set; after
   a leak, check the provider's usage for the exposed window as well.

The date of the last rotation is the secret's last update on GitHub, and
`scripts/gh-secrets-check.sh` lists every repository secret not updated for more than 180
days under **Rotation**; the warning does not change its exit code.
