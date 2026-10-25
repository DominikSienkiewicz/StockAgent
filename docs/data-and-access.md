# Data and access

What the Supabase database holds, which job reads and writes each relation, which key the
app needs, how long data is kept, and how Row Level Security is rolled out.

> **Status.** Row Level Security arrives with migration
> [`038_enable_rls.sql`](../supabase/migrations/038_enable_rls.sql). Production gets it only
> when the owner runs the manual **🗄️ DB Migrate** workflow. Until that run, every relation
> below is still readable and writable with the project's public key, and the "RLS" column
> describes the state after `038`.

## Relations

All relations live in the `public` schema, which Supabase exposes through PostgREST. The
writer and reader columns name entry points: **Fast Loop** = `main_agent.py`, **Slow Loop** =
`main_trainer.py`, **Shock Watch** = `main_watch.py`, **Deadman** = `main_deadman.py`,
**tools** = `main_tools.py`, **backtest** = `main_backtest.py`, **Scorecard** = `main_scorecard.py`, **Bot** = `main_bot.py`. A local run writes nothing
unless `DB_WRITES_ENABLED=true` ([running-locally](running-locally.md#local-runs-are-read-only-by-default)).

| Relation | Migrations | Data class | RLS after `038` | Written by | Read by |
|---|---|---|---|---|---|
| `prediction_logs` | `001`, columns added up to `043`, `055` (`stale_quote`), `059` (session-close settlement), `072` (`entry_price`), `074` (`invalidation_price`, `invalidation_alerted_at`) | Model output: predictions, council verdicts, receipts, settlements, embeddings | on, no policies | Fast Loop (prediction, settlement, calibration, attestation reveal), Shock Watch (thesis-invalidation alert stamp, ML-27) | Fast Loop (reflection, RAG, report, track record), tools (`evaluate`), Scorecard (public scorecard) |
| `price_snapshots` | `002`, `013` | Market data: one price per symbol and hour | on, no policies | Fast Loop (every cycle) | Fast Loop, Shock Watch (fallback reference price when the free quote has no fresh daily change, ALR-02), Deadman (database probe), Scorecard (IVV benchmark) |
| `ml_feature_store` | materialized view, `001` rebuilt in `004`, `012`, `022`, `026`, `027`, `039`, `047`, `050`, `058` | Derived training features from `prediction_logs`; from `058` without settlements on a stale quote (`stale_quote IS NOT TRUE`, DAT-14) | not applicable (a materialized view has no RLS; grants only) | Slow Loop and backtest refresh it (`refresh_ml_feature_store`) | Slow Loop, backtest |
| `fundamentals_cache` | `006`, `056` (`market_cap`) | Market data: Alpha Vantage fundamentals cache | on, no policies | Fast Loop | Fast Loop |
| `council_votes` | `007`, `054` (unique vote per persona and prediction) | Model output: one vote per persona and prediction | on, no policies | Fast Loop | Fast Loop (persona weights and track record) |
| `quota_alerts` | `010` | Operational: API quota exhaustion events | on, no policies | Fast Loop, Slow Loop | Fast Loop (report banner) |
| `subscribers` | `014` | **Personal data**: recipient email addresses and watchlists | on, no policies | nobody in the app (edited by hand in Supabase) | Fast Loop (`subscriptions_enabled`) |
| `model_scorecards` | `019`, `028`, `049`, `052` | Model output: walk-forward evaluation of each training run and its feature distribution | on, no policies | Slow Loop | Fast Loop (report), Deadman (last training run) |
| `shock_alerts` | `021` | Operational: one sent price-shock alert per symbol and day | on, no policies | Shock Watch | Shock Watch (debounce), Fast Loop (the email's `⚡ Między mailami` section, ALR-06, with `shock_alerts_enabled`) |
| `positions` | `023`, `035` | **Owner's financial data**: holdings, quantities, average cost | on, no policies | tools (`import-positions --apply`) | Fast Loop (`portfolio_positions_enabled`), tools |
| `feature_snapshots` | `031` | Derived features: the vector each prediction was made from | on, no policies | Fast Loop (`feature_snapshots_enabled`) | nobody yet |
| `regime_history` | `033` | Model output: market regime per cycle | on, no policies | Fast Loop | Fast Loop (regime hysteresis) |
| `fx_rates` | `036` | Market data: NBP exchange-rate cache | on, no policies | Fast Loop | Fast Loop |
| `llm_calls` | `040` | Operational: one row per LLM or embedding request (tokens, cost, run, node, symbol), no prompt text | on, no policies | Fast Loop, Slow Loop (`cost_ledger_enabled`) | nobody in the app yet |
| `cycle_runs` | `041` | Operational: one row per loop run (run id and attempt, start, duration, symbol counts, paid calls per source, email delivered), no symbols | on, no policies | Fast Loop (`cycle_runs_enabled`) | Fast Loop (start of the last run whose email went out, for the `⚡ Między mailami` window, ALR-06); public status page (`public_status_page_enabled`: Fast Loop and `main_scorecard.py`, aggregates only, NIEZ-33) |
| `reference_predictions` | `042` | Model output: the free reference signal (WIA-24) — one row per weekly read with its call and settlement, `recipe_version = 'reference-v1'`, kept out of `prediction_logs` | on, no policies | Slow Loop (`reference_signal_enabled`) | Slow Loop (baseline read and open calls) |
| `earnings_cache` | `053` | Market data: the next earnings report date per stock symbol from Alpha Vantage (FIN-11) | on, no policies | Slow Loop (`earnings_cache_enabled`) | Fast Loop (earnings gate) |
| `price_level_alerts` | `057` | User settings: price levels to alert on (symbol, level, direction, note) with the watch's last reading and the hit time (ALR-11) | on, no policies | Shock Watch (`price_level_alerts_enabled`: adds the `price_alerts` list, records readings, disables a hit) | Shock Watch |
| `digests` | `061` | Model output: index of the private digest archive (MAIL-24) — one row per cycle with the subject, the lead and the top signals (tickers and labels, no amounts); the full HTML sits in the private Storage bucket `digests` | on, no policies | Fast Loop (`digest_archive_enabled`) | nobody in the app yet |
| `corporate_actions` | `062` | Market data: suspected stock splits — the quote's previous close against the last price snapshot gave a split ratio (DAT-23), one row per symbol, kind and UTC day | on, no policies | Fast Loop (`split_detection_enabled`) | nobody in the app yet |
| `shadow_predictions` | `075`, `083` | Model output: the champion's and the challengers' forecasts of the ML leg per saved prediction (model name, weights sha256, return or probability, ML-33); since `083` also the trend and confidence of a prompt or model variant run in shadow (EXP-26) | on, no policies | Fast Loop (`shadow_predictions_enabled`, `shadow_prompt_variant_enabled`) | Slow Loop (promotion verdict in the step summary; the ML read skips the variant's rows) |
| `paper_trades` | `073` | Model output: the append-only paper ledger (WIA-59) — one paper position per saved prediction (side, entry price and time, cost), closed once at settlement; `row_hash = sha256(canonical entry + prev_hash)` chains the rows, triggers allow only the one-time close and refuse `DELETE` and `TRUNCATE` | on, no policies; `service_role` without `DELETE` and `TRUNCATE` | Fast Loop (`paper_ledger_enabled`: entry in save, close in reflect) | Fast Loop (equity curve, chain head for publication) |
| `bot_state` | `079` | Operational: the `getUpdates` offset of the owner's Telegram bot (MAIL-48) — one row per channel, no message text and no chat id | on, no policies | Bot (`telegram_bot_enabled`, after the commands of a run) | Bot (start of a run) |
| `symbol_mutes` | `080` | User settings: symbols muted from the owner's bot (MAIL-45) — symbol, end of the mute, time of the last `/mute`; symbols are added only through `config.toml` | on, no policies | Bot (`symbol_mutes_enabled`: `/mute` upserts, `/unmute` deletes) | Fast Loop (`symbol_mutes_enabled`: a muted symbol gets no paid analysis and no section, only its price snapshot), Bot (`/watch`) |

The functions in `public`, all `SECURITY INVOKER` (they run with the caller's privileges) except `schema_version()`. Migration `070` (BEZ-18) pins `search_path = public, extensions, pg_catalog` on `match_news_embeddings`, `persona_accuracy_weights` and `persona_accuracy_stats`, so a call from a session with another path cannot swap in a same-named object; `refresh_ml_feature_store`, `persona_vote_outcomes` and the trigger function still run with the caller's path:

| Function | Migration | Called by | `EXECUTE` after `038` |
|---|---|---|---|
| `refresh_ml_feature_store()` | `001`, `004` | Slow Loop, backtest (RPC) | `service_role` only |
| `match_news_embeddings(vector, int, text)` | `011`, `017` | Fast Loop (RAG, RPC) | `service_role` only |
| `match_news_embeddings_exact(vector, int, text)` | `063` | Fast Loop (RAG, RPC, `rag_exact_search_enabled`) | `service_role` only |
| `match_news_embeddings_pit(vector, int, text, timestamptz, text, int)` | `076` | Fast Loop (RAG, RPC, `rag_point_in_time_enabled`) | `service_role` only |
| `persona_accuracy_weights(int)` | `015`, `058` (without `stale_quote = true` settlements) | Fast Loop (RPC) | `service_role` only |
| `persona_accuracy_stats(int)` | `018`, `058` (without `stale_quote = true` settlements) | Fast Loop (RPC) | `service_role` only |
| `persona_vote_outcomes(int)` | `045`, `058` (without `stale_quote = true` settlements) | Fast Loop (RPC, `persona_vote_outcomes_enabled`), tools (`persona-ablation`) | `service_role` only |
| `persona_scoring_stats(int, text[])` | `065` (without `stale_quote = true` settlements) | Fast Loop (RPC, `persona_brier_ranking_enabled`) | `service_role` only |
| `stamp_prediction_settled_at()` | `029` | trigger on `prediction_logs` | `service_role` only; the trigger fires for any writer |
| `schema_version()` | `071` | Fast Loop and Slow Loop at start (RPC, NIEZ-30) | `service_role` only. `SECURITY DEFINER` with `search_path = ''`: it reads `supabase_migrations.schema_migrations`, which `service_role` cannot, and returns only the highest numeric version (`NULL` when the registry is missing or empty) |
| `save_calibration_scores(jsonb, smallint)` | `077` | Slow Loop (RPC, `calibration_judge_enabled`, WIA-65): the arithmetic calibration scores of the 30-day window in one bulk `UPDATE` of `confidence_calibration` and `calibration_score_version` | `service_role` only; `search_path = ''` |
| `paper_trades_chain_head()`, `paper_trades_close_only()`, `paper_trades_append_only()` | `073` | triggers on `paper_trades` | nobody; the triggers fire for any writer |

`scripts/prod-state-check.sh` reads several of these relations with plain `GET` requests and
the same `SUPABASE_KEY`; it writes nothing and calls no function.

One Storage bucket sits next to them (MAIL-24, `digest_archive_enabled`): `digests`, private
(`public = false`, created by `061` where the migration role may write `storage.buckets`,
otherwise by hand), with no policies on `storage.objects`. The Fast Loop uploads the owner's
full email as `<cycle_id>.html` (the UTC cycle start and the run number, no personal data)
with the service key and puts a signed URL valid for 7 days into the owner's email and push.
Anyone holding that URL can open the email, portfolio included, until it expires — a bearer
token, hence the short TTL and the path without personal data.

## Keys

Supabase issues two kinds of API keys, and PostgREST runs every request as the database role
the key maps to:

- **Public keys**: the publishable key (`sb_publishable_…`) and the legacy `anon` JWT. They
  are meant to ship in browser code, so they are not secrets. Requests run as `anon` (or as
  `authenticated` with a signed-in user's JWT) and get only what grants and RLS policies allow.
- **Secret keys**: the secret key (`sb_secret_…`) and the legacy `service_role` JWT. Requests
  run as `service_role`, which has `BYPASSRLS`. They belong only in GitHub Secrets and the
  local `.env`.

The app is backend-only and writes to almost every table, so it needs a secret key. After
`038` the public roles hold no privilege on any table, view, sequence or app function, and
RLS without policies hides every row from any role without `BYPASSRLS`. A public key would
read nothing and fail every write, while the run itself looked healthy. `Settings` therefore
refuses to start with a publishable key or an `anon` / `authenticated` JWT (BEZ-10): every
entry point stops with a validation error that names the fix and does not print the key.

`SUPABASE_DB_URL`, the Postgres connection string of the `postgres` role, is a separate secret
used only by the DB Migrate workflow. The app never reads it.

## Retention

Nothing is deleted automatically: there is no retention job and no TTL in the database.

- `prediction_logs`, `price_snapshots`, `council_votes`, `quota_alerts`, `shock_alerts`,
  `model_scorecards`, `feature_snapshots`, `regime_history`, `fx_rates`, `llm_calls`,
  `reference_predictions` and `shadow_predictions` grow by append or upsert and keep every row
  (a reference row is updated once, when its call is settled). Deleting a prediction also
  deletes its `council_votes` and `shadow_predictions` (`ON DELETE CASCADE`).
- `fundamentals_cache` holds one row per symbol, overwritten on refresh. The 144-hour TTL is
  applied when the row is read, not by deleting it. `earnings_cache` works the same way: one
  report date per stock symbol, overwritten by the weekly refresh, read only within 144 hours.
- `positions` is replaced as a whole by `main_tools.py import-positions --apply`: the new rows
  are inserted and the old ones deleted.
- `subscribers` is maintained by hand. Deleting a row removes that recipient's email address
  from the database; no other relation stores it.
- `bot_state` holds one row per bot channel, overwritten after every bot run that handled
  a command. `symbol_mutes` holds one row per muted symbol: `/mute` overwrites it, `/unmute`
  deletes it, and an expired mute stays until the next `/mute` or `/unmute` of that symbol
  (the Fast Loop ignores it after `muted_until`).
- `digests` grows by one upserted row per cycle, and the `digests` bucket keeps every archived
  email; the signed links expire after 7 days, the files do not.

## Rolling out Row Level Security

The order matters: once `038` runs, only a secret key can read or write, and once the BEZ-10
code is deployed, a public key stops every job at start.

1. **Key first.** Copy the secret key (`sb_secret_…`) from the Supabase dashboard (Project
   Settings → API Keys), or use the legacy `service_role` key. Put it into the GitHub
   `SUPABASE_KEY` secret (`scripts/gh-secrets-sync.sh` pushes it from `.env`) and into the
   local `.env`. `scripts/prod-state-check.sh --skip-github` prints the role of the configured
   key without printing the key.
2. **Then deploy** the code. With a secret key nothing changes; with a public key still set
   somewhere, that job fails at start with the BEZ-10 error, which points to step 1.
3. **Then migrate.** Run **🗄️ DB Migrate (manual)** with the dry run, check that it lists
   `038_enable_rls.sql`, and run it again with `dry_run` off.
4. **Check.** Dispatch the Deadman or a Fast Loop run: both must pass. A request with the
   publishable key must now get `permission denied` instead of rows.

Migration `001` and `002` contain an explicit `DISABLE ROW LEVEL SECURITY`. Running either of
them again by hand turns RLS off on `prediction_logs` or `price_snapshots`; run `038` again
afterwards.

### Rules for new migrations

- A new table needs `ALTER TABLE … ENABLE ROW LEVEL SECURITY` in its own migration;
  `038` covered only the tables that existed when it ran.
- A new function needs `REVOKE EXECUTE ON FUNCTION … FROM PUBLIC` and, if the app calls it,
  `GRANT EXECUTE … TO service_role`. Grants that name the Supabase roles go into a `DO` block
  guarded by `pg_roles`, as in `038`, because the test container has no such roles.
- New tables, views and sequences no longer get grants for `anon` and `authenticated`:
  `038` removed them from the default privileges of the migration role.

The container tests in `tests/infrastructure/test_migrations.py` enforce this:
`test_every_public_table_has_row_level_security` fails on a table without RLS, and
`TestSupabaseRoleEmulation` applies every migration to a database with Supabase-like roles and
default grants, then checks through `SET ROLE` that `anon` and `authenticated` reach nothing
and that `service_role` still reads, writes and calls the RPC functions. The emulation is not
Supabase: it has no PostgREST and no JWTs, and it runs Postgres 16, where only the owner of
`ml_feature_store` may refresh it.

`tests/infrastructure/test_supabase_repo_postgrest.py` (TEST-17) adds the missing seam between
`SupabaseRepository` and the schema: one module-scoped set of containers runs the same
migrations and roles behind a real `postgrest/postgrest` container (`PGRST_DB_MAX_ROWS=1000`,
as on hosted Supabase) with JWTs signed by a local test secret, and the repository talks to
it through `supabase-py` as in production. The contract tests cover `save_prediction` and the
idempotent upsert on `(symbol, timestamp_hour)`, `get_unverified_prediction` (oldest first,
the age window, prices back as `Decimal` in the `DECIMAL(14,4)` scale although PostgREST
sends numbers), `find_similar_predictions` through the `match_news_embeddings` RPC, reads
past the 1000-row cap, `PGRST204` for a column without its migration, and an `anon` key that
can write nothing. The one difference from Supabase: there is no gateway serving PostgREST
under `/rest/v1`, so the test points the client at the PostgREST root.
