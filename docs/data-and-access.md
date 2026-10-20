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
**tools** = `main_tools.py`, **backtest** = `main_backtest.py`. A local run writes nothing
unless `DB_WRITES_ENABLED=true` ([running-locally](running-locally.md#local-runs-are-read-only-by-default)).

| Relation | Migrations | Data class | RLS after `038` | Written by | Read by |
|---|---|---|---|---|---|
| `prediction_logs` | `001`, columns added up to `037` | Model output: predictions, council verdicts, receipts, settlements, embeddings | on, no policies | Fast Loop (prediction, settlement, calibration, attestation reveal) | Fast Loop (reflection, RAG, report, track record), tools (`evaluate`) |
| `price_snapshots` | `002`, `013` | Market data: one price per symbol and hour | on, no policies | Fast Loop (every cycle) | Fast Loop, Shock Watch (fallback reference price when the free quote has no fresh daily change, ALR-02), Deadman (database probe) |
| `ml_feature_store` | materialized view, `001` rebuilt in `004`, `012`, `022`, `026`, `027`, `039` | Derived training features from `prediction_logs` | not applicable (a materialized view has no RLS; grants only) | Slow Loop and backtest refresh it (`refresh_ml_feature_store`) | Slow Loop, backtest |
| `fundamentals_cache` | `006` | Market data: Alpha Vantage fundamentals cache | on, no policies | Fast Loop | Fast Loop |
| `council_votes` | `007` | Model output: one vote per persona and prediction | on, no policies | Fast Loop | Fast Loop (persona weights and track record) |
| `quota_alerts` | `010` | Operational: API quota exhaustion events | on, no policies | Fast Loop, Slow Loop | Fast Loop (report banner) |
| `subscribers` | `014` | **Personal data**: recipient email addresses and watchlists | on, no policies | nobody in the app (edited by hand in Supabase) | Fast Loop (`subscriptions_enabled`) |
| `model_scorecards` | `019`, `028` | Model output: walk-forward evaluation of each training run | on, no policies | Slow Loop | Fast Loop (report), Deadman (last training run) |
| `shock_alerts` | `021` | Operational: one sent price-shock alert per symbol and day | on, no policies | Shock Watch | Shock Watch (debounce) |
| `positions` | `023`, `035` | **Owner's financial data**: holdings, quantities, average cost | on, no policies | tools (`import-positions --apply`) | Fast Loop (`portfolio_positions_enabled`), tools |
| `feature_snapshots` | `031` | Derived features: the vector each prediction was made from | on, no policies | Fast Loop (`feature_snapshots_enabled`) | nobody yet |
| `regime_history` | `033` | Model output: market regime per cycle | on, no policies | Fast Loop | Fast Loop (regime hysteresis) |
| `fx_rates` | `036` | Market data: NBP exchange-rate cache | on, no policies | Fast Loop | Fast Loop |
| `llm_calls` | `040` | Operational: one row per LLM or embedding request (tokens, cost, run, node, symbol), no prompt text | on, no policies | Fast Loop, Slow Loop (`cost_ledger_enabled`) | nobody in the app yet |
| `cycle_runs` | `041` | Operational: one row per loop run (run id and attempt, start, duration, symbol counts, paid calls per source, email delivered), no symbols | on, no policies | Fast Loop (`cycle_runs_enabled`) | nobody in the app yet |

The functions in `public`, all `SECURITY INVOKER` (they run with the caller's privileges):

| Function | Migration | Called by | `EXECUTE` after `038` |
|---|---|---|---|
| `refresh_ml_feature_store()` | `001`, `004` | Slow Loop, backtest (RPC) | `service_role` only |
| `match_news_embeddings(vector, int, text)` | `011`, `017` | Fast Loop (RAG, RPC) | `service_role` only |
| `persona_accuracy_weights(int)` | `015` | Fast Loop (RPC) | `service_role` only |
| `persona_accuracy_stats(int)` | `018` | Fast Loop (RPC) | `service_role` only |
| `stamp_prediction_settled_at()` | `029` | trigger on `prediction_logs` | `service_role` only; the trigger fires for any writer |

`scripts/prod-state-check.sh` reads several of these relations with plain `GET` requests and
the same `SUPABASE_KEY`; it writes nothing and calls no function.

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
  `model_scorecards`, `feature_snapshots`, `regime_history`, `fx_rates` and `llm_calls` grow by
  append or upsert and keep every row. Deleting a prediction also deletes its `council_votes`
  (`ON DELETE CASCADE`).
- `fundamentals_cache` holds one row per symbol, overwritten on refresh. The 144-hour TTL is
  applied when the row is read, not by deleting it.
- `positions` is replaced as a whole by `main_tools.py import-positions --apply`: the new rows
  are inserted and the old ones deleted.
- `subscribers` is maintained by hand. Deleting a row removes that recipient's email address
  from the database; no other relation stores it.

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
