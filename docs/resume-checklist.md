# Resume checklist: Fast Loop and Shock Watch crons

**Status (2026-10-23): both crons are resumed.** The resume moved to a new, empty Supabase project, so DB Migrate applied every migration from `001` to `042`, not only `026`–`042`. Steps 1–4, 6, 7 and 10 are done: every flag of step 4 is on, both experiments run from 2026-10-26 (the first scheduled cycle) to 2027-01-29, and the deadman alerts after 26 h. Still open: step 5 (no `HEARTBEAT_URL` secret yet), step 8's check of the number of Alpha Vantage keys, step 9 (the attestation flags stay off), a push channel for Shock Watch (`TELEGRAM_*` or `SLACK_WEBHOOK_URL`; without one it logs an error and sends nothing), and step 11 after the first scheduled run.

The daily Fast Loop cron has been commented out since 2026-08-17 (commit `02fc397`, decision 2): the loop cost money and its failures were silent, and nothing raised an alarm during the quiet weeks from 2026-08-17 to 2026-10-15. The hourly Shock Watch cron was paused on 2026-10-16 to match (`f886da0`). Waves A to R added what a resume needs: a cost ceiling per cycle, a daily Alpha Vantage budget, a daily deadman, an external heartbeat, explicit write and paid-call switches in the workflows, a cost ledger, a run history and a free reference signal. This page is the owner's list of steps, in order. Switching the `schedule` back on is the last step, and it is the owner's call.

The resume brings back the **full watchlist** of [`config.toml`](../config.toml): 43 stocks and ETFs plus 2 crypto, of which 42 have a price (`symbols_unsupported_price` drops 3).

## What a resumed week costs

All figures are the estimate of `UNIT_COST_USD` in [`src/domain/finops.py`](../src/domain/finops.py), an order-of-magnitude model, not an invoice; `config.toml` as of 2026-10-20.

| | Calls | USD |
|---|---:|---:|
| One cycle, every eligible symbol past the volatility gate (`main_tools.py cost-envelope`) | 588 | 1.7237 |
| One cycle at the FIN-12 ceiling (`max_paid_llm_calls_per_cycle = 100`, 4 symbols at a time) | 383 | 1.2112 |
| One week at the ceiling: 1 cycle a day × 5 trading days (the weekly line of `cost-envelope`) | | 6.06 |
| Weekly budget `fast_loop_weekly_budget_usd` | | 7.00 |

The ceiling cuts the council (at most 99 + 4 × 8 = 131 requests a cycle), not the predictions: a symbol whose move is below `council_volatility_threshold` spends only its prediction, so the 42 predictions, sentiment, news and embeddings all fit under 100 calls. A cycle where the gate filters every symbol costs nothing. Shock Watch and the deadman make no paid call. How the bound is computed is in [configuration](configuration.md) under **Fast Loop weekly budget**.

## Steps

### 1. Secret Supabase key (BEZ-04)

Put the project's secret key (`sb_secret_…`, Supabase dashboard → Project Settings → API Keys) or the legacy `service_role` JWT into the `SUPABASE_KEY` repository secret and into the local `.env` (`scripts/gh-secrets-sync.sh` pushes it from `.env`). `scripts/prod-state-check.sh --skip-github` prints the role of the configured key without printing the key. This comes before the deploy: the code refuses a public key at start (BEZ-10), and migration `038` turns on Row Level Security, after which only a secret key can read or write. [Data and access](data-and-access.md#rolling-out-row-level-security) has the full rollout order.

### 2. Merge and push

Merge the pending branches into `main` and push. The workflows run the code of the branch they start on, so the crons only pick up waves A to R once they are on `main`.

### 3. Run DB Migrate for `026`–`042`

Run **🗄️ DB Migrate (manual)** from the Actions tab with `dry_run` on, check that the log lists `026_feature_store_clean_rows.sql` to `042_reference_predictions.sql`, then run it again with `dry_run` off. `026` and `039` rebuild the `ml_feature_store` view and `038` enables RLS; none of the three has a flag.

### 4. Turn on the flags the migrations unlock

Each of these flags is off in `config.toml` only because its column or table did not exist before the migration. Turn them on after step 3 succeeded; [configuration](configuration.md) describes each one.

| Migration | Flag in `config.toml` | Note |
|---|---|---|
| `027` | `feature_set_version_enabled`, `feature_set_filter_enabled` | The filter trains only on rows of the current feature set, so the Slow Loop skips the shared model until enough of them exist. |
| `028` | `model_scorecard_baselines_enabled` | |
| `029` | `settled_at_windows_enabled` | |
| `030` | `recipe_version_enabled` | |
| `031` | `feature_snapshots_enabled` | |
| `032` | `interval_coverage_enabled` | |
| `033` | `regime_hysteresis_enabled` | Part of the prediction recipe: `config_hash` changes and a new recipe segment starts. |
| `034` | `gate_delta_enabled` | |
| `035` | `positions_currency_enabled` | |
| `036` | `fx_rates_cache_enabled` | |
| `037` | `experiment_arms_enabled` | |
| `040` | `cost_ledger_enabled` | |
| `041` | `cycle_runs_enabled` | |
| `042` | `reference_signal_enabled` | The free weekly reference signal in the Slow Loop (WIA-24); it can run before the Fast Loop resumes. |

### 5. Add the `HEARTBEAT_URL` secret

Create one check at an external heartbeat service (for example healthchecks.io), add its ping URL as the `HEARTBEAT_URL` repository secret, and `scripts/gh-secrets-check.sh` stops reporting it missing. Fast Loop, Slow Loop and Shock Watch ping the same URL at the end of each run (NIEZ-13), so the check's period has to cover the longest gap between runs: the weekly Slow Loop, a week plus a grace period. A tighter period for the daily Fast Loop or the hourly watch needs a separate check and secret per loop, which the workflows do not map yet.

### 6. Set the experiment windows of EXP-02 and EXP-03

Both experiments are defined in `config.toml` and inactive until the Fast Loop resumes. Set `experiments.EXP-02.start` / `end` (the blind council) and `experiments.EXP-03.start` / `end` (the memory A/B) so they begin with the first real cycle, and set `enabled = true` in the same commit. They change different prompt parts, so they may run together. `experiments` is part of the recipe, so this commit changes `config_hash`. See [configuration](configuration.md#experiments).

### 7. Set `liveness_fast_loop_max_gap_h`

`0` tells the deadman that the Fast Loop is paused on purpose. Set it to the gap that should raise an alert, for example `26`: the gap counts from midnight ET of each NYSE trading day, so `26` mails after two trading days without a cycle, and weekends and holidays raise none.

### 8. Check the cost settings

- `max_paid_llm_calls_per_cycle` (`100`) and `fast_loop_weekly_budget_usd` (`7.0`) are set. After a change to the watchlist, the council, the ceiling or the cron, compare the budget with the worst case again: the Fast Loop start logs a `WARNING` and adds a banner alert to the email when the budget no longer covers it.
- `fast_loop_cycles_per_day` equals the runs a day of the cron you uncomment (`1` for `30 5 * * 1-5`); `tests/test_config_toml.py` fails when the two drift apart.
- `ALPHA_VANTAGE_API_KEYS` holds at least two keys: the worst case is 42 `NEWS_SENTIMENT` requests a day and one free key allows 25 (FIN-32). With fewer keys the start warns.

### 9. Pages and the `attestation-log` branch before the attestation flags

`attestation_enabled` and `public_scorecard_enabled` stay off until their artefacts have a durable home: deploy GitHub Pages (Settings → Pages → Source: GitHub Actions) and protect the `attestation-log` branch, then turn both flags on (WIA-62). The workflows have no publish step yet, and `tests/test_workflow_env_wiring.py` fails either flag while no workflow that runs `main_agent.py` has an `actions/deploy-pages` or `git push` step, so the publish step lands first.

### 10. Uncomment the crons

- **Shock Watch** can go first and on its own: since ALR-02 it reads each symbol's change for the day from the free quote, so it does not need the Fast Loop's snapshots. Uncomment the `schedule` block in [`shock_watch_hourly.yml`](../.github/workflows/shock_watch_hourly.yml).
- **Fast Loop**: uncomment the `schedule` block in [`fast_loop_12h.yml`](../.github/workflows/fast_loop_12h.yml). The guard `tests/test_workflow_env_wiring.py::test_enabled_fast_loop_schedule_requires_deadman` runs only once the cron is active, and fails unless a workflow runs `main_deadman.py` from an active cron, `max_paid_llm_calls_per_cycle`, `fast_loop_weekly_budget_usd` and a positive `liveness_fast_loop_max_gap_h` are set in `config.toml`, and the agent step sets `PAID_CALLS_ENABLED: "true"` and `DB_WRITES_ENABLED: "true"`.

### 11. After the first scheduled run

Check that the email arrived with its cycle-cost section, that the run page shows the run summary, that the heartbeat check received its ping, that a `cycle_runs` row exists (with `cycle_runs_enabled`), and that the next deadman run passes.
