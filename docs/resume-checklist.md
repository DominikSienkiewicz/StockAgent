# Resume checklist: Fast Loop and Shock Watch crons

**Status (2026-10-23): both crons are resumed.** The resume moved to a new, empty Supabase project, so DB Migrate applied every migration from `001` to `042`, not only `026`–`042`. Steps 1–4, 6, 7 and 10 are done: every flag of step 4 is on, both experiments run from 2026-10-26 (the first scheduled cycle) to 2027-01-29, and the deadman alerts after 26 h. Still open: step 5 (no `HEARTBEAT_URL` secret yet), step 8's check of the number of Alpha Vantage keys, step 9 (the `attestation-log` branch, its protection and the attestation flags, which stay off), a push channel for Shock Watch (`TELEGRAM_*` or `SLACK_WEBHOOK_URL`; without one it logs an error and sends nothing), and step 11 after the first scheduled run.

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

### 3. Run DB Migrate for `043`–`085`

Run **🗄️ DB Migrate (manual)** from the Actions tab with `dry_run` on, check that the log lists `043_attestation_payload.sql` to `085_function_search_path_rest.sql` (production has `001`–`042` since the resume on 2026-10-23; numbers `066`–`069` are unused), then run it again with `dry_run` off. `026` and `039` rebuild the `ml_feature_store` view and `038` enables RLS; none of the three has a flag. `047` rebuilds the view again with `LAG` over every logged prediction (ML-17), and `050` adds `settled_at` and `label_horizon_hours` to it (ML-25); neither has a flag. `048` (ML-19) upper-cases and trims `prediction_logs.predicted_trend`, then adds the `prediction_logs_predicted_trend_known` CHECK (`BULLISH`, `BEARISH`, `SIDEWAYS` or `NULL`) as `NOT VALID`, so history is not scanned; its `NOTICE` in the log counts the rows still outside the list. Such a row rejects every later `UPDATE` (settlement, calibration) until its trend is fixed by hand; when the count is `0`, `ALTER TABLE prediction_logs VALIDATE CONSTRAINT prediction_logs_predicted_trend_known` finishes the job. `051` adds `prediction_logs.trend_probabilities` (ML-29) and `052` adds `model_scorecards.feature_stats` (ML-21); both columns are written only behind their flags in step 4. `055` adds `prediction_logs.stale_quote` (DAT-14), `056` adds `fundamentals_cache.market_cap` (DAT-21) and `057` creates the `price_level_alerts` table (ALR-11), each written only behind its flag. `058` rebuilds the `ml_feature_store` view and the persona RPCs of `015`, `018` and `045` without settlements on a stale quote (`stale_quote IS NOT TRUE`, DAT-14); it has no flag, and rows stay in until `stale_quote_enabled` marks them. `059` adds the session-close settlement columns of `prediction_logs` (WIA-61), written only behind its flag. `060` adds `model_scorecards.weights_sha256` (CI-13), written only behind its flag. `064` adds the training provenance columns of `model_scorecards` (ML-22), written only behind its flag. `075` creates the `shadow_predictions` table (ML-33), written only behind its flag. `077` adds the Slow Loop RPCs `db_size_report()` (NIEZ-37, no flag: the weekly size report starts by itself) and `save_calibration_scores` (WIA-65); `078` adds the diagnosis judge columns of `prediction_logs` (EXP-10); `079` and `080` create the bot tables `bot_state` and `symbol_mutes` (MAIL-48, MAIL-45); `081` adds `shock_alerts.level` and moves the daily unique key to `(symbol, alert_date, level)` (ALR-16); `082` adds `council_votes.round` and moves the vote key to `(prediction_id, investor_name, round)`, with the persona RPCs counting round 1 only (ML-40); `083` adds the trend columns of `shadow_predictions` (EXP-26). Each is written only behind its flag in step 4.

Migrations `070`, `071`, `084` and `085` have no flag and belong to the same DB Migrate run. `070` (BEZ-18) pins `search_path` on `match_news_embeddings`, `persona_accuracy_weights` and `persona_accuracy_stats`; `084` (BEZ-17) does the same for `refresh_ml_feature_store`, `persona_vote_outcomes` and the settlement trigger function `stamp_prediction_settled_at`, and `085` (BEZ-33) for `match_news_embeddings_exact`, `persona_scoring_stats` and the three `paper_trades` trigger functions. `071` (NIEZ-30) adds `public.schema_version()`, which the Fast Loop and the Slow Loop call at start to switch off a flag whose migration the database lacks (see [configuration](configuration.md#feature-flags)). Until `071` is applied the call fails, every flag stays on and the run logs a `WARNING` — nothing breaks. Check it after the run with `select public.schema_version();` in the SQL Editor: it must return `85`, the last migration of the run; `NULL` means the migrations were not applied through `supabase db push`.

### 4. Turn on the flags the migrations unlock

Each of these flags is off in `config.toml` only because its column or table did not exist before the migration. The flags of `027`–`042` are on since 2026-10-23; turn on the ones from `044` after step 3 succeeded; [configuration](configuration.md) describes each one.

Status (2026-10-26): step 3 applied `043`–`085`. The flags of that range that are outside `config_hash` and do not wait for 2027-01-29 are on: `calibration_score_version_enabled`, `correction_category_enabled`, `model_scorecard_run_id_enabled`, `model_scorecard_feature_stats_enabled`, `council_votes_upsert_enabled`, `stale_quote_enabled`, `fundamentals_market_cap_enabled`, `session_close_settlement_enabled`, `model_scorecard_weights_sha256_enabled`, `digest_archive_enabled`, `model_scorecard_provenance_enabled`, `persona_brier_ranking_enabled`, `executable_entry_enabled`, `paper_ledger_enabled`, `shadow_predictions_enabled` and `shock_escalation_enabled`. Still off: `telegram_bot_enabled` and `symbol_mutes_enabled` until the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` secrets exist, `price_level_alerts_enabled` until `price_alerts` lists levels, `earnings_cache_enabled` until it goes on with `earnings_calendar_enabled` (step 12), the recipe flags `persona_vote_outcomes_enabled` and `trend_probabilities_enabled`, and every flag the table below defers to 2027-01-29.

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
| `035` | — | No flag since PORT-23: the positions adapter and the CSV import always use `currency` and `purchase_date`. |
| `036` | `fx_rates_cache_enabled` | |
| `037` | `experiment_arms_enabled` | |
| `040` | `cost_ledger_enabled` | |
| `041` | `cycle_runs_enabled` | |
| `042` | `reference_signal_enabled` | The free weekly reference signal in the Slow Loop (WIA-24); it can run before the Fast Loop resumes. |
| `044` | `calibration_score_version_enabled` | The calibration judge stores the Brier rule with `calibration_score_version = 2` (WIA-37); only matters with `calibration_judge_enabled`. |
| `045` | `persona_vote_outcomes_enabled` | Persona weights and the persona track record score HOLD on crypto with the crypto flat band (WIA-51); part of `config_hash`, so turning it on starts a new `recipe_version` segment. |
| `046` | `correction_category_enabled` | Each paid mistake diagnosis stores its cause category, and the lessons panel shows a histogram of the categories (WIA-57). |
| `049` | `model_scorecard_run_id_enabled` | One `run_id` per Slow Loop run on every scorecard row (ML-24); a same-day re-run no longer merges with the cron run. |
| `051` | `trend_probabilities_enabled` | Also a product decision, not only the migration: it changes the prediction prompt (the three-trend distribution, ML-29) and is part of the recipe, so `config_hash` changes and a new recipe segment starts. |
| `052` | `model_scorecard_feature_stats_enabled` | The training feature distribution and the 30-day PSI on every scorecard row (ML-21); the model card warns above `0.25`. |
| `053` | `earnings_cache_enabled` | The Fast Loop reads the earnings date only from `earnings_cache`, which the Slow Loop refreshes (FIN-11); it is outside `config_hash` (the same date from a cache, not a new recipe), so turning it on does not start a new `recipe_version` segment. |
| `054` | `council_votes_upsert_enabled` | Council votes are upserted on the unique `(prediction_id, investor_name)` key (NIEZ-14), so a retried save no longer duplicates them. |
| `055` | `stale_quote_enabled` | Predictions and settlements made on a stale quote get `prediction_logs.stale_quote = true` and drop out of the hit rate (DAT-14). |
| `056` | `fundamentals_market_cap_enabled` | The Slow Loop stores the market capitalisation from the same `OVERVIEW` response; the significant insider purchase threshold scales with it (DAT-21). |
| `057` | `price_level_alerts_enabled` | The watch alerts on user-set price levels from `price_alerts` (ALR-11); the list ships empty, add levels first. |
| `059` | `session_close_settlement_enabled` | Settlements also record the official NYSE session close with its date and source, and the email shows its hit rate next to the `/quote` one (WIA-61, transition period). |
| `060` | `model_scorecard_weights_sha256_enabled` | The sha256 of the weights file of the run on the scorecard row (CI-13), the same file the `model-latest` Release holds and the predictions hash as `model_version`. |
| `061` | `digest_archive_enabled` | The full email goes to the private Supabase Storage bucket `digests` and the owner's email and push carry an `Otwórz w przeglądarce` link valid for 7 days (MAIL-24); `061` creates the `digests` index table and, where the migration role may write `storage.buckets`, the private bucket. Check under Dashboard → Storage that `digests` exists and **Public bucket** is off; if it is missing, create it there with Public off. |
| `062` | `split_detection_enabled` | A suspected split (the quote's `pc` against the last snapshot) adjusts the gate, the settlement and `price_delta` and is recorded in `corporate_actions` (DAT-23); part of `config_hash` and it changes settlements, so turn it on after EXP-02/03 end (2027-01-29). |
| `063` | `rag_exact_search_enabled` | RAG analogs come from an exact nearest-neighbour scan instead of the ivfflat index at `probes = 1` (ML-23); it changes the analogs in the prediction prompt and is part of `config_hash`, so turn it on after EXP-03 ends (2027-01-29). |
| `064` | `model_scorecard_provenance_enabled` | The training provenance on every scorecard row (ML-22): dataset fingerprint, date range, rows per symbol, parameters with the seed, XGBoost version, feature set version and commit. |
| `065` | `persona_brier_ranking_enabled` | The persona leaderboard of the email sorts by the Brier skill of the personas' confidence and shows `Brier` / `BSS` under the hit rate (WIA-44); presentation only, so it does not start a new `recipe_version` segment. |
| `072` | `executable_entry_enabled` | Settlements also record the executable entry — the open of the first NYSE session after the prediction from the same `/quote` — and the email adds a second equity curve "from the open" (WIA-54). A measurement, not part of `config_hash`. |
| `073` | `paper_ledger_enabled` | Every real prediction opens a paper position in the append-only, hash-chained `paper_trades` and its settlement closes it (WIA-59); the email's equity curve reads the ledger, and with `attestation_enabled` the chain head is published once a day. A record and a measurement, not part of `config_hash`. |
| `074` | `invalidation_level_enabled` | Each prediction carries the price level that breaks its thesis, validated by the domain (right side of the price, at most 15% away); the email shows it under the reasoning and the Shock Watch alerts once when it is crossed (ML-27). It changes the prediction prompt and is part of `config_hash`, so turn it on after EXP-02/03 end (2027-01-29). |
| `075` | `shadow_predictions_enabled` | The Fast Loop stores the champion's and the challengers' forecasts of the ML leg and the Slow Loop reports the promotion verdict in its step summary (ML-33). |
| `076` | `rag_point_in_time_enabled` | RAG analogs come only from predictions settled before the moment of the prediction, without the same symbol's last 72 hours (`match_news_embeddings_pit`, ML-41); it changes the analogs in the prediction prompt and is part of `config_hash`, so turn it on after EXP-03 ends (2027-01-29). |
| `077` | `calibration_judge_enabled` | The weekly confidence calibration writes its arithmetic score with one bulk RPC; there is no LLM judge any more (WIA-65). Part of `config_hash`, so after EXP-03 ends (2027-01-29). |
| `078` | `diagnosis_judge_enabled`, then `diagnosis_judge_rejects_enabled` | After EXP-03 ends (2027-01-29). The judge first only grades up to 10 diagnoses a week (shadow mode); grade 20 of them yourself with the same rubric, and only when you agree with it turn on the rejections, which take low-graded diagnoses out of lessons and RAG (EXP-10). `batch_diagnosis_enabled` (EXP-17, no migration) waits for the same date. |
| `079` | `telegram_bot_enabled` | The owner bot (`bot_poll.yml`, every 15 minutes) answers `/last`, `/pos`, `/watch`, `/mute`, `/unmute`, `/why` and `/help` from the chat `TELEGRAM_CHAT_ID` (MAIL-48). Needs the secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (a private chat or a group, not a channel) and no webhook on the bot. |
| `080` | `symbol_mutes_enabled` | A symbol muted from the bot skips the paid analysis and the email sections but keeps its price snapshot (MAIL-45). |
| `081` | `shock_escalation_enabled` | The Shock Watch may send a second alert a day for a symbol whose move doubled or reversed past the threshold (ALR-16). |
| `082` | `council_votes_upsert_enabled` | Its upsert key now includes the round. `debate_mode_enabled` also needs `082`, but stays off until EXP-03 ends (2027-01-29), like `model_routing_enabled` (FIN-21; the frontier escalation of the Slow Loop is not charged to the weekly Slow Loop allowance yet — decide that before turning routing on). |
| `083` | `shadow_prompt_variant_enabled` | A prompt or model variant runs in shadow for a fixed sample of symbols above the gate, at most 5 calls a cycle within the cycle ceiling (EXP-26). Set `shadow_prompt_variant` first; after EXP-03 ends (2027-01-29). |

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

`attestation_enabled` and `public_scorecard_enabled` stay off until their artefacts have a durable home. The workflows are ready (WIA-14, WIA-16); what is left is repository settings, which only the owner can change.

**9a. Create the `attestation-log` branch** (an orphan branch holding only the two ledger files, created from a throwaway clone so your checkout is untouched). The Fast Loop checks this branch out into `public/attestation/` before the agent runs, and the separate `publish-attestation-ledger` job (the only one with `contents: write`, no API key) appends the new lines in a `[skip ci]` commit and pushes to this branch only, never to `main`:

```bash
ledger=$(mktemp -d) \
  && git clone --quiet --no-checkout "$(git remote get-url origin)" "$ledger" \
  && git -C "$ledger" switch --orphan attestation-log \
  && : > "$ledger/commitments.jsonl" && : > "$ledger/reveals.jsonl" \
  && git -C "$ledger" add commitments.jsonl reveals.jsonl \
  && git -C "$ledger" commit -m "ledger: init [skip ci]" \
  && git -C "$ledger" push origin attestation-log \
  && rm -rf "$ledger"
```

**9b. Protect it** (no force-push, no deletion, admins included; pushes stay open, because the workflow pushes with the job token):

```bash
gh api -X PUT "repos/DominikSienkiewicz/StockAgent/branches/attestation-log/protection" --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

**9c. Check the pinned `actions/download-artifact`** in `fast_loop_12h.yml` (`publish-attestation-ledger` job) still points at the commit of its tag: `gh api repos/actions/download-artifact/git/ref/tags/v4.3.0 --jq .object.sha` must print the SHA in the `uses:` line. The job runs only when the agent appended to the ledger, so a wrong SHA would surface on the first run with the flag on.

**9d. Turn the flag on** (attestation also needs migration `043` from step 3, or each cycle runs without it): set `attestation_enabled = true` in `config.toml`. A flag on without the branch fails the Fast Loop in its first step, before any paid call, with an `::error` annotation. Then run the Fast Loop once by hand and check that `attestation-log` has a `ledger: run <id> [skip ci]` commit whose `commitments.jsonl` gained **one** cycle root line (`cycle_id`, `root`, `n_leaves`, `leaves`, WIA-60), even when the cycle saved no prediction. Reveals follow in later runs: the sweep after the report appends them to `reveals.jsonl` without stamping, and the next run stamps `revealed_at` only for the reveals it finds in its checkout of `attestation-log` (WIA-15), so a cancelled run or a failed push costs nothing but a repeat. The first push is also the start of the verifiable track record (WIA-27). Anyone can then check the log with the three commands of [Verify it yourself](configuration.md#verify-it-yourself).

**9e. Pages for the public scorecard:** switch GitHub Pages on with GitHub Actions as its source (the same as Settings → Pages → Source: GitHub Actions); the first command creates the site, the second one switches an existing site to Actions, the third shows the result:

```bash
gh api -X POST "repos/DominikSienkiewicz/StockAgent/pages" -f build_type=workflow
gh api -X PUT "repos/DominikSienkiewicz/StockAgent/pages" -f build_type=workflow
gh api "repos/DominikSienkiewicz/StockAgent/pages" --jq '.build_type, .html_url'
```

Then turn `public_scorecard_enabled` on. `tests/test_workflow_env_wiring.py::test_enabled_public_artifacts_have_a_durable_publish_step` checks each flag against its own mechanism, in the job that runs `main_agent.py` or in any job that waits for it through `needs` (directly or transitively): `public_scorecard_enabled` needs an `actions/upload-pages-artifact` of the scorecard's directory and an `actions/deploy-pages` step, which the Fast Loop has since WIA-16; `attestation_enabled` needs a `git add` of both ledger files (`attestation_commitments_path`, `attestation_reveals_path`) and a `git push`, which the `publish-attestation-ledger` job has since WIA-14, because a Pages deploy replaces the whole site and cannot keep an append-only ledger. The public scorecard has its own deploy: [`scorecard_pages.yml`](../.github/workflows/scorecard_pages.yml) runs the free `main_scorecard.py` and deploys only `dirname(public_scorecard_path)` to Pages (WIA-52), and every Fast Loop run that wrote the page deploys it the same way (WIA-16). Once Pages is set up and `public_scorecard_enabled` is on, run **🏅 Public Scorecard** from the Actions tab with a `days` input that reaches back to the last settled predictions, then check the page URL on the run summary and `<page URL>/scorecard.json` next to it (WIA-81; its contract is [`docs/schemas/scorecard.schema.json`](schemas/scorecard.schema.json)).

**9f. Both public flags on, in one commit (WIA-62).** The code for both artefacts is ready and `config.toml` keeps `attestation_enabled` and `public_scorecard_enabled` off until 9a–9e are done and migration `043` is applied. Then set both to `true` in one commit; `tests/test_workflow_env_wiring.py::test_enabled_public_artifacts_have_a_durable_publish_step` passes each flag only next to its durable publish step, and `test_workflows_publish_only_the_public_artifact_dirs` fails any workflow that would publish a path outside the scorecard's directory and the ledger's. After the first two runs, check from a clone that `uv run python main_tools.py verify-attestation` against `attestation-log` prints `OK` (see [Verify it yourself](configuration.md#verify-it-yourself)). The public status page (NIEZ-33) is a third flag, `public_status_page_enabled`: it needs migration `041` with `cycle_runs_enabled` (on since 2026-10-23) and the scorecard flag, because it rides on the scorecard's Pages deploy; turn it on with them or later, then check `<Pages URL>/status/` and `<Pages URL>/status/badge.json`.

### 10. Uncomment the crons

- **Shock Watch** can go first and on its own: since ALR-02 it reads each symbol's change for the day from the free quote, so it does not need the Fast Loop's snapshots. Uncomment the `schedule` block in [`shock_watch_hourly.yml`](../.github/workflows/shock_watch_hourly.yml).
- **Public Scorecard** can go on its own once step 9 is done: uncomment the weekly `schedule` block in [`scorecard_pages.yml`](../.github/workflows/scorecard_pages.yml). It reads only the window of `track_record_days`, so until the Fast Loop settles new predictions a scheduled run writes no page and Pages keeps the last one.
- **Fast Loop**: uncomment the `schedule` block in [`fast_loop_12h.yml`](../.github/workflows/fast_loop_12h.yml). The guard `tests/test_workflow_env_wiring.py::test_enabled_fast_loop_schedule_requires_deadman` runs only once the cron is active, and fails unless a workflow runs `main_deadman.py` from an active cron, `max_paid_llm_calls_per_cycle`, `fast_loop_weekly_budget_usd` and a positive `liveness_fast_loop_max_gap_h` are set in `config.toml`, and the agent step sets `PAID_CALLS_ENABLED: "true"` and `DB_WRITES_ENABLED: "true"`.

### 11. After the first scheduled run

Check that the email arrived with its cycle-cost section, that the run page shows the run summary, that the heartbeat check received its ping, that a `cycle_runs` row exists (with `cycle_runs_enabled`), and that the next deadman run passes.

### 12. After EXP-02 and EXP-03 end: the earnings gate

Owner decision (DAT-40): `earnings_calendar_enabled` stays `false` while EXP-02 and EXP-03 run, because the earnings multiplier changes the volatility gate, which is part of the recipe both experiments compare. After their `end` dates (2027-01-29 at the latest), set `earnings_calendar_enabled = true` in `config.toml` (this commit changes `config_hash`). Since FIN-09 the Fast Loop asks Alpha Vantage for the earnings calendar once per run — one `EARNINGS_CALENDAR` request without a symbol covers the whole watchlist, with key rotation over `ALPHA_VANTAGE_API_KEYS` — so it no longer needs four keys; with `earnings_cache_enabled` (migration `053`) the Fast Loop asks nothing and the Slow Loop refreshes the cache. Check the next email: no `Alpha Vantage earnings` quota warning, and a symbol a few days before its report shows the earnings proximity in the Alpha Signals table.

The order of the two switches matters (DAT-40):

1. **DB Migrate has applied `053`** (step 3). Until then `earnings_cache` does not exist and the cache flag must stay `false`.
2. **Set both in one commit:** `earnings_calendar_enabled = true` and `earnings_cache_enabled = true`. The cache flag alone does nothing, and the Slow Loop refreshes `earnings_cache` only when both are on, so the cache is filled by the first Slow Loop run after the commit.
3. **Fill the cache before the first weekday cycle:** dispatch **🔵 Slow Loop (Weekly ML Retraining)** once from the Actions tab, or wait for the Sunday run. Until a refresh has run the cache is empty and no symbol gets the earnings multiplier (the gate stays as it was, it does not fail).
4. **Do not turn on the calendar without the cache** unless you accept one Alpha Vantage request in every cycle, made before the volatility gate.

### 13. After SEC lifts the EDGAR block and EXP-02/03 end: 8-K filings in the prompt

Owner decision (2026-10-25, DAT-17): `corporate_events_enabled` puts a fresh SEC 8-K current report into the prediction prompt and `fresh_8k` into `feature_snapshots`, so it changes the recipe both experiments compare and stays `false` until their `end` dates (2027-01-29 at the latest). It also needs EDGAR to answer: SEC blocks this client with a 403 (see [configuration](configuration.md), insider flow). Once SEC lifts the block and the experiments have ended, set `corporate_events_enabled = true` in `config.toml` (this commit changes `config_hash`; no migration) and start one manual Fast Loop run. Check the log for no `SEC EDGAR answered 403` warning; on the next email with a stock that filed an 8-K in the last 3 days, the prompt of its prediction carries the `<corporate_events>` block, and `feature_snapshots.features` carries `fresh_8k`.
