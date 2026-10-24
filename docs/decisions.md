# Decision log

The decisions behind the code that a reader cannot recover from the code alone: what was
chosen, what it replaced, what it costs, and the commit and the test that hold it in place.
Each entry is short on purpose (an ADR-lite). Two kinds of identifier appear in the code, in
[`config.toml`](../config.toml) and in the other documents:

- **`R-N`** is a shipped roadmap item, a feature with a number. The table below is the only
  place that defines the numbers, and
  [`tests/test_decision_log.py`](../tests/test_decision_log.py) fails when a file mentions an
  `R-N` that the table lacks.
- **`D-N`** is a decision, written out in the second part.

Every entry carries commit hashes, and the test checks that each one exists in the history.
What the repository does not record is not claimed here either: where a number or a reason
could not be recovered, the entry says nothing about it. For incidents and the guard that
closed each class of them, see the [engineering log](engineering-log.md).

## Roadmap items

R-1 to R-18 are the July 2026 roadmap; the commits that shipped them are titled with
`roadmap #N` and the same number. R-19 to R-27 are the nine council and AI upgrades of June
2026, which the code called `feature #N` or a bare `#N`. Those numbers collided with the
roadmap's (`#5` meant both cross-symbol contagion and the messenger digest), so they got new
ones here. A bare `#N` in an older code comment is one of three series: a roadmap item
(R-1 to R-18), a June upgrade (R-19 to R-27), or a numbered finding of the first audit
(`finding #N`), which has no entry here.

| ID | Item | Commit | Flag |
|---|---|---|---|
| R-1 | Five-second lead that opens the email and becomes its subject | `400e44a` | — |
| R-2 | Charts as inline HTML, no `quickchart.io` (see D-5) | `2f0ffe9` | — |
| R-3 | Council persona leaderboard by historical hit rate | `31e3e76` | — |
| R-4 | "Dzień 1" greeting for the first cycle instead of a wall of skips | `55a282f` | — |
| R-5 | Five-line messenger digest instead of the full report | `1499960` | `messenger_digest_enabled` |
| R-6 | Earnings gate: the volatility threshold tightens near a report | `09cfe8b` | `earnings_calendar_enabled` |
| R-7 | Yield-curve inversion as a RISK_OFF vote in the market regime | `dd86102` | `yield_curve_enabled` |
| R-8 | Council flips and sentiment jumps against the previous cycle | `8f14188` | `cycle_diff_enabled` |
| R-9 | Signals ranked by history-calibrated confidence | `5efb406` | `calibrated_confidence_enabled` |
| R-10 | Public calibration scorecard, aggregates only | `cf56051` | `public_scorecard_enabled` |
| R-11 | Free Shock Watch for out-of-cycle market shocks | `c00027e` | — |
| R-12 | Walk-forward model scorecard, rejected candidates included | `bf42370` | `model_scorecard_enabled` |
| R-13 | Decision receipts: an audit trail for every prediction | `e6647fe` | `receipts_enabled` |
| R-14 | Alpha Fusion Score from the free alpha sources | `02ef47f` | `alpha_fusion_enabled` |
| R-15 | Real portfolio from positions: weights, P/L, VaR; phase 2 sizes against existing exposure (`83a5e56`) | `ff96797` | `portfolio_positions_enabled` |
| R-16 | Commit-reveal attestation of predictions (see D-4) | `7c37fe1` | `attestation_enabled` |
| R-17 | Sunday narrative recap from the Slow Loop | `64c7eec` | `weekly_recap_enabled` |
| R-18 | Model move against the options-implied move (edge) | `1a9500f` | `options_flow_enabled`, `deribit_dvol_enabled` |
| R-19 | Debate mode: one revision round when round 1 of the council is split | `9d98e16` | `debate_mode_enabled` |
| R-20 | Confidence-gated model routing: cheap model first, frontier on low confidence | `9d98e16` | `model_routing_enabled` |
| R-21 | Adaptive persona weighting by historical hit rate | `9d98e16` | `persona_weighting_enabled` |
| R-22 | LLM-as-judge calibration of the stated confidence (Slow Loop) | `9d98e16` | `calibration_judge_enabled` |
| R-23 | Cross-symbol contagion: this cycle's verdicts of sector peers in the prompt | `9d98e16` | `contagion_enabled` |
| R-24 | Tool-use research agent for the largest moves | `9d98e16` | `tool_use_enabled` |
| R-25 | Regime-aware thresholds: RISK_OFF tightens the volatility gate | `9d98e16` | `regime_aware_enabled` |
| R-26 | Counterfactual dissent replay in the self-reflection diagnosis | `ef921b9` | — |
| R-27 | Outcome-aware rerank of the RAG analogues | `ef921b9` | `rag_outcome_weight` |

## Decisions

### D-1 · The model learns the return, not the price level

**Date** 2026-06-20 · **Status** in force · **Commits** `bae5111`, `5da4808`, `82b1fd9`, `eff54d7`

The first XGBoost target was the absolute price. `bae5111` replaced it with `target_return` of
the `ml_feature_store` view (migration `012`, breaking: the old weights were deleted and the
model retrained), made the baseline the zero return, and computed `price_delta` once, in the
view, so that training and serving read the same number. A return can be compared across
symbols and against a "no change" baseline; a price level is in the units of its own symbol.

From 2026-10-18 one shared model is fitted on the pooled predictions of all symbols over
dimensionless features, which also fixed the per-symbol weights file that each run
overwrote. Weights ship only when the candidate beats the strongest of three baselines (no
change, mean drift, the LLM direction alone) by a 2% relative margin on at least 60
out-of-sample points; otherwise the leg runs on the no-change baseline and the report and the
council prompt label it cold-start rather than presenting it as a forecast.

*Costs:* weights ship rarely, and with a short history never. *Replaced:* the absolute price
target of each symbol.

**Guards** [`tests/application/test_train_model_use_case.py`](../tests/application/test_train_model_use_case.py) `TestRun::test_trains_model_with_correct_features_and_target` · [`tests/infrastructure/test_xgboost_adapter.py`](../tests/infrastructure/test_xgboost_adapter.py) `TestTrain::test_baseline_rmse_is_zero_return_persistence` · [`tests/application/test_agent_graph.py`](../tests/application/test_agent_graph.py) `TestColdStartBaseline::test_cold_start_is_flagged_in_final_state`

### D-2 · The volatility gate is a rule of the domain

**Date** 2026-05-13 (first commit), hardened 2026-06-20 and 2026-10-21 · **Status** in force · **Commits** `3e4a37f`, `bae5111`, `9e41b51`, `25479b8`

Every paid call (LLM, Alpha Vantage, embeddings) sits behind one question: did the price move
enough to be worth analysing. The answer is `Asset.evaluate_volatility` and the threshold
composition `effective_threshold` in `src/domain/asset.py`, computed on a fraction, never a
percent. The market-regime and earnings multipliers compound but are at least 1.0, so the gate
can only tighten. The graph executes the decision and never reimplements it; FIN-08 moved the
composition of the threshold for every gate (analysis, council, tool use, diagnosis) into the
domain as well.

Two exceptions exist and both are explicit and budgeted: the Slow Loop (weekly recap, judge)
calls the LLM outside the gate under `slow_loop_max_paid_llm_calls_per_week`, and a ceiling on
paid calls per cycle cuts the council of the symbols past it.

*Costs:* a quiet market yields no analysis, no prediction and no settlement for most symbols
on most days, which lengthens the time to a meaningful track record.

**Guards** [`tests/application/test_agent_graph.py`](../tests/application/test_agent_graph.py) `TestVolatilityGateIsSystemic::test_no_optional_paid_port_touched_when_gated` · [`tests/domain/test_asset.py`](../tests/domain/test_asset.py) `TestAssetEvaluateVolatility::test_gates_on_fraction_not_percent`, `TestEffectiveThreshold::test_a_multiplier_below_one_never_loosens_the_gate` · [`tests/test_config.py`](../tests/test_config.py) `test_zero_volatility_threshold_rejected`

### D-3 · The council verdict is computed from the votes, not asked from a model

**Date** 2026-06-20 · **Status** in force · **Commits** `bae5111`, `9f2562a`

The council is a set of personas, each a vote with a confidence. The recommendation and the
strength of the consensus used to come from a chairman model; when that call failed, an
invented HOLD with 0.5 overwrote seven real votes. `derive_consensus` in `src/domain/council.py`
now computes both from the actual opinions: each recommendation collects the mass
`confidence × persona weight`, the largest mass wins, a tie goes BUY over SELL over HOLD, and
no opinions mean HOLD with strength 0. A persona whose call failed is not a vote (`9f2562a`).

*Costs:* the council cannot "overrule" its own votes with a synthesis. *Gains:* the verdict is
reproducible from the stored votes, so flips between cycles (R-8), persona weights (R-21) and
leave-one-out ablations are recomputed from data instead of trusted.

**Guards** [`tests/domain/test_council.py`](../tests/domain/test_council.py) `TestDeriveConsensus::test_three_buy_one_sell_three_hold_deterministic`, `TestDeriveConsensus::test_unanimous_buy_high_strength`

### D-4 · Commit-reveal attestation, with a claim that stays small

**Date** 2026-07-10, claim narrowed 2026-10-16, ledger 2026-10-24 · **Status** in force · **Commits** `7c37fe1`, `562fe73`, `5f005aa`, `60f95fb`

A track record is only believable if the predictions could not be tuned after the move. At
prediction time a SHA-256 commitment over the canonicalised prediction and a random salt is
written; at settlement the plaintext and the salt are revealed, and anyone can recompute the
hash. The canonicalisation is frozen from day one by golden tests, because a change would
break the verification of the whole history and cannot be repaired afterwards.

The claim is deliberately small: Git commit dates can be forged, so this is tamper-evidence
under branch protection, not a timestamp. On 2026-10-16 the owner chose to switch the
attestation and the public scorecard off and to make the documents say so until something is
actually published (`562fe73` removed claims that nothing backed). The append-only ledger on
the `attestation-log` branch (WIA-14) and the rule that `revealed_at` is stamped only from that
ledger (WIA-15) followed. `attestation_enabled` stays off until the owner has protected the
branch and enabled Pages, as the [resume checklist](resume-checklist.md) lists.

*Costs:* nothing is verifiable by a stranger until the owner switches it on. *Rejected:* an
external timestamping service for now (OpenTimestamps stays an optional later stage).

**Guards** [`tests/domain/test_attestation.py`](../tests/domain/test_attestation.py) `test_golden_hash_v2`, `test_commit_golden_flat` · [`tests/test_readme_claims.py`](../tests/test_readme_claims.py) `test_readme_links_attestation_docstring`

### D-5 · Charts are inline HTML

**Date** 2026-07-10 · **Status** in force · **Commits** `2f0ffe9`

The three charts of the email were `<img>` tags that pointed at `quickchart.io`. Gmail and
Outlook block remote images by default, so the middle of the report was holes, and every send
handed the portfolio's deltas and sentiment scores to a third party inside the URL. The charts
are table and `div` bars in the body of the message now, built with the technique of the
tested sparkline, and each bar keeps a minimum width and prints its value, so a −0.5% bar can
never look like −8%. The functions that returned a URL were renamed to return HTML.

*Costs:* a chart is less expressive than a rendered image. *Gains:* no external host, no data
in a URL, and the email renders the same with images blocked.

**Guards** [`tests/application/test_report_charts.py`](../tests/application/test_report_charts.py) `TestBuildChartUrl::test_returns_inline_html_not_url` · [`tests/application/test_report_builder.py`](../tests/application/test_report_builder.py) `TestDeltaChart::test_renders_inline_bars_not_a_remote_image`

### D-6 · The crons paused on 2026-08-17 and resumed on 2026-10-23

**Date** 2026-08-17 and 2026-10-23 · **Status** in force · **Commits** `02fc397`, `f886da0`, `7a7be05`, `87bd9b0`, `974e133`, `ef2af51`, `e457434`

The daily Fast Loop cron was commented out on 2026-08-17: the loop cost money and its
failures were silent, and nothing raised an alarm during the quiet weeks. On 2026-10-16 the
owner extended the pause to the hourly Shock Watch (`f886da0`), which compared prices with the
reference snapshots that the Fast Loop writes and went blind when they aged. Both crons resumed
on 2026-10-23, and that second decision supersedes the extension of the pause; the watch now
reads each symbol's change for the day from a free quote and no longer depends on the daily
run.

The resume waited for what the pause had shown to be missing (`7a7be05` and the commits around
it): a ceiling on the paid calls of a cycle, an explicit weekly budget, a daily deadman that
alerts after two trading days without a run, an external heartbeat, and write and paid-call
switches in the workflows. It moved to a new, empty Supabase database, so every history
restarts from zero; the conditions and the order of the owner's steps are in the
[resume checklist](resume-checklist.md). The experiments of D-9 start on 2026-10-26.

*Costs:* the paid-call ceiling cuts the council of the symbols past it, and the first weeks
have no track record to show. *Rejected:* staying paused, because nothing accumulates then, and
resuming without a budget, which is how the pause began.

**Guards** [`tests/test_workflow_env_wiring.py`](../tests/test_workflow_env_wiring.py) `test_enabled_fast_loop_schedule_requires_deadman` · [`tests/test_main_watch.py`](../tests/test_main_watch.py) `TestWatchCycle::test_all_stale_snapshots_log_blind_warning` · [`tests/test_config_toml.py`](../tests/test_config_toml.py) `test_real_config_toml_runs_the_blind_council_experiment_from_the_resume`

### D-7 · `config.toml` is the one place for non-secret configuration

**Date** 2026-06-03, tightened 2026-10-21 · **Status** in force · **Commits** `1a68875`, `e71a1fe`, `781fda0`, `c83ee77`

`1a68875` moved the configuration out of variables into a file. The committed `config.toml`
now holds everything that is not a secret: symbols, thresholds, flags, models,
budgets and the experiment definitions. `.env` and the repository secrets hold only secrets
and the two environment switches `DB_WRITES_ENABLED` and `PAID_CALLS_ENABLED`. The order of
precedence is environment, `.env`, `config.toml`, then the defaults in code, and a key in
`config.toml` that `Settings` does not know is a startup error, so a typo cannot silently do
nothing. A test builds the Fast Loop from the production file, and another fails when
non-secret configuration leaks into a workflow.

*Costs:* the file is public, so an address in it is public (the SEC contact there is an alias
for that reason). *Gains:* one versioned, reviewed, validated source, and a diff that shows
what changed between two runs.

**Guards** [`tests/test_config_toml.py`](../tests/test_config_toml.py) `test_every_real_config_toml_key_is_a_settings_field`, `test_unknown_toml_key_is_rejected` · [`tests/test_workflow_env_wiring.py`](../tests/test_workflow_env_wiring.py) `test_no_non_secret_config_leaks_into_workflow`

### D-8 · The owner's six decisions after the 2026-10-15 audit

**Date** 2026-10-16 · **Status** partly superseded by D-6 · **Commits** `d7bbecb`, `562fe73`, `f886da0`, `5da4808`, `0afd866`, `a8e4c27`, `3ccaf52`

An audit of the claims against the code ended in six choices, all taken on 2026-10-16:

1. **Audience.** For the next quarter StockAgent is a portfolio project that carries live,
   verifiable evidence, not a product for strangers and not a private tool. That orders the
   work: documents that claim only what the code does, then public proof, then everything else.
2. **The pause covers Shock Watch** (`f886da0`). Superseded by D-6.
3. **Attestation and public scorecard off, documents honest** (`562fe73`, D-4), with the
   durable ledger and Pages as later steps.
4. **One shared model** over dimensionless features (`5da4808`, D-1).
5. **The pinned TTL test changes.** The fundamentals cache lives 144 hours instead of 24, so
   that a weekly Slow Loop refresh is still fresh for the Fast Loop of the six days after it;
   the test now states the cadence invariant `124 < TTL < 167` (`0afd866`).
6. **mypy scope.** The gate covers `src` and the entry points, and `tests/` joins it directory
   by directory, only when a directory is clean under strict mode, starting with
   `tests/domain` (`a8e4c27`, `3ccaf52`). A directory added to the gate cannot be removed.

**Guards** [`tests/domain/test_value_objects.py`](../tests/domain/test_value_objects.py) `test_fundamentals_ttl_spans_weekly_refresh_cadence` · [`tests/test_quality_gate_sync.py`](../tests/test_quality_gate_sync.py) `test_mypy_ratchet_keeps_the_typed_test_directories`, `test_bare_mypy_config_does_not_cover_untyped_tests`

### D-9 · Recipe changes wait behind flags while EXP-02 and EXP-03 run

**Date** 2026-10-24, confirmed 2026-10-25 · **Status** in force · **Commits** `c2e00b5`, `5a315c8`, `95dc8cf`, `a3d7ac8`, `d769137`, `2ee4ae5`, `dd116f4`, `a8b59a8`

Two in-loop experiments compare arms on the same cycles from 2026-10-26 to 2027-01-29: EXP-02
(a blind council that votes without the main LLM's verdict) and EXP-03 (predictions with and
without lessons and RAG analogues). A change to the recipe in the middle of that window would
make the two arms differ in more than the arm. So every change to how a prediction is made or
settled (the prompt and the output schema of the LLM, thresholds and gates, the horizon, the
settlement, the entry price, what the Fast Loop serves from the model) and every new write to
the database goes behind a new flag that is off by default. Off, the behaviour and the golden
snapshot of the email stay byte for byte what they were.

Each new `Settings` field is classified exactly once as a secret, as outside the recipe
(presentation, delivery, the switch of a write) or as part of the recipe; the recipe fields
form the `config_hash` of the `recipe_version`, so a flag that changes the forecast opens a
new segment of the track record instead of mixing two recipes. A new recipe field changes the
hash even while its flag is off, and the owner accepted the new segment at the merge.

*Costs:* improvements wait for the end of January 2027 (the earnings calendar of DAT-40, the
NYSE calendar horizon of WIA-28, the split correction of DAT-23 and the return-over-sigma
target of ML-11 are built and off). *Rejected:* shipping them now and discarding the experiments.

**Guards** [`tests/test_config.py`](../tests/test_config.py) `TestConfigHashScope::test_every_field_is_classified_exactly_once`, `TestConfigHashScope::test_flipping_any_recipe_flag_moves_the_hash` · [`tests/test_config_toml.py`](../tests/test_config_toml.py) `test_real_config_toml_runs_the_blind_council_experiment_from_the_resume`

### D-10 · Provider failures have a taxonomy, and broad handlers only go down

**Date** 2026-10-24 · **Status** in force · **Commits** `806bb00`, `dd69f93`, `5b65b4b`

`src/application/errors.py` defines `PortError` and four classes under it: `ProviderUnavailable`,
`QuotaExhausted`, `MalformedPayload` and `NotFound`. An adapter translates the exception of its
provider into one of them and keeps the provider's type as a base class, so an old
`except requests.RequestException` still works while the graph and the entry points catch the
taxonomy. The type of a translated error is logged with its origin, for example
`ProviderUnavailable[HTTPError]`. New code does not write `except Exception`; a test counts the
broad handlers in `src` and the entry points, and its baseline may only be lowered: when a
change really needs a new best-effort handler, another one is narrowed in the same change.

*Costs:* each adapter needs a translation. *Gains:* a quota, an outage and a garbage answer are
told apart in the logs, the telemetry and the degradation paths, and a programmer's error stops
passing for a missing signal.

**Guards** [`tests/test_exception_ratchet.py`](../tests/test_exception_ratchet.py) `test_broad_except_count_does_not_grow`

### D-11 · The email is a registry of sections

**Date** 2026-10-24 · **Status** in force · **Commits** `0509d91`, `608f3a8`, `25b9554`

The email used to be one function with a slot per section. It is now the tuple
`REPORT_SECTIONS` in `src/application/reporting/registry.py`, rendered by iterating it, one
order for the HTML and the plain text. A section is a module with a `ReportSection` (a key, an
HTML and a text renderer, and the `ReportContext` fields it reads), one entry in the registry
and one context field. Every context field has exactly one consumer, and the golden snapshot
proves that the registry renders the same bytes as the slots did. The order of the keys is also
fixed in the registry test, and the owner gave a standing approval to append the keys of later
sections there.

*Costs:* a new section touches four places. *Gains:* the order of the mail is data, a field that
no section reads fails a test, and the section table in [the report email](report-email.md) is
checked against the registry.

**Guards** [`tests/application/test_report_registry.py`](../tests/application/test_report_registry.py) `test_registry_covers_every_slot_with_html_and_text`, `test_every_report_context_field_has_exactly_one_consumer`

### D-12 · A private archive of the email, no public web digest

**Date** 2026-10-24 · **Status** in force · **Commits** `0b0a92d`, `1f72838`

The web digest (U3) wrote the full private digest, watchlist symbols and all, to a file under
`public/`, next to the scorecard and the attestation log, and a guard in the workflow tests kept
every workflow from deploying it. The email carries the owner's portfolio and the data of
third-party providers, whose terms are not reviewed for redistribution (see *Data sources and
terms* in [how it works](how-it-works.md)), so a file that only a guard keeps off a public page
was the wrong default. It was removed with its code, its settings and its tests (owner
decision of 2026-10-24), and `digest_archive_enabled` replaces it: the Fast Loop uploads the owner's email to a
private Supabase Storage bucket and renders it again with an "open in the browser" link, a
signed URL valid for 7 days, which only the owner's email and push carry. The public scorecard
keeps its own publisher and carries aggregates of the agent's own predictions only.

*Costs:* a link expires after a week and a stranger cannot read an issue. *Gains:* no private
file sits one configuration mistake away from a public page, and a guard keeps the workflows
from publishing anything but the scorecard and ledger directories.

**Guards** [`tests/infrastructure/test_supabase_storage_archive.py`](../tests/infrastructure/test_supabase_storage_archive.py) `test_store_uploads_private_and_returns_signed_url`, `test_object_path_is_the_cycle_id_without_personal_data` · [`tests/test_workflow_env_wiring.py`](../tests/test_workflow_env_wiring.py) `test_workflows_publish_only_the_public_artifact_dirs`
