# Changelog

The history of StockAgent as milestones. The versions below were laid by hand on commits that
already existed (CI-21): there is no release tooling, and the tags are created and pushed by the
owner. Numbers `R-N` are the roadmap items of [`docs/decisions.md`](docs/decisions.md), the one
place that defines them together with their commits; `D-N` are its decisions. Dates are the
author dates of the commits. A commit count is `git rev-list --count <commit>`.

Each release lists what it shipped (items to commits) and a **Divergences** section: where the
history does not match the plan, the numbering or the commit messages. `pyproject.toml` still
says `0.1.0`; the milestones do not bump it.

## [Unreleased]

Everything after `fd7110c` (v0.3) up to the integration merge `cc6db87` (730 commits),
summarised by the merges and the decisions that shaped them rather than commit by commit:

- 2026-07-10: the follow-ups of the same day, after the roadmap closed: the free alpha sources
  switched on and a real SEC contact in the user agent (`bd92e2e`, `417ffc0`), schema migrations
  managed with the Supabase CLI (`c7986d6`), the Alpha Vantage keys wired into the shock-watch
  workflow (`a7440cb`).
- 2026-08-07: SonarCloud findings fixed across domain, CI and prompt safety (`f93fc76`), the
  Supabase action pinned and dependencies resolved from the lockfile (`38561fa`).
- 2026-08-17: the daily Fast Loop cron paused (`02fc397`, D-6).
- 2026-08-18: the README split into a router plus reference documents (`ed0fdb9`) and a CI gate
  on Markdown links that do not resolve (`1f044e7`).
- 2026-10-15 to 2026-10-16: the audit of the claims against what runs (`30f6303`, `d7bbecb`);
  the public commit-reveal and scorecard flags switched off because nothing published them
  (`562fe73`); the Anthropic SDK made a core dependency (`3e4c234`); the Shock Watch paused with
  the Fast Loop (`f886da0`).
- 2026-10-18: a read-only production state check (`b429800`).
- 2026-10-20, `7a7be05`: waves C, S and R: report and ops fixes, Row Level Security and secret
  hygiene, paid-call budgets, the cost ledger and the Fast Loop resume checklist.
- 2026-10-23: the weekday Fast Loop cron resumed with the deadman watching it (`30ad2ac`), the
  hourly Shock Watch resumed (`eb8222d`), the flags unlocked by migrations 027-042 turned on
  (`87a931f`), the EXP-02 and EXP-03 experiments running from the first resumed cycle (`a3d7ac8`).
- 2026-10-24, `7b44c99`: waves W1, W2 and round 3: attestation v2, the public scorecard, honest
  statistics, ML features, observability, data and alert fixes (migrations 043-059); EDGAR made
  to follow SEC fair access, with the EDGAR and Reddit sources that answer 403 switched off
  (`7dbd060`, `647b276`).
- 2026-10-25, `cc6db87`: program P1+P2: the port error taxonomy, the report section registry,
  the attestation proof, ML champion/challenger, the paper ledger, the data source terms, strict
  mypy over all tests (migrations 060-076).

## [v0.3] - 2026-07-10

Tag target: `fd7110c` (71 commits). The July roadmap, closed by the merge "close the roadmap".

### Shipped

| ID | Item | Commit |
|---|---|---|
| R-1 | Five-second lead that opens the email and becomes its subject | `400e44a` |
| R-2 | Charts as inline HTML, no `quickchart.io` (see D-5) | `2f0ffe9` |
| R-3 | Council persona leaderboard by historical hit rate | `31e3e76` |
| R-4 | "Day 1" greeting for the first cycle instead of a wall of skips | `55a282f` |
| R-5 | Five-line messenger digest instead of the full report | `1499960` |
| R-6 | Earnings gate: the volatility threshold tightens near a report | `09cfe8b` |
| R-7 | Yield-curve inversion as a RISK_OFF vote in the market regime | `dd86102` |
| R-8 | Council flips and sentiment jumps against the previous cycle | `8f14188` |
| R-9 | Signals ranked by history-calibrated confidence | `5efb406` |
| R-10 | Public calibration scorecard, aggregates only | `cf56051` |
| R-11 | Free Shock Watch for out-of-cycle market shocks | `c00027e` |
| R-12 | Walk-forward model scorecard, rejected candidates included | `bf42370` |
| R-13 | Decision receipts: an audit trail for every prediction | `e6647fe` |
| R-14 | Alpha Fusion Score from the free alpha sources | `02ef47f` |
| R-15 | Real portfolio from positions: weights, P/L, VaR; phase 2 sizes against existing exposure (`83a5e56`) | `ff96797` |
| R-16 | Commit-reveal attestation of predictions (see D-4) | `7c37fe1` |
| R-17 | Sunday narrative recap from the Slow Loop | `64c7eec` |
| R-18 | Model move against the options-implied move (edge) | `1a9500f` |

Also in this release, without a roadmap number:

- 2026-06-21: `gh-secrets-check` and `gh-secrets-sync` helper scripts (`5e99e8c`).
- 2026-06-23 to 2026-06-24: SonarCloud wired into CI and its first findings fixed (`c1ff822`,
  `475a676`, `998a06a`, `6fb7956`).
- The cycle's own bill reported in the digest (`bc7fcf5`) and the BTC and ETH models retrained
  with their sideways band scored correctly (`4785eab`).
- Phase 0 activation, `5c4f01b`: the cron and fourteen of the shipped features switched on.

### Divergences

- The roadmap was integrated in three merges: `c39dc25` ("ten roadmap items"), `f26c788` ("five
  roadmap items") and `fd7110c` ("close the roadmap"). Their titles do not add up to eighteen
  the way they read: `c39dc25` names ten items but contains eleven, because R-3 (the persona
  leaderboard, `31e3e76`) is not in its title; `f26c788` holds five (R-14 to R-18); `fd7110c`
  holds the last two, R-2 and R-10, so the order of delivery is not the order of the numbers.
- Merge commits repeat the title of the feature commit they merge, sometimes reworded: `3f19d5b`
  merges `31e3e76` under the same subject, and later in the day `ab46d27`/`bd92e2e`,
  `046fa09`/`417ffc0`, `59cc4f3`/`c7986d6` and `1750245`/`a7440cb` do the same. The table names
  the original, as the decision log does.
- R-15 shipped in two phases: the weights, P/L and VaR from real positions (`ff96797`) and, in
  the closing merge, the position bands that see existing exposure (`83a5e56`).
- The commit titles of the series carry the item number in a trailing parenthesis, equal to the N
  of R-N; the June series used the same bare numbers for other things, which is why the nine
  council and AI upgrades got R-19 to R-27 in the decision log.
- R-10 and R-16 shipped as code, and the flags that were meant to publish them shipped on while
  nothing deployed or pushed the result; `562fe73` (2026-10-16) switched them off and corrected
  the claims. R-18 compares against option-chain data that the free Finnhub tier answers with 403,
  so `options_flow_enabled` stays off (see its comment in `config.toml`).
- Eight more commits of the same day (four of them merges) come after the tag target and are
  listed under Unreleased.

## [v0.2] - 2026-06-21

Tag target: `9d98e16` (37 commits). A single commit, "Ship: council/AI upgrades".

### Shipped

| ID | Item | Commit |
|---|---|---|
| R-19 | Debate mode: one revision round when round 1 of the council is split | `9d98e16` |
| R-20 | Confidence-gated model routing: cheap model first, frontier on low confidence | `9d98e16` |
| R-21 | Adaptive persona weighting by historical hit rate | `9d98e16` |
| R-22 | LLM-as-judge calibration of the stated confidence (Slow Loop) | `9d98e16` |
| R-23 | Cross-symbol contagion: this cycle's verdicts of sector peers in the prompt | `9d98e16` |
| R-24 | Tool-use research agent for the largest moves | `9d98e16` |
| R-25 | Regime-aware thresholds: RISK_OFF tightens the volatility gate | `9d98e16` |

Also in the same commit, as its message lists them:

- explainability and UX: SHAP-lite feature attribution, per-recipient subscriptions, a static web
  digest, a council-vote drill-down;
- a risk and track-record suite: portfolio VaR/CVaR, a beta-to-portfolio stress test,
  hedge effectiveness, a paper-PnL equity curve, a calibration reliability diagram, a lessons
  panel and an offline walk-forward backtest harness;
- seven optional alpha-data sources (EDGAR insider flow, Alpha Vantage earnings, Finnhub options
  and analyst consensus, Reddit social velocity, the FRED yield curve, a regime-tagged RAG), each a
  port, an adapter and a Null fallback.

Needs migrations 012 to 017; 012 is breaking (see v0.1).

### Divergences

- One commit carries seven numbered upgrades, the three feature groups above and seven data
  sources, so none of it has a history of its own. R-26 and R-27, the other two of the nine June
  upgrades, are one commit earlier and therefore in v0.1.
- The message says everything is flag-gated off by default with zero new paid calls, and quotes
  "1498 unit + 25 container tests"; both are the message's own claims, not re-measured here.
- The static web digest announced here was removed on 2026-10-24: the email carries the owner's
  portfolio and third-party data, so D-12 replaced it with a private archive.

## [v0.1] - 2026-06-21

Tag target: `3e6a493` (36 commits), the parent of the commit that is v0.2. The first working
agent and the audit fixes that made it honest, from 2026-05-13.

### Shipped

| ID | Item | Commit |
|---|---|---|
| R-26 | Counterfactual dissent replay in the self-reflection diagnosis | `ef921b9` |
| R-27 | Outcome-aware rerank of the RAG analogues | `ef921b9` |

- 2026-05-13, `3e4a37f`: the first code: the Fast Loop and Slow Loop workflows, a LangGraph agent,
  the XGBoost predictor, the Supabase repository, the Finnhub and Alpha Vantage adapters, the
  Resend notifier and the first migration, with tests.
- 2026-05-15 and 2026-05-19: the investor advisory board, the council (`a4a52ae`), moved into
  separate persona files (`8723d37`).
- 2026-06-01: BTC and ETH through CoinGecko (`c51954a`).
- 2026-06-03: configuration moved from variables to `config.toml` (`1a68875`).
- 2026-06-20, `bae5111`: the P0 and P1 audit fixes: a 12-hour return as the ML target with an
  honest baseline, a hardened FinOps gate, idempotent upserts, paginated PostgREST reads and
  prompt-injection fencing (migrations 012 and 013).
- 2026-06-21: hardened ML evaluation with walk-forward cross-validation (`cac2bc9`), the digest
  fan-out to Telegram and Slack with push alerts for critical quota events (`5f4a638`), and
  drawdown flags, a correlation radar and Kelly-lite position bands (`3e6a493`).

### Divergences

- `205c834`, "Initial commit", holds only a `.gitignore`; the code starts with the next commit,
  `3e4a37f`.
- Fifteen of the 36 commits are titled with little more than "Bugfix", "Bug fix", "Small fix" or
  "Crypto fix", so their scope is only in the diff. The first commit with a described scope is
  `bae5111`.
- Migration 012 changes the training target from an absolute price to a 12-hour return and is
  breaking: weights trained before it must be deleted and retrained.
- `3e6a493` is a boundary chosen as the last commit before `9d98e16`, not a milestone commit.
  R-26 and R-27 (`ef921b9`) belong to the June series that the v0.2 commit completes.

[Unreleased]: https://github.com/DominikSienkiewicz/StockAgent/compare/v0.3...main
[v0.3]: https://github.com/DominikSienkiewicz/StockAgent/compare/v0.2...v0.3
[v0.2]: https://github.com/DominikSienkiewicz/StockAgent/compare/v0.1...v0.2
[v0.1]: https://github.com/DominikSienkiewicz/StockAgent/releases/tag/v0.1
