# An agent that asks before it pays

> **DRAFT. Not for publication.** This text lives in the repository so the owner can read it
> first; it goes to an external blog only after that reading. Two numbers are placeholders
> marked `[PROD-1]` and `[PROD-2]`: they come from the production database and the SQL that
> produces them is in [the placeholders](#the-two-numbers-that-come-from-production). Every
> other number is measured from the repository by a command listed in
> [How to reproduce the numbers](#how-to-reproduce-the-numbers), at commit `cc6db87`. It makes
> no claim about forecast accuracy.

StockAgent is a loop that wakes up on weekdays, looks at the price of a watchlist of 43 stocks and ETFs
plus BTC and ETH, and decides, symbol by symbol, whether a move is worth paying to analyse.
The analysis is the expensive part: a language model, a council of seven investor personas,
news and sentiment from a rate-limited API, embeddings for the memory of past calls. This post
is about the one rule that keeps that bill under control, why it lives in the domain and not
in the graph that runs the loop, and about three bugs that taught the repository more than
the rule did.

## The rule

Every paid call sits behind one question: did the price move enough? The answer is
`Asset.evaluate_volatility` and the threshold composition `effective_threshold` in
[`src/domain/asset.py`](../../src/domain/asset.py). The thresholds are plain configuration
in [`config.toml`](../../config.toml): `volatility_threshold = "0.02"` (2%) for the main
analysis, `council_volatility_threshold = "0.03"` for the council and
`crypto_volatility_threshold = "0.05"` for crypto, which moves more on an ordinary day. Two
multipliers (a risk-off market regime and an imminent earnings report) can raise a threshold
and are guaranteed to be at least 1.0, so they can only tighten the gate, never loosen it.

Three decisions make it a rule rather than a habit:

- **It is in the domain.** The domain package imports nothing outside the standard library
  (`domain_stdlib_only` in the repo stats below is checked by an AST walk, not by trust). The
  graph that orchestrates the cycle calls the rule and executes its answer; it does not
  reimplement it. The composition of the threshold for every gate (analysis, council, tool use
  and the diagnosis of a wrong call) was later moved into the same function.
- **The exceptions are explicit and have a budget.** The weekly recap and the calibration
  judge run in the Slow Loop, outside any price move, so they get their own ceiling:
  `slow_loop_max_paid_llm_calls_per_week = 15`. A per-cycle ceiling,
  `max_paid_llm_calls_per_cycle = 100`, cuts the council of the symbols that come after it.
- **A test says "nothing paid was touched".**
  `TestVolatilityGateIsSystemic::test_no_optional_paid_port_touched_when_gated` runs a gated
  symbol through the graph and fails if any optional paid port was called.

The design has a price and the [decision log](../decisions.md) states it: a quiet market
yields no analysis, no prediction and no settlement for most symbols on most days, so the
track record grows slowly.

## What the rule saves

The repository can estimate the ceiling itself. `main_tools.py cost-envelope` multiplies the
calls a cycle can make by a unit cost per source (an order-of-magnitude model, not an invoice).
With all 42 priced symbols past the gate, seven personas and no debate, it counts 588 calls and
at most 1.7237 USD for one cycle; with every symbol below the gate it counts zero. The gate is
the difference between those two numbers. How often production actually lands near the first
one is a question for the ledger:

- Share of Fast Loop cycles that made no paid call at all: **`[PROD-1]`**.
- Cost per saved prediction, from the cost ledger: **`[PROD-2]`**.

## Three bugs and the tests that closed them

The [engineering log](../engineering-log.md) lists 13 incidents. Each row has the rule it
taught and the test that fails when the rule is broken again, and a test checks that every
cited commit and test exists. Three of them show the pattern.

**A factor of 100 in the lead.** The first line of the email, which is also its subject,
formatted a fraction as a percent, so a 4.2% drop on NVDA read `NVDA -0.0%` (commit
`627e106`, 2026-10-16). The rule is that a relative change carries its unit in its type:
`Ratio` is a fraction, `Percent` is a percent, and the gate compares fractions. The guard
pairs a rendering test (`test_split_council_lead_renders_fractional_delta_as_percent`) with a
domain one (`test_gates_on_fraction_not_percent`), so the bug cannot return through either
end.

**A watch that looked healthy while blind.** The hourly Shock Watch compares prices with the
snapshots that the daily loop writes and never writes any itself. When the daily loop was
paused, its references aged past the staleness guard: every alert was suppressed, while the job
kept fetching prices and logged a calm "0 alerts" (commit `f886da0`, 2026-10-16). The age rule
now lives in the domain (`is_reference_stale`), the watch checks it before it fetches a price,
and when every reference is stale it logs a blind warning. Two tests pin both halves:
`test_stale_snapshot_skips_the_price_fetch` and `test_all_stale_snapshots_log_blind_warning`.

**A forecast that was only the current price.** With no trained model weights, the target of
the XGBoost leg was the current price, the council prompt called it the model's target, and
the email showed a `+0.00%` forecast (commit `eff54d7`, 2026-10-18). Now no weights means
the model leg is the baseline and is labelled cold-start in the report and in the prompt, and
the edge and options paths do not run on it. The tests are
`test_cold_start_is_flagged_in_final_state` and `test_reports_cold_start_when_model_untrained`.

In each case the fix and its guard test are in the same commit (`627e106`, `f886da0`,
`eff54d7`), and the log row links the pair. What the repository adopted is a habit more than
a tool: a fix that is only a fix gets undone by the next refactor, a fix that is also a test
does not.

## The two numbers that come from production

The loop records one row per run in `cycle_runs` (migration `041`) and one row per model
request in `llm_calls` (migration `040`), with the cost computed from the tokens in the
response. Both are private to the database owner, so the two numbers below are left for the
owner to fill from these queries before anything is published. Choose the window first and
state it in the post, with the date the queries were run.

`[PROD-1]` — the share of Fast Loop cycles with no paid call. A cycle counts as free when the
`paid_calls` object of its `cycle_runs` row sums to zero; a rerun of the same workflow run
(`run_attempt` above 1) is its own row.

```sql
select
  count(*)                                                       as fast_cycles,
  count(*) filter (where paid_total = 0)                         as free_cycles,
  round(100.0 * count(*) filter (where paid_total = 0)
        / nullif(count(*), 0), 1)                                as free_pct
from (
  select r.id,
         coalesce((select sum(value::numeric)
                   from jsonb_each_text(r.paid_calls)), 0)       as paid_total
  from cycle_runs r
  where r.loop = 'fast'
    and r.started_at >= timestamptz '<FROM>'
    and r.started_at <  timestamptz '<TO>'
) cycles;
```

`[PROD-2]` — cost per saved prediction. The ledger has no `run_attempt`, so runs are summed by
`run_id` on both sides. `cost_usd` is null for a model that is not in the price list; the
query reports how many calls are unpriced, because a nonzero count makes the cost a lower
bound.

```sql
with fast_runs as (
  select run_id, sum(saved) as saved
  from cycle_runs
  where loop = 'fast'
    and started_at >= timestamptz '<FROM>'
    and started_at <  timestamptz '<TO>'
  group by run_id
),
ledger as (
  select run_id,
         sum(cost_usd)                           as cost_usd,
         count(*) filter (where cost_usd is null) as unpriced_calls
  from llm_calls
  where run_id in (select run_id from fast_runs)
  group by run_id
)
select sum(f.saved)                                                  as predictions,
       round(coalesce(sum(l.cost_usd), 0)::numeric, 4)               as cost_usd,
       round((coalesce(sum(l.cost_usd), 0)
              / nullif(sum(f.saved), 0))::numeric, 4)                as usd_per_prediction,
       coalesce(sum(l.unpriced_calls), 0)                            as unpriced_calls
from fast_runs f
left join ledger l using (run_id);
```

## How to reproduce the numbers

Everything below runs from a clean checkout at commit `cc6db87`; none of it needs a key or a
network.

| Number | Command | Result at `cc6db87` |
|---|---|---|
| Commits | `git rev-list --count cc6db87` | 801 |
| First commit | `git log --reverse --format=%ad --date=short cc6db87 \| head -1` | 2026-05-13 |
| Domain modules, adapter modules, ports, migrations, test functions, domain stdlib-only | `uv run python main_tools.py repo-stats --out /tmp/repo-stats.json` | 85, 52, 40, 72, 6620, `true` |
| Watchlist size | `python -c "import tomllib; c = tomllib.load(open('config.toml','rb')); print(len(c['symbols']), len(c['crypto_symbols']))"` | 43, 2 |
| Gate thresholds | `grep -n "volatility_threshold" config.toml` | 2%, council 3%, crypto 5%, tool use 5% |
| Calls and USD of one cycle, minimum and maximum | `SUPABASE_URL=https://x.supabase.co SUPABASE_KEY=sb_secret_x uv run python main_tools.py cost-envelope` | 0 and 588 calls, 0 and 1.7237 USD |
| Incidents in the engineering log | `grep '^\| ' docs/engineering-log.md \| grep -vc '^\| Incident'` | 13 |

The two dummy Supabase values only satisfy the settings validator; the `cost-envelope`
command reads `config.toml` and the price list and opens no connection.

## Before this goes anywhere

- [ ] Fill `[PROD-1]` and `[PROD-2]` from the queries above, state the window and the date, and
  remove the placeholders and the draft banner.
- [ ] Run the repository commands again at the commit that is current on the day of
  publication and update the table.
- [ ] Read the three bug stories once more as someone who does not know the project: the
  weakness they admit is deliberate, and the text should not promise more than the log shows.
- [ ] Decide the title. This one says what the loop does; it makes no claim about money earned,
  because the agent earns none.
