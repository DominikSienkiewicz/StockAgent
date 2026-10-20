# Reference signal protocol v1 (`reference-v1`, WIA-24)

This file is the pre-registered rule of the free reference signal. It is frozen:
its sha256 is pinned as `REFERENCE_PROTOCOL_SHA256` in
`src/domain/reference_signal.py`, and a test fails when the two disagree. Before
the first reference run, v1 may still be amended (file and constant change in
the same commit). After it, any change to the rule, even a single word, is a new
version: a new file `docs/reference-protocol-v2.md`, a new constant and the label
`reference-v2`, so calls made under different rules are never pooled.

## Purpose

A deterministic rule that costs nothing (no LLM, no Alpha Vantage, no
embeddings) and produces a live, checkable track record while the paid loop is
paused. It is the zero-cost benchmark the paid LLM has to beat.

## Rule

1. **Cadence.** One evaluation per Slow Loop run (`main_trainer.py`, weekly cron
   on Sunday 03:00 UTC; a manual dispatch is a run too).
2. **Universe.** Every symbol the Slow Loop trains on (`symbols` plus
   `crypto_symbols`), minus `symbols_unsupported_price`.
3. **Read.** One price per symbol per run, taken by the reference signal itself
   from the free quote: Finnhub `/quote` field `c` for stocks and ETFs,
   CoinGecko `/simple/price` in USD for crypto. The read time is the time of
   the read in UTC. A read is never written to `price_snapshots`, so the Fast
   Loop reference prices stay untouched.
4. **Baseline.** The most recent earlier read of the same symbol that is at
   least 6 days and at most 8 days older than the current read. Without such a
   read the rule abstains for that symbol in that run.
5. **Call.** `BULLISH` when the current price is above the baseline price,
   `BEARISH` when it is below. An unchanged price, or a price that is not
   positive, abstains. The rule never calls `SIDEWAYS`.
6. **Horizon.** Until the next run. A call is settled by the first later read of
   the same symbol taken at least 6 days and at most 8 days after the call. A
   read taken less than 6 days after the call leaves it open. A call with no
   settling read within 8 days expires: it is reported as expired and left out
   of the hit rate.
7. **Settlement.** A `BULLISH` call is a hit when the settling price is above
   the price at the call, a `BEARISH` call when it is below; an unchanged price
   is a miss. This is the directional rule of the LLM track record
   (`Prediction.is_trend_correct`).
8. **First run.** With no earlier read the rule abstains for every symbol and
   only stores the read, which becomes the baseline of the following week.
9. **Label.** Every call carries `recipe_version = 'reference-v1'`, so it is
   never mistaken for an LLM prediction.

## What the rule is not

The rule is a benchmark, not a recommendation. A weekly cadence gives roughly
one call per symbol per week, so the Wilson intervals of its hit rate stay wide
for the first months.
