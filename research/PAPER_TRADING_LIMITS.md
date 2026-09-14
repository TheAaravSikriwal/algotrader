# What paper trading here can and cannot tell you

Measured on the Alpaca paper account, 2026-09-14, during market hours.
Reproduce with `research/plumbing_test.py` and `research/paper_fill_realism.py`.

## The short version

The plan was to paper trade the limit-order strategy to answer one question:
**do limit orders fill in life the way the backtest assumes?** Every result in
this repo that involves a resting limit rests on that assumption.

**Paper trading on Alpaca cannot answer it.** The simulator uses the same fill
rule the backtest does, so running the experiment would confirm the assumption
by construction rather than test it.

## The evidence

### The paper engine fills a limit as soon as price reaches it

Three probes, most passive to most aggressive, on SPY:

| Probe | Limit placed | Result |
|---|---|---|
| Behind the bid | bid − $0.05 | **not filled** in 25s (bid never came down) |
| **At the bid** | bid exactly | **filled in 2 seconds** |
| Through the ask | ask + $0.05 | filled in 2 seconds (control) |

Joining the bid on a real venue puts your order at the **back of the queue** at
that price. You fill only after everyone already resting there, and only if
someone sells into it — which is disproportionately when the seller is right
and you are wrong. That adverse selection is the entire thing the experiment
was meant to measure.

Filling in two seconds means the simulator is applying roughly "price reached
the level, so you are filled". That is the backtest's rule.

### The round trip cost a fifteenth of the spread

| | |
|---|---|
| Bought (marketable limit) | 763.10 |
| Sold (market) | 763.09 |
| Round-trip cost | **0.13 bps** |
| Quoted spread at the time | **2.0 bps** |

A round trip that crosses the spread twice should cost roughly the spread.
Paying 0.13 bps against a 2.0 bps quote means the paper engine is not charging
realistic spread either, so slippage measured here is also optimistic.

### Market data is fifteen minutes behind

Alpaca's free plan delays the consolidated (SIP) feed by exactly 15 minutes,
sampled three times twenty seconds apart:

```
12:49:20  newest 1-min bar 12:34  age 15.3 min
12:49:41  newest 1-min bar 12:34  age 15.7 min
12:50:01  newest 1-min bar 12:35  age 15.0 min
```

The real-time IEX feed is current to about a minute but carries a median
**1,779 shares a minute against SIP's 42,986** — roughly 4% of the volume.
The fair-value-gap signal is defined by bar highs and lows, and on 4% of
volume those are not the market's highs and lows. So an IEX signal is a
*different* signal from the one that was backtested, not a faster version of
the same one.

`core/daytrader.py` now refuses to act on bars older than six minutes rather
than quietly trading a stale price.

## What paper trading here *can* tell you

All of this was verified today and all of it is worth having
(`research/plumbing_test.py`, 12 checks, all passing):

- Bracketed limit orders submit, rest at the venue, and cancel cleanly.
- The venue rejects a stop on the wrong side of the entry, and a target on the
  wrong side — checked before submission rather than read off a 422.
- A fill opens a position and its stop and target attach **at the venue**, so
  they survive the app being closed.
- Flatten closes the position and cancels the working exits.
- The fill log records the fill and computes slippage from prices, signed so
  positive always means a cost.
- Session handling, the flatten deadline, half-days, and the daily-loss rail
  all behave.

That is the plumbing, and it works. It is not the edge.

## What would actually answer the fill question

1. **Real money, very small.** One share at a time, real venue, real queue.
   The only faithful test, and the honest reason to consider it is information
   rather than profit.
2. **A broker whose simulator models queue position.** Some do; Alpaca's does
   not appear to.
3. **Reconstruct it offline from historical quote data.** Take the level, the
   time, and the depth at that price, and estimate how much volume had to
   trade before your order would have been reached. Needs Level 2 history,
   which is not free, but it needs no capital and no waiting.

Until one of those happens, treat every limit-based backtest number in this
repo as an **upper bound**, and the paper fill rate as unable to correct it.
