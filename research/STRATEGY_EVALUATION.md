# Are any of these strategies actually good?

Thirty strategies, twelve instruments, 348 strategy-symbol pairs. Daily bars
2010–2026, split at **2020-01-01** — everything before is in-sample, everything
after is out-of-sample. The split date was fixed before any result was looked
at.

Reproduce with:

```bash
.venv/Scripts/python research/evaluate_all.py
```

## The answer

**Nothing works.** Zero of 348 pairs cleared the Bonferroni bar (|t| > 3.81).
Zero cleared Benjamini-Hochberg FDR at 5%. Only 7.5% beat buy-and-hold at all.

Every single strategy has a **negative** median excess return against simply
holding the same instrument.

## Three things that rule out the easy excuses

### It is not the trading costs

| Round-trip cost | Cleared the bar | Beat buy-and-hold |
|---|---|---|
| 0 bps | 0 of 348 | 10.3% |
| 2 bps | 0 of 348 | 10.8% |
| 5 bps | 0 of 348 | 10.3% |
| 10 bps | 0 of 348 | 10.0% |

At **zero cost** — free trading, perfect fills — still nothing. The rules
themselves do not predict.

### It is not an unfair benchmark

Buy-and-hold was excluded from the candidate set, because it is the benchmark.
Worth noting why: its measured "excess" is 0.0004%/yr, economically zero, but
the engine enters on the first bar's *open* rather than its close, leaving a
tiny constant offset with almost no variance — which produces t-stats near 2.7
on nothing at all. Left in, the benchmark would have been the top-ranked
"discovery" in the table.

### It is not that the rules are defensive and the period was a bull market

This is the objection that had to be tested, and testing it produced the
clearest result in the whole exercise:

| Symbol | Strategies beating buy-and-hold | Median excess | Buy-and-hold Sharpe |
|---|---|---|---|
| **TLT** | **86%** | **+2.16%** | **−0.19** |
| IWM | 3% | −10.62% | 0.51 |
| GLD | 0% | −9.93% | 0.95 |
| AAPL | 0% | −15.51% | 0.88 |
| SPY | 0% | −10.07% | 0.82 |
| NVDA | 0% | −37.20% | 1.31 |
| *(and six more, all 0%)* | | | |

TLT is the one instrument that **lost money** over the test period. It is also
the only one where the strategies win — and they win on 86% of attempts.

That is the whole story. These rules are out of the market part of the time.
When the instrument falls, being out helps. When it rises, being out hurts.
They are not predicting anything; they are **diluting exposure**, and dilution
looks like skill only when you test it on something that went down.

And even on TLT, none of it is significant: the best result there is
`Gap fade` at +4.30%/yr with **t = 0.57, p = 0.57**.

## A trap worth knowing about

Ranking by **Sharpe ratio** instead of return appears to rescue several
strategies. `Day of week` — a deliberate null that holds on one weekday and
nothing else — posts the best risk-adjusted score of any strategy tested, with
a Sharpe gap of **+0.47** and positive on 75% of symbols.

It is also negative on excess return for **all twelve**, with not one
significant t-statistic.

Both facts have the same cause: it holds a position 19% of the time. Sitting in
cash cuts volatility faster than it cuts return, so Sharpe rises while the
money falls. If a calendar artefact out-scores every real strategy on your
chosen metric, the metric is measuring exposure, not skill — and the ranking
among the real strategies is noise.

## Full results

| strategy | med OOS | Sharpe gap | exposure | trades |
|---|---|---|---|---|
| Faber TAA | −4.92% | −0.01 | 0.73 | 54 |
| Price vs moving average | −5.01% | +0.02 | 0.72 | 64 |
| Golden cross | −5.70% | −0.12 | 0.73 | 10 |
| Time-series momentum | −6.04% | −0.14 | 0.76 | 40 |
| Absolute momentum + trend filter | −7.16% | −0.07 | 0.65 | 58 |
| Buy the dip | −7.32% | −0.11 | 0.38 | 92 |
| SMA crossover | −7.67% | −0.20 | 0.64 | 45 |
| Day of week *(null control)* | −7.71% | **+0.47** | 0.19 | 784 |
| Donchian breakout | −9.20% | −0.08 | 0.49 | 70 |
| Turtle breakout | −9.20% | −0.08 | 0.49 | 70 |
| Regime-filtered trend | −9.32% | −0.10 | 0.55 | 59 |
| MACD trend | −9.75% | −0.24 | 0.51 | 168 |
| Sell in May | −10.46% | −0.37 | 0.49 | 16 |
| Ensemble vote | −10.47% | −0.24 | 0.57 | 206 |
| Volatility-targeted trend | −10.59% | +0.01 | 0.52 | 39 |
| Bollinger reversion | −11.21% | −0.36 | 0.20 | 72 |
| Volatility breakout | −11.27% | −0.28 | 0.48 | 128 |
| Stochastic reversion | −11.68% | −0.33 | 0.42 | 78 |
| VWAP reversion | −12.92% | −0.44 | 0.17 | 67 |
| Turn of month | −13.41% | −0.32 | 0.29 | 200 |
| Z-score mean reversion | −13.85% | −0.40 | 0.15 | 76 |
| ADX trend | −13.97% | −0.38 | 0.21 | 62 |
| Trend strength gated | −14.24% | −0.34 | 0.24 | 52 |
| Keltner breakout | −14.80% | −0.34 | 0.29 | 63 |
| Walk-forward logistic | −14.84% | −0.56 | 0.18 | 288 |
| 52-week high | −15.48% | −0.53 | 0.21 | 144 |
| RSI mean reversion | −16.30% | −0.54 | 0.13 | 20 |
| Gap fade | −17.57% | −0.76 | 0.15 | 540 |
| Adaptive channel | −20.13% | −0.96 | 0.07 | 198 |

**Complexity did not help.** The most elaborate entries — a walk-forward
logistic regression refitted on expanding windows, a five-rule ensemble, an
adaptive channel, a regime filter — sit at the bottom, not the top. The
simplest defensive rules (Faber TAA, price versus its moving average) lose the
least, and they lose the least because they are invested the most.

## What this does not prove

- **A bull market is a hard test.** 2020–2026 rose sharply for eleven of twelve
  instruments. These rules could be genuinely useful in a decade-long bear
  market and this test could not tell you.
- **Default parameters only.** Each strategy ran at its stated defaults. A
  parameter search would find better in-sample numbers on every one of them,
  which is precisely why it was not run — with 348 tests already, adding a grid
  would raise the bar faster than it raised the results.
- **Long-only, single-instrument.** The cross-sectional strategies in
  `strategies/cross_sectional.py` are ranked separately by `panel_backtest.py`;
  they are not in this table.

## What it does support

The one conclusion this cannot dodge: **for a single instrument on daily bars,
none of thirty well-known rules beat holding it, and the ones that look good on
risk-adjusted terms are just holding less.** That is consistent with sources 06,
07 and 08 — the three independent sources that all landed on low-cost indexing —
and with the SPIVA figure that 85.6% of professional US large-cap funds trail
the index over ten years.
