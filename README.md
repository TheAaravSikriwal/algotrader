# algotrader

A backtesting and paper-trading workbench for US equities, built around Alpaca.
Pick a symbol, pick a rule, see what it would have done — with a dashboard, a
CLI, and an engine that structurally cannot cheat.

I built it to find out whether I could day-trade profitably with an algorithm.
The answer turned out to be no, and most of the work in here is the effort of
establishing that properly instead of guessing.

## What it found

I implemented 33 strategies — moving-average crossovers, RSI reversion,
breakouts, some published academic rules, and a handful pulled out of trading
books and forums. All of them got tested on daily bars and 5-minute bars,
across a basket of liquid ETFs and large caps, with costs on both sides of
every trade.

Nothing worked. Not "nothing worked well" — nothing cleared the bar at all.

Of 348 strategy-symbol pairs on daily data, **zero** passed a
Bonferroni-corrected significance test and **zero** passed Benjamini-Hochberg
FDR at 5%. Only 7.5% beat simply buying and holding, which is about what you'd
get from coin flips. I re-ran the whole sweep at 0, 2, 5 and 10 basis points of
cost in case I'd been unfair on fees. Same answer every time, so the problem
wasn't my cost model. Intraday was no better: 0 of 70 pairs on SIP data, 0 of
118 on IEX.

### The thing that nearly fooled me

Early on a lot of strategies looked like they were beating buy-and-hold. 86% of
them beat it on TLT.

TLT was the one instrument in the basket that went *down* over the test period.
On the eleven that went up, the strategies won 0% of the time.

So they weren't timing anything. They were just out of the market some of the
time, and being out of the market is wonderful when the market falls. That's
why there are deliberately meaningless strategies in here — "Day of week",
"Turn of month" — running alongside the real ones as controls. The nonsense
ones kept scoring better than the real ones. If your fake strategy beats your
real strategy, your real strategy is noise.

### The one that survived, and how it died

One rule got further than the rest: a limit order resting at the midpoint of a
three-bar fair value gap. The gross edge was real and strongly significant —
t of 5.89 over 880 trades in the first fifteen minutes of the session. I was
quite pleased with myself for about a day.

Then I ran the same rule at different times of day:

| Window | Trades | Gross (bps) | t | Spread | Net |
|---|---|---|---|---|---|
| 09:30–09:45 | 880 | 5.38 | 5.89 | 3.78 | **+1.60** |
| 09:30–10:30 | 12,029 | 3.41 | 6.19 | 2.34 | **+1.07** |
| Full session | 140,879 | 0.91 | 5.80 | 2.10 | **−1.19** |
| 10:30–15:30 | 121,631 | 0.67 | 3.85 | 1.59 | **−0.92** |

The edge lives almost entirely in the opening minutes, which is also exactly
where the spread is widest. Move to a quieter window where trading is cheaper
and the edge vanishes along with the cost.

The edge *is* the spread. I was getting paid for providing liquidity roughly
what providing liquidity is worth — which, said out loud, is what an efficient
market is supposed to do. Even the two positive rows clear a 90th-percentile
opening spread of 10.6 bps by nothing; at that percentile the best variant nets
about −5 bps.

The rule is still here (`core/daytrade.py`) and the paper trader still runs it,
but not because I think it makes money. It's fully specified and
non-discretionary, which makes it a good vehicle for testing whether the order
plumbing works.

### Paper trading doesn't prove what I assumed it proved

Worth saying plainly. Alpaca's paper simulator fills orders using something
very close to the backtest's own fill rule. I put a limit at the bid and it
filled in two seconds, round-trip cost 0.13 bps against a quoted spread of
2.0 bps. No real venue gives you that. Real fills are fewer and adversely
selected — you get filled when the other side wanted you to be.

So paper trading here validates the plumbing: orders go out correctly, brackets
attach, positions close when they should, the rails fire. It cannot validate
profitability. Longer writeup in `research/PAPER_TRADING_LIMITS.md`.

---

Nothing in this repo places a live order. There's a hard guard against
non-paper accounts with no override flag, and the live API credentials are
separate environment variables I've deliberately left empty.

## Setup

```bash
pip install -r requirements.txt
```

Then run the dashboard:

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. The default data source is **yfinance**, which
needs no account — you can start immediately. Switch the source to **alpaca** in
the sidebar once you have keys:

```bash
cp .env.example .env   # then paste your paper keys in
```

Free paper keys: <https://app.alpaca.markets> → *Paper Trading* → *API Keys*.

## What's in the box

| File | What it does |
|---|---|
| `app.py` | The Streamlit dashboard |
| `cli.py` | Same engine, no browser — plus parameter sweeps |
| `core/engine.py` | The backtest simulator |
| `core/data.py` | Bar loading + on-disk cache (yfinance / Alpaca) |
| `core/metrics.py` | Sharpe, Sortino, drawdown, win rate, profit factor… |
| `core/indicators.py` | SMA, EMA, RSI, ATR, Bollinger, Donchian, MACD |
| `core/charts.py` | Plotly figures |
| `strategies/builtin.py` | Reference rules — **your templates** (33 registered in total) |
| `run_live.py` | The paper/live trading loop |
| `core/trader.py` | Reconciliation, risk rails, kill switch |
| `core/broker.py` | The `Broker` interface every venue implements |
| `brokers/alpaca.py` | Alpaca adapter (paper and live) |
| `brokers/fake.py` | In-memory broker for testing the loop offline |
| `replay.py` | Replay history through the live loop at full speed |
| `core/daytrade.py` | The intraday fair-value-gap rule, and what it really scores |
| `core/intraday.py` | 5-minute backtester — flat by the close, window-confined |
| `core/daytrader.py` | Live day-trading loop, risk rails, paper-only guard |
| `core/marketclock.py` | Exchange calendar, half-days, flatten deadlines |
| `core/fills.py` | Realised slippage against the quoted spread |
| `core/expectancy.py` | Kelly, break-even win rate, risk of ruin |
| `autorun.py` | Background loop so the end-of-day flatten actually happens |
| `core/heartbeat.py` | Whether a background loop is genuinely alive |
| `reset.py` | Put the paper account back to a clean state |
| `brokers/replay.py` | Historical broker with a strictly enforced clock |
| `tournament.py` | Walk-forward leaderboard across strategies and symbols |
| `core/tournament.py` | Walk-forward folds, parameter search, ranking filters |
| `core/bundle.py` | Export/import a run (strategy + settings + results) |
| `test_engine.py` | Correctness checks for the engine |
| `test_trader.py` | Correctness checks for the trading loop |
| `test_bundle.py` | Export/import round-trip checks |
| `news_backtest.py` | News pipeline: fetch, score, analyse, backtest |
| `core/news.py` | Alpaca news ingestion + point-in-time session mapping |
| `core/sentiment.py` | Lexicon and FinBERT scorers, event tagging |
| `core/newsfeatures.py` | Scored articles to bar-aligned features |
| `strategies/news.py` | News sentiment, drift and event-drift strategies |
| `test_tournament.py` | Walk-forward and leaderboard checks |
| `test_news.py` | News alignment, scoring and signal-detection checks |
| `test_replay.py` | Replay clock, fills and backtest-agreement checks |

## The rule that makes results trustworthy

A signal computed from bar *t* is filled at the **open of bar t+1**.

This is the single most common way backtests lie to people: they compute a signal
from a day's closing price and then buy at that same closing price, which is a
trade you could not have made. The engine here structurally cannot do that — the
loop sets `desired` only at the *end* of each bar's processing.

Costs are modelled too. Slippage defaults to 5bps per side; Alpaca charges no
commission on US equities, but you still cross the spread on every fill.

## Writing your own strategy

Copy any class in `strategies/builtin.py`. Two things matter:

```python
@register
class MyRule(Strategy):
    name = "My rule"
    description = "Shown under the strategy picker."

    # every Param becomes a slider/checkbox in the UI automatically
    params = [
        Param("lookback", 20, "Lookback (bars)", "int", 5, 200, 1),
        Param("threshold", 1.5, "Threshold", "float", 0.1, 5.0, 0.1),
    ]

    def generate_signals(self, df):
        # df has columns: open, high, low, close, volume
        # return 1.0 = long, 0.0 = flat, -1.0 = short, per bar
        strength = df["close"] / df["close"].rolling(self.lookback).mean()
        return (strength > 1 + self.threshold / 100).astype(float)

    def indicators(self, df):        # optional — overlays on the price chart
        return {"Mean": df["close"].rolling(self.lookback).mean()}
```

Save the file. The dashboard picks it up on reload; no other file needs editing.

Only use data available *at or before* each bar's close. If you use
`.shift(-1)`, `.rolling(...).mean()` centred, or anything else that peeks
forward, the engine will happily report a fantasy.

## CLI

```bash
python cli.py --list
python cli.py --symbol AAPL --strategy "SMA crossover" --fast 20 --slow 50
python cli.py --symbol SPY --strategy "RSI mean reversion" --sweep period=5:30:5
python cli.py --symbol QQQ --strategy "Donchian breakout" --stop-loss 8 --save-trades out.csv
```

## Tests

```bash
python -m pytest
```

905 of them, across 36 files. They check the no-lookahead rule, that costs
actually cost money, that stops cap losses at the stop level, that realised
trade P&L reconciles with the equity curve, and a great deal about the live
loop — that it refuses a non-paper account, that it won't double an order while
the first is unfilled, that it flattens before the close.

The habit that made them worth anything: after fixing a bug, break it again on
purpose and confirm a test goes red. If it doesn't, the test is decorative.
That's caught more bad tests here than bad code.

## The tournament — finding which strategies actually work

```bash
python tournament.py --symbols SPY,QQQ,AAPL --top 5 --save runs/leaderboard.csv
```

Runs every registered strategy against every symbol and ranks them **out of
sample**. Parameters are optimised on a training window, then applied unchanged
to the window that follows, which the optimiser never saw. Only those held-out
segments are stitched into the equity curve that gets ranked.

This matters more than anything else in the project. Search a thousand
combinations over one stretch of history, keep the best, and you have not found
an edge — you have found whatever best fits that noise. It will look superb and
lose money.

Three columns exist to stop you trusting a fluke:

| Column | What a bad value means |
|---|---|
| `folds` | Fewer than 3 and there is not enough evidence to rank it at all |
| `fold_win_rate` | Below 0.5 means it made money in a minority of periods — one lucky window carried it |
| `param_stability` | Near 0 means the best parameters change every fold, so they are fitting noise, not signal |
| `params_at_edge` | Near 1 means the winning parameters are pinned to the edge of the search grid |

That last one catches something subtle. An RSI rule that optimises to
`oversold=50, exit_level=95` is not mean reversion any more — it is "be long
almost always." The optimiser discovered buy-and-hold was better and quietly
degenerated the strategy into an imitation of it. The Sharpe looks real; the
strategy name has become a lie. `leaders()` filters these out.

### What it says about the built-in strategies

Run over 8 years on SPY, QQQ and AAPL with 11 folds each, **buy-and-hold takes
the top three places.** None of the seven textbook rules beat it out of sample.

That is the expected result, and it is why the harness exists. Simple
price-only rules on liquid large caps are the most picked-over ground in
markets. If you want an edge, you need information the price does not already
contain.

## News-driven strategies

```bash
# no Alpaca keys yet? watch the whole pipeline run on synthetic news
python news_backtest.py --demo

# is there signal in this symbol's news, and at what horizon?
python news_backtest.py --symbols AAPL --decay

# sweep the holding period
python news_backtest.py --symbols AAPL,MSFT,NVDA --sweep-horizons

# walk-forward tournament over the news strategies
python news_backtest.py --symbols AAPL,MSFT --tournament
```

News comes from Alpaca (Benzinga-sourced, back to 2015, ~130 articles/day), so
news strategies are **backtestable** rather than an article of faith.

### Scoring is free

| Scorer | Needs | Notes |
|---|---|---|
| `lexicon` (default) | nothing | Loughran-McDonald style financial word lists, with negation handling. Instant and deterministic. |
| `finbert` | `transformers` + `torch` | A transformer trained on financial text. Better on context, far slower. |

General-purpose sentiment tools misread financial language — "liability",
"aggressive" and "cut" are neutral or positive in ordinary English and strongly
negative in filings — which is why both options here are finance-specific.

Scores are cached by content hash, so re-running a ten-year backtest costs
nothing after the first pass. That matters: scoring 2015→today is roughly
**520,000 articles**, which is why this runs locally instead of through a paid API.

### The point-in-time rule

A story printed at 18:40 ET could not have informed that day's close, so it is
assigned to the **next** session. Weekend news rolls to Monday. A session with
no bar (a holiday) rolls *forward* to the next bar, never backwards. On top of
that the engine still delays execution by one bar.

Get this wrong and you invent an edge from nothing, invisibly. `test_news.py`
pins every one of those boundaries.

### What the horizon sweep shows

Running the demo — synthetic news carrying a **genuinely real** 5-day signal:

| hold_bars | return | Sharpe | trades |
|---|---|---|---|
| 1 | −6.6% | −0.29 | 146 |
| 2 | −10.2% | −0.32 | 116 |
| 5 | **+22.0%** | **0.53** | 55 |
| 20 | +9.5% | 0.25 | 5 |

The signal is real at every horizon. At 1–2 day holds it still **loses money**,
because 146 round trips of spread costs more than the edge is worth. This is the
concrete version of the research finding that news signals are near-zero at one
day and peak at 3–10 — and the reason fast in-and-out news trading is the
version most likely to lose.

## Cross-sectional strategies

```bash
python panel_backtest.py --universe sectors --split 0.6
python panel_backtest.py --symbols AAPL,MSFT,NVDA,JNJ,KO,XOM,JPM,PG --top-n 3
```

The single-symbol engine answers *"should I be long SPY today?"*. The panel
engine answers *"of these 100 stocks, which 10 should I hold?"* — which is the
shape of the best-replicated findings in finance. Those cannot be expressed one
symbol at a time, because the signal **is** the comparison between names.

Five strategies ship: cross-sectional momentum, low volatility, cross-sectional
reversal, inverse volatility, and equal-weight (the benchmark).

### Two things that make these results honest

**Equal-weight is the benchmark, not SPY.** Beating SPY may only mean your
universe outperformed. Beating an equal-weighted basket of the *same universe*
is the only way to show the ranking added something.

**The `t vs EW` column.** A paired t-test on daily return differences against
the equal-weight basket. Without it, comparing Sharpe ratios by eye manufactures
findings — see below.

### What it found

Out-of-sample (2022–2026), cross-sectional momentum beat equal-weight in all
four universes tested, on both return and Sharpe. It looked like a result.

It isn't. The t-stats were **−0.01, +0.31, +0.59, +0.34** — nothing near the
|t| ≥ 2 threshold. The four universes are correlated views of one market period,
so "4 for 4" is closer to one observation than four.

The sectors run is the clearest lesson: momentum showed Sharpe 0.88 vs 0.72 and
a higher total return, with a t-stat of **−0.01**. Its daily return advantage
was exactly zero; the entire Sharpe gap came from lower volatility.

One caveat in the other direction: this test asks *"does it earn more per day?"*,
not *"is it better risk-adjusted?"*. Momentum did show consistently smaller
drawdowns. Testing whether that is significant needs a Sharpe-difference test,
which is not built yet.

### Two findings worth keeping

**Dual momentum (Antonacci GEM) fails its own thesis.** The absolute-momentum
filter exists to sidestep bear markets. Run over 2007–2015 it posted a
**−54.5% drawdown — worse than equal-weighting's −36.6%.** On a 2022–2026
window alone it looked strong (t = 1.62, the best figure this project ever
produced); extending the history through an actual crisis erased it. A
promising result on a window that excludes the event a strategy claims to
handle is not evidence.

**Two strategies are significantly *worse* than equal-weighting.** Pairs
trading (t = −2.94) and low volatility (t = −2.37) both clear |t| ≥ 2 in the
wrong direction. Pairs trading showed Sharpe 1.41 on a 4-year window and
returned **−10.6%** on the 11-year holdout — the high Sharpe was a
low-volatility artifact of a short sample.

Across 22 strategies, nothing has significantly beaten equal-weighting.

### From *151 Trading Strategies*

Two strategies from Kakushadze and Serur, implemented to their specifications:

**Residual momentum** (§3.7) ranks on the part of a return the market does *not*
explain. Plain momentum quietly favours high-beta names, so in a rising market
it is partly a leveraged index bet; regressing that out leaves stock-specific
momentum. The paper uses three Fama-French factors — this uses the
equal-weighted universe as a single market factor, which removes market beta
but not size or value tilts.

**Cluster mean-reversion** (§3.9) generalises pairs trading past two names:
demean the cluster's returns and hold each stock in proportion to how far it
strayed. Dollar-neutral by construction, and different from the long-only
top-N bucket implemented above.

### The `t vs 0` column, and why it exists

Comparing a **dollar-neutral** book to a long-only basket penalises it for the
market exposure it deliberately does not take. A market-neutral strategy can be
a perfectly good standalone return stream and lose that comparison every time.
So the report carries both: `t vs EW` (did the ranking add anything) and
`t vs 0` (is the return distinguishable from zero at all).

### What the two paper strategies did

Held out on the Dow universe, 2022–2026:

| Strategy | turnover | 5bps | 0bps | t vs 0 (0bps) |
|---|---|---|---|---|
| Cluster mean-reversion | 71×/yr | −5.9% | +11.1% | 0.61 |
| Residual momentum | 3.6×/yr | +20.3% | +21.3% | 1.11 |
| Pairs trading | 6.2×/yr | −4.4% | −3.0% | −0.27 |

Cluster mean-reversion is **cost-driven** — it breaks even around 3bps, which
retail execution does not reach. Residual momentum is not; its turnover is low
enough that costs barely matter, so the signal simply is not strong.

**Under the fair test, none produce a return distinguishable from zero even
frictionless.** The unfair benchmark was not hiding anything.

Meanwhile the long-only strategies all clear `t vs 0` — momentum 2.42, equal
weight 2.34, inverse volatility 2.23, low volatility 2.13 — and none clear
`t vs EW`. That is the equity risk premium showing up as significant, and no
skill on top of it.

### Survivorship

`Panel.survivorship_warning()` flags what it can detect, but it cannot fix the
core problem: a universe you picked today excludes everything that went bankrupt
or got delisted. The `megacap` universe is the worst offender — its equal-weight
basket returned 174% out-of-sample against SPY's 71%, because the *universe* was
chosen knowing who won. Momentum within a basket of known winners means little.

## The research journal — where testing accumulates

```bash
python research.py test --strategy "Faber TAA" --symbols SPY --months 10 \
    --hypothesis "A 10-month trend filter beats buy and hold on SPY"
python research.py summary
python research.py register --strategy "Faber TAA" --symbols SPY --months 10
python research.py score
```

Every backtest above was a one-off: run it, read it, forget it. That loses the
two things that matter more than any single result.

### How many times you looked

Test one strategy at the 5% level and |t| > 2 means something. Test forty and
**two will clear that bar on noise alone** — you have not found an edge, you
have found the best of forty coin flips. The journal counts distinct
hypotheses and raises the bar accordingly (Bonferroni, plus a Benjamini-Hochberg
FDR column for shortlisting).

Re-running an identical spec does not count as a new look, so the bar tracks
genuine searching rather than repetition.

### What it says about this project

Nine strategies, backfilled from the work above, all against the correct
benchmark:

| | |
|---|---|
| Tests run | 9 |
| Naive bar | \|t\| ≥ 1.96 |
| **Corrected bar** | **\|t\| ≥ 2.77** |
| Beat benchmark at corrected bar | **0** |

**All nine t-stats are negative.** Not one earned more per day than simply
holding. Three cleared the naive bar — MACD (p=0.007), RSI (p=0.013), pairs
trading (p=0.032) — and none survive correction.

### Forward tests — the one thing that cannot be gamed

Every backtest here runs on history that has already been picked over,
including by us. The only unsearched dataset is the future.

`research.py register` locks a specification and stamps it with today's date.
Scoring uses **only bars after that date**, so no amount of prior searching can
contaminate it — not optimisation, not survivorship, not hindsight in choosing
the universe. It is slow by construction, and it is the only test here immune
to everything the rest of this README warns about.

## Reading the output honestly

A few things worth internalising before you trust any number on the dashboard:

- **Beating buy-and-hold is the bar.** Every tile shows the baseline next to it.
  A strategy returning 40% over five years on SPY *lost* to doing nothing.
- **A sweep's best row is overfitted by construction.** You searched for it in
  that data. Re-test the winner on a window you did not search over.
- **Trade count matters.** Twelve trades tells you nothing, whatever the win rate.
- **The survivor-bias trap.** Backtesting today's index members over ten years
  quietly excludes everything that got delisted.
- **yfinance data is free, and priced accordingly.** It back-adjusts for splits
  and dividends and has occasional bad prints. Verify anything surprising against
  your broker's own bars before acting on it.

## Replay — testing the system, not just the strategy

```bash
python replay.py --symbols SPY --strategy "SMA crossover" --fast 20 --slow 50 --compare
```

A backtest tests the *strategy*: signals in, equity out. Replay tests the
**whole system** — it drives `core.trader.Trader`, the same object that talks to
Alpaca, against historical bars. Order submission, position reconciliation, the
working-order guard, buying-power checks and the risk rails all execute for
real. Eight years of SPY runs in about **two seconds** (~1,000 bars/sec).

This is the gap between "the backtest passed" and "paper trading works." Bugs
like sending a duplicate order while the first is still unfilled do not show up
in a backtest at all, because a backtest has no concept of an unfilled order.

The property that makes it worth anything is that `ReplayBroker.get_bars` **cannot
return a bar past the current one**, enforced in a single place and tested
directly. Orders fill at the next bar's open, matching the engine.

`--compare` runs a plain backtest of the same strategy alongside and reports the
gap. On SPY over 8 years the two land within 0.83% of each other — the residual
is whole-share rounding, which the live loop does and the simulator does not.
**A large gap means the live loop and the simulator disagree about something,**
which is exactly what you want to discover here rather than with real money.

## Paper trading

Same strategy classes, same target-vs-actual logic — the only difference is that
step 4 talks to a broker instead of a simulator.

```bash
# 1. see what it WOULD do. Sends nothing.
python run_live.py --symbols SPY --strategy "SMA crossover" --once

# 2. actually trade the paper account, one cycle
python run_live.py --symbols SPY --strategy "SMA crossover" --execute --once

# 3. leave it running, checking every 5 minutes
python run_live.py --symbols SPY,QQQ --strategy "SMA crossover" --execute --interval 300
```

**Dry run is the default.** Without `--execute` it logs intended orders and sends
nothing. Read a few days of dry-run output before you let it trade even fake money.

### The kill switch

```bash
# stop trading before the next cycle, from anywhere
echo stop > HALT
```

The loop checks for a `HALT` file every cycle. Delete the file to resume. Ctrl-C
also works, and leaves positions untouched.

### Risk rails, all on by default

| Rail | Default | What it does |
|---|---|---|
| Dry run | on | Logs orders instead of sending them |
| Daily loss limit | 3% | Flattens everything and halts for the day |
| Order notional cap | 1.5× a full position | Blocks an order a sizing bug made 10× too big |
| Market hours | required | Skips the cycle when the market is closed |
| Closed bars only | on | Ignores today's half-formed bar, matching the backtest |
| Whole shares | on | Rounds toward zero, so rounding never *increases* exposure |
| Working-order guard | on | Skips a symbol whose previous order is still unfilled |
| Buying-power check | on | Won't send an order the broker would reject |

That working-order guard matters more than it sounds. The loop reconciles
against *filled* positions, so an order that hasn't filled yet makes the next
cycle think the target was never reached — and it would order the same thing
again. At a 60-second interval with a slow fill, that is a doubled position.

Every cycle appends to `logs/activity.jsonl` — one JSON object per event, so you
can reconstruct exactly what the bot saw and did.

### Going live

`--live` needs `--execute` **and** `--i-understand-this-is-real-money`. That is
deliberate friction. Before you use it, run the identical strategy on paper for
weeks and read the activity log to confirm it did what you expected.

Note that paper and live Alpaca keys are different — swapping one for the other
in `.env` is not enough on its own.

## Cost of running this

Paper trading is entirely free. Live trading has no commissions on US stocks,
ETFs or options, but three things do cost:

- **Regulatory pass-through fees** — pennies on equity sells; roughly $0.048 per
  options contract round trip (ORF + OCC + TAF).
- **Market data** — the free tier is IEX-only (~2.5% of US volume) or 15-minute
  delayed. Fine for daily-bar strategies on liquid ETFs. Full consolidated SIP
  data and OPRA options quotes need Algo Trader Plus at $99/month.
- **Slippage** — always the largest real cost, and the one the backtester models.

## Margin and buying power

FINRA retired the Pattern Day Trader rule, and Alpaca replaced it with an
intraday margin framework on **4 June 2026**. Two consequences for this code:

- **There is no longer a $25,000 minimum to day trade.** 4× intraday buying
  power now starts at $2,000. The old `daytrade_count` and `pattern_day_trader`
  API fields were removed on 6 July 2026, so nothing here reads them.
- **`buying_power` is the number that matters.** Orders creating a margin
  deficit are rejected pre-trade, so the loop sizes against buying power rather
  than discovering the limit through rejections. Exits are exempt — closing a
  position frees capital, so a low buying power never blocks getting out.

An unresolved intraday margin deficit gives you five business days to fix before
the account can be restricted for 90 days. The daily-loss rail exists partly to
keep you well clear of that.

## Adding another broker

Implement `core.broker.Broker` — six methods — and the trading loop works
unchanged. `brokers/fake.py` is the smallest working example.

## Things I got wrong

Keeping this section because the bugs taught me more than the successes did,
and because a repo with no mistakes in it is usually a repo that hasn't been
looked at hard enough.

**The timezone one.** `df.index.tz_convert(None)` doesn't convert to local
time. It drops the timezone and leaves you sitting in UTC. I had a
`between_time("09:30", "16:00")` filter downstream that I thought was selecting
the US session — it was selecting a window in UTC, capturing 36.3% of the day's
volume and mostly the wrong part of it. Every intraday number I had was
garbage, and none of it looked wrong.

**Tests that passed for the wrong reason.** I got into the habit of breaking
each fix on purpose to check that a test caught it. That turned up several
tests which were testing nothing at all. The best one: a test that verified
tamper-detection by string-replacing `10.0` in a JSON file, where the value had
actually serialised as `9.999999999999432`. The replacement never matched, so
the file was never tampered with, so the test passed. It had been green for
days. I also needed three attempts to write the lookahead test correctly — the
first two let a strategy that could see the future sail straight through, which
is worse than having no test.

**A `--dry-run` that wasn't.** The flag whose entire job is to send nothing
sent three real market orders closing a live position, because the end-of-day
flatten sat outside the `if execute:` check. I found it by watching the order
log while testing something unrelated.

**A trade held overnight.** The UI said "closes automatically at 15:55 —
nothing is held overnight". Nothing was running to do it. Cycles only happened
when I clicked a button and I'd stopped clicking. The position sat five minutes
past its own deadline and then through the night. That's what `autorun.py` and
the heartbeat are for — the fix wasn't to soften the wording, it was to make
the sentence true.

**`os.kill(pid, 0)` on Windows.** Not a working liveness check. It only tells
you `OpenProcess` succeeded, and the handle stays valid after the process
exits, so anything still holding one makes a dead process look alive. It also
asks for `PROCESS_ALL_ACCESS`, so a process you don't own a handle to can come
back looking dead — and that's the dangerous direction, because a running
trading loop reading as dead means the app offers to start a second one, and
two loops on one account both act on the same signal. Replaced with
`OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` plus `GetExitCodeProcess`.

**Colours that disagreed with their own legend.** The cycle strip captioned
"watched, no setup" with a blue square and drew it in orange, and captioned
"told to wait" orange while drawing it the same green as "placed an order" — so
the block meaning a trade happened looked like the block meaning nothing did. A
legend that disagrees with the picture is worse than no legend, because it
doesn't leave you guessing, it tells you the wrong thing confidently.

## Where I'd go next

Not further down this road. The result matches what the academic literature
says about retail-accessible technical strategies, which is that they don't
survive costs. If I picked it up again I'd want something structurally
different — genuinely alternative data, or a market where I have some actual
informational reason to think I know something.

What I'd keep is the harness. The engine, the cost model, the significance
testing and the fill measurement are all reusable, and infrastructure that
tells you the truth is worth considerably more than another strategy that
doesn't.

---

Backtested results are hypothetical. This is software for testing your own ideas,
not financial advice, and past performance of any rule here says nothing about its
future performance.

One more time, since this is public: nothing in here has a demonstrated edge,
and I've put real effort into showing exactly that. Don't remove the paper-only
guard and point it at real money. If you want to reuse parts of it, the engine
and the statistics are the bits worth taking.
