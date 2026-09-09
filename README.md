# Backtester

A backtesting workbench for US equities, built around Alpaca. Pick a symbol, pick
a rule, see what it would have done — with a dashboard, a CLI, and an engine you
can trust not to cheat.

Nothing here places an order. That comes later, and only after a rule has
survived a backtest and a paper account.

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
| `strategies/builtin.py` | Seven reference rules — **your templates** |
| `run_live.py` | The paper/live trading loop |
| `core/trader.py` | Reconciliation, risk rails, kill switch |
| `core/broker.py` | The `Broker` interface every venue implements |
| `brokers/alpaca.py` | Alpaca adapter (paper and live) |
| `brokers/fake.py` | In-memory broker for testing the loop offline |
| `replay.py` | Replay history through the live loop at full speed |
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
python test_engine.py
```

They check the no-lookahead rule, that costs actually cost money, that stops cap
losses at the stop level, and that realised trade P&L reconciles with the equity
curve.

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

---

Backtested results are hypothetical. This is software for testing your own ideas,
not financial advice, and past performance of any rule here says nothing about its
future performance.
