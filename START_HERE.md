# Start here

Written for Monday morning. Assumes you have not looked at this in a while.

## Run the app

```bash
.venv/Scripts/streamlit run app.py
```

It opens at <http://localhost:8501>.

## How the app is laid out

The landing page is four quadrants. Pick the kind of algorithm you care about
and everything else follows from that choice.

|  | Fixed rule | Everchanging |
|---|---|---|
| **Short term** | in and out within days | reviewed while you are in the position — day trading |
| **Long term** | held weeks to years | held long, adjusted as the world changes |

"Fixed" means the rule was tested once and left alone, so the backtest you see
is the rule that runs. "Everchanging" means it is reviewed as news lands and
the position moves, and may be swapped for a better-evidenced rule. That review
is deliberately a human step — an algorithm that quietly rewrites itself has no
backtest at all, because whatever was tested is not what is running.

Four pages sit under that:

- **My algorithms** — what is filed in a quadrant, what it made, where it gives
  up, and the reason you wrote down when you saved it.
- **Workshop** — add an algorithm. It is backtested the moment you save it.
  There is no save-without-testing path.
- **Live workshop** — which tested rule should be running right now, the news
  that would make you override that, the orders, and the fill quality.
- **Research** — the old tools (basket tests, event studies, news, scoreboard).

## The honest summary, in one paragraph

Eleven sources of trading advice were collected, transcribed, independently
reviewed, and tested, and then thirty well-known strategies were measured on
top of that. **Nothing beats buying and holding after costs.** Zero of 348
strategy-symbol pairs cleared the corrected significance bar — and not because
of trading costs, since the answer is the same at zero cost. The rules that
appear to win do so only on the one instrument that fell, which is reduced
exposure rather than prediction.

Full detail: [research/STRATEGY_EVALUATION.md](research/STRATEGY_EVALUATION.md)
and [research/audit/REPORT.md](research/audit/REPORT.md).

## What is worth doing Monday

**Run the live workshop as an experiment, not as an investment.**

It trades the practice account only — the code raises an exception if pointed
at a live one, and there is no flag to override that. The point is not profit.
The point is one question no amount of testing on old data can answer:

> When you place an order to buy at a set price, does it actually fill?

Every backtest here assumes a limit order fills whenever the price trades
through it. Real markets fill you when someone wants the other side, which
correlates with you being wrong. If real fills come in far below what the
backtest assumed, then **every** limit-based result in this repo is an
overestimate, and you want to know that before it matters.

### How to run it

1. Open **Live workshop** after 10:30 Eastern on a trading day.
2. Step 1 tells you what should be running. Right now it says the best
   short-term candidate is *inside the noise* — take that seriously.
3. Click **Show me the plan**. Nothing is sent. Most checks find nothing;
   that is normal.
4. If there is a plan, click **Place these orders**.
5. Watch step 4: the **fill rate**, not the profit.

### What to watch for

| Number | Good | Bad | Why |
|---|---|---|---|
| Fill rate | ≥60% | <40% over 20 days | Below 40% the backtest counted fills that are not happening |
| Entry slippage | ≤1 bp | >5 bps | Worse than 5 and the edge is gone by construction |
| Overnight positions | 0 | any | The rule is defined by not holding overnight |

Fifty trades cannot tell a 3 basis point edge from zero, so the profit column
is noise for a long time. The fill rate is answerable in a week.

## The arithmetic behind "big profits, big risks"

The Workshop reports **expectancy** on everything you save — the average result
of one trade. Everything follows from its sign:

- **Positive.** Bet size raises long-run growth up to the Kelly fraction, then
  destroys it. Past roughly twice Kelly the expected growth is *negative* even
  though every individual trade has positive expected value.
- **Negative.** No size is profitable. Larger bets reach zero sooner.

On the 9,825 real day trades measured here: frictionless the rule clears
breakeven by 0.7 percentage points and Kelly says bet **1.3%**. At the measured
spread it is *below* breakeven and Kelly says **0%**. Bet size is set by the
size of the edge, and this edge is nearly zero.

## If something breaks

- **A page shows a stale error after you change code.** Streamlit caches
  imported modules. Restart the server rather than rerunning the page.
- **"No cached calendar".** Click the download button on the page, or run
  `MarketCalendar.fetch("2020-01-01", "2027-12-31")` once while online.
- **Stop everything.** Create a file called `HALT` in the project folder. The
  loop refuses to open anything until you delete it.

## Checking the work

```bash
.venv/Scripts/python -m pytest -q
```

677 tests. They are written to fail when the code is wrong, not to agree with
it — every fix in this repo was checked by deliberately breaking it and
confirming a test caught the break.
