# Start here

Written for Monday morning. Assumes you have not looked at this in a while.

## Run the app

```bash
.venv/Scripts/streamlit run app.py
```

It opens at <http://localhost:8501>. Seven pages down the left. The home page
lists them in the order they are meant to be used; you do not need all of them.

## The honest summary, in one paragraph

Eleven sources of trading advice were collected, transcribed, independently
reviewed, and tested. **None of them contains a day-trading rule that makes
money after costs.** The best candidate has a real edge before costs, but the
edge sits in the first few minutes after the open, and that is exactly when
trading is most expensive. In the cheap hours the edge is smaller than the
spread. That is not a failure of the testing — it is the answer.

Full detail: [research/audit/REPORT.md](research/audit/REPORT.md). Read the
correction box at the top first; it supersedes the report's own recommendation.

## What is worth doing Monday

**Run the day-trading page as an experiment, not as an investment.**

Page 7. It trades the practice account only — the code raises an exception if
pointed at a live one, and there is no flag to override that. The point is not
profit. The point is one question no amount of testing on old data can answer:

> When you place an order to buy at a set price, does it actually fill?

Every backtest here assumes a limit order fills whenever the price trades
through it. Real markets fill you when someone wants the other side, which
correlates with you being wrong. If real fills come in far below what the
backtest assumed, then **every** limit-based result in this repo is an
overestimate, and you want to know that before it matters.

### How to run it

1. Open page 7 after 10:30 Eastern on a trading day.
2. Click **Show me the plan**. Nothing is sent. Most checks find nothing —
   that is normal.
3. If there is a plan, click **Place these orders**.
4. Check step 3 on the page. Watch the **fill rate**, not the profit.

### What the numbers mean

| Reading | What it tells you |
|---|---|
| Fill rate below 40% over 20 days | The backtest was wrong. Stop; every limit result here is inflated. |
| Fill rate above 60% | The assumption is holding. Keep going. |
| Entry slippage above 5 bps | The edge is gone by construction. |
| Under 30 orders | Too early to read anything. The page says so. |

**One caveat about size.** With a $5,000 practice account and SPY near $765,
the per-order notional cap binds long before the risk rule does, so orders come
out at one or two shares. That is fine for learning the mechanics, but a
one-share limit fills far more easily than a realistic one — it can slot into a
queue gap that a hundred shares could not. Treat a good fill rate as an upper
bound rather than a result.

**Ignore the profit and loss.** Fifty trades cannot tell a 3 bps edge from
zero — you would need roughly 2,300 trades, about two years. If the equity
curve looks good after a week, that is noise, and acting on it is the mistake
this whole project exists to avoid.

## Stopping it

Create an empty file called `HALT` in the project folder. The loop refuses to
open anything until you delete it.

```bash
touch HALT      # stop
rm HALT         # resume
```

## Safety, as actually implemented

- Paper account only, enforced in code, no override flag.
- Flat before the close every day, using the venue's real calendar — so it
  closes at 12:55 on a half-day rather than trying at 15:55.
- Daily loss limit, with the baseline stored on disk so it survives page
  reloads.
- Caps on position count and single-order size.
- Stops and targets are attached at the broker as a bracket, so they survive
  the app being closed.
- `HALT` file.
- Live Alpaca keys live in separate environment variables and are blank, so
  live mode cannot start by accident.

## If something looks wrong

The test suite is the fastest check:

```bash
.venv/Scripts/python -m pytest -q
```

412 tests. Every fix in this repo was verified by deliberately breaking it
and confirming a test failed — several times that exposed a test which proved
nothing, so it is worth trusting a failure here.

## What is not built

- No automatic loop. You click to plan and click to send. Deliberate: the
  order sent is the order you saw.
- No streaming. The page reads bars when you click.
