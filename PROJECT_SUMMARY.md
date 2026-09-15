# Algorithmic Trading Research Platform — Project Summary

*Prepared as source material for a résumé. Every figure below is measured from
the repository, not estimated. Notes on honest framing are at the end.*

---

## One-line description

A Python research platform for designing, backtesting, and paper-trading
equity strategies, built around a falsification-first methodology that
concluded — with statistical evidence — that none of the 33 implemented
strategies has a tradeable edge after costs.

---

## Quantified scope

| Metric | Value |
|---|---|
| Python files | 178 |
| Lines of Python | ~37,900 |
| Automated tests | **905** (36 test modules, 684 test functions + parametrised cases) |
| Core library modules | 47 |
| Trading strategies implemented | 33 |
| Git commits | 64 |
| Active development window | 2026-09-09 → 2026-09-15 |
| Backtested strategy-symbol pairs | 348 daily + 188 intraday |
| Research scripts (audit sub-project) | ~60, producing 130+ result CSVs |

**Stack:** Python 3.12, pandas, NumPy, Plotly, Streamlit, Alpaca Trading API
(`alpaca-py`), yfinance, pytest, Win32 API via `ctypes`, Git.

---

## What the system does

### 1. Backtesting engine
- Two engines: daily-bar (`core/engine.py`) and intraday 5-minute
  (`core/intraday.py`).
- **No-lookahead execution enforced structurally**: a signal generated at bar
  *t* fills at the **open of bar t+1**, never at bar *t*'s close. Verified by
  oracle tests that inject a strategy with perfect knowledge of the future and
  assert the delay neutralises it.
- Transaction costs modelled on both entry and exit; intraday backtests are
  window-confined and flat by the close.
- Market-calendar aware (`core/marketclock.py`): real half-days verified
  against the exchange calendar (2026-11-27, 2026-12-24, 2027-11-26 all close
  13:00). Backtests **refuse to run** over date ranges the loaded calendar
  does not cover, rather than silently skipping them.

### 2. Statistical evaluation
- **Multiple-comparison correction**: Bonferroni and Benjamini-Hochberg FDR
  applied across all strategy-symbol pairs — the core defence against
  selecting a winner out of noise.
- **Null controls**: deliberately meaningless strategies ("Day of week",
  "Turn of month") run alongside real ones as a calibration check. The null
  controls repeatedly outscored the real strategies, which is itself the
  finding.
- **Expectancy toolkit** (`core/expectancy.py`): expectancy, Kelly fraction,
  quarter-Kelly position sizing, break-even win rate, sample size required for
  significance, and Monte Carlo risk-of-ruin.
- **Power analysis** before drawing conclusions from small samples.

### 3. Live paper trading
- Alpaca integration, **paper accounts only**: `DayTrader.execute()` raises
  `NotPaper` on any non-paper account, with no override flag. Live API
  credentials are held in separate, deliberately empty environment variables
  so live mode cannot start.
- Risk rails enforced in one place: daily loss limit, position caps,
  stale-data refusal, trading-window confinement, and an end-of-day flatten.
- **Fill-quality measurement** (`core/fills.py`): logs every order's realised
  slippage against the quoted spread to test whether the backtest's fill
  assumption survives contact with a real venue. Slippage is signed so
  positive is always a cost on both sides; unfilled orders return `None`, not
  `0.0`; derived columns are recomputed on read so a tampered log cannot
  flatter a result; and it refuses to render a verdict under 30 orders.

### 4. Visual operations interface
- Streamlit app with three views: **Trade** (auto/manual), **Dashboard**
  (across multiple running instances), and **Workshop** (strategy testing).
- A four-stage workflow visualisation (Watching → Order placed → Holding →
  Closed) where each stage exposes, on hover *and keyboard focus*, what has to
  happen to advance and what the system will do — including stages not yet
  reached.
- Live money panel showing capital deployed, current value, and realisable
  P&L **at the bid/ask you would actually transact at**, not the midpoint,
  with the spread cost stated explicitly.

### 5. Autonomous execution
- `autorun.py`: a background process that cycles on a timer, obeys queued
  instructions, selects a rule, acts, and publishes state.
- **Liveness detection** (`core/heartbeat.py`): a heartbeat carrying a PID,
  checked for both freshness *and* real process liveness, so the UI can never
  claim positions are being watched when nothing is running.
- Start/stop from the UI via a detached child process with per-instance
  interlocking that prevents two loops trading one account.
- A file-based control bridge (`core/livestate.py`) with atomic writes,
  instruction expiry, and single-consumption semantics.

---

## Engineering practices worth highlighting

### Mutation testing as standard practice
Every bug fix was validated by deliberately re-breaking it and confirming a
test failed. This caught **multiple hollow tests that passed for the wrong
reason**, including:

- A test that "verified" slippage tampering by string-replacing `10.0`, when
  the value actually serialised as `9.999999999999432` — the replacement never
  matched, so the test passed without testing anything. Fixed with a regex
  plus an assertion that exactly one substitution occurred.
- A lookahead-detection test written **twice** that both times let an oracle
  strategy pass. The third version introduced an immediate price-level shift,
  after which all four probes were caught.

Mutation sweeps were run with git-based restore and a working-tree
cleanliness assertion after *every* probe — added after an interrupted sweep
left a mutated line in a commit.

### Structural fixes over local patches
Recurring classes of bug were fixed at the source rather than per-occurrence:
- Currency formatting that Streamlit misread as LaTeX was fixed by a single
  escaping helper, after being patched three times locally.
- Chart legend colours and the chart palette were unified into one mapping, so
  the caption cannot drift from the picture it describes.

### Honest instrumentation
A recurring theme: the UI was repeatedly found asserting things the code could
not deliver, and each was fixed in the code, not the wording.
- The panel displayed the *recommender's* top-ranked strategy beside orders
  placed by a **different** strategy on a different symbol. Corrected to name
  the rule that actually executes, with its own measured (negative) edge.
- The UI promised "closes automatically at 15:55 — nothing is held overnight"
  while nothing ran between button presses. A real position went past its
  deadline and was held overnight. Fixed by building the background runner
  *and* making the page read a heartbeat rather than assume.

---

## Notable defects found and fixed

These make strong interview material because each has a clear mechanism and a
measurable consequence.

| Defect | Consequence | Resolution |
|---|---|---|
| Timezone bug: `tz_convert(None)` left the index in UTC | Session filter captured only **36.3%** of daily volume; every intraday result invalid | Convert to Eastern before dropping the zone; mutation-tested 5/5 |
| `--dry-run` placed real market orders | A flag whose sole purpose is "send nothing" closed a live position — three times | Flatten path was outside the `execute` guard; verified fixed against the live account (27 orders before, 27 after) |
| Daily trade cap applied during scanning | Hid 6 of 7 valid setups; live loop found nothing all day | Cap now counts orders actually sent, read from the fill log |
| Short positions described as "bought" | UI told the user they'd *receive* money they were about to *pay* | Side-aware wording moved onto the valuation object so it cannot drift from the number beside it |
| `os.kill(pid, 0)` used for process liveness on Windows | Reported dead processes alive (stale handle) and live processes dead (`PROCESS_ALL_ACCESS` denial) — the latter would let a second trading loop start | Replaced with `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess`; access-denied correctly read as *alive* |
| `DETACHED_PROCESS` + `CREATE_NO_WINDOW` | Mutually exclusive; background loop died instantly with `0xC000013A` | Measured both flags in two parent contexts and documented the matrix in the source |
| Plotly `title=None` | Rendered the literal string "undefined" above every chart | `title=None` builds an empty Title object; an explicit empty string is required |

---

## Research findings (the actual result)

The project's headline output is a **rigorous negative result**, backed by
pre-registered falsification criteria.

**Daily-bar strategies — 348 strategy-symbol pairs:**
- **0** cleared the Bonferroni-corrected significance bar (|t| > 3.81)
- **0** cleared Benjamini-Hochberg FDR at 5%
- Only 7.5% beat buy-and-hold at all
- Result unchanged at 0, 2, 5, and 10 bps of modelled cost — i.e. the failure
  is not a cost-modelling artefact

**Intraday strategies:** 0 of 70 cleared on SIP data; 0 of 118 on IEX data.

**Apparent winners were exposure dilution, not skill.** 86% of strategies beat
buy-and-hold on TLT — the single instrument that fell over the period — and 0%
did on the eleven that rose. Strategies were being rewarded for being out of
the market, not for timing it.

**The one rule that survived initial screening did not survive scrutiny.** A
fair-value-gap limit-entry rule showed a strongly significant gross edge, and
a window-sensitivity analysis showed that edge is concentrated entirely in the
opening minutes — precisely where the bid-ask spread is widest:

| Window | Trades | Gross (bps) | t | Spread | **Net** |
|---|---|---|---|---|---|
| 09:30–09:45 | 880 | 5.38 | 5.89 | 3.78 | **+1.60** |
| 09:30–10:30 | 12,029 | 3.41 | 6.19 | 2.34 | **+1.07** |
| Full session | 140,879 | 0.91 | 5.80 | 2.10 | **−1.19** |
| 10:30–15:30 | 121,631 | 0.67 | 3.85 | 1.59 | **−0.92** |

**Conclusion: the edge *is* the spread.** Providing liquidity pays roughly what
providing liquidity is worth — which is what an efficient market predicts.
Even the positive rows fail to clear a 90th-percentile opening spread of
10.6 bps.

**A methodological finding on data feeds:** switching from SIP (full
consolidated tape, 15-minute delayed on the free tier) to IEX (real-time, ~4%
of volume) **reversed the ranking**. RSI mean reversion scored +13.99 bps on
SIP and −5.46 bps on IEX. Strategy selection was sensitive to data source in a
way that invalidates conclusions drawn from either alone.

**A limit of paper trading, documented rather than glossed over**
(`research/PAPER_TRADING_LIMITS.md`): the paper simulator fills orders using
the backtest's own fill rule. A limit resting at the bid filled in 2 seconds
with a round-trip cost of 0.13 bps against a 2.0 bps quoted spread — a fill
quality no real venue would provide. Paper trading therefore validates
infrastructure and plumbing, **not** profitability.

---

## Suggested résumé bullets

Pick 3–5. Each is defensible under questioning.

> **Algorithmic Trading Research Platform** — Python, pandas, Streamlit, Alpaca API
>
> - Built a 37,000-line backtesting and paper-trading platform covering 33
>   equity strategies, with 905 automated tests and structurally enforced
>   no-lookahead execution verified by oracle tests using future-knowledge
>   strategies.
> - Applied Bonferroni and Benjamini-Hochberg corrections across 536
>   strategy-symbol backtests; demonstrated that **zero** cleared significance
>   after multiple-comparison correction, and that apparent outperformance was
>   exposure dilution rather than timing skill.
> - Identified via window-sensitivity analysis that a strategy's statistically
>   significant gross edge (t = 5.89) was entirely consumed by the bid-ask
>   spread, converting a candidate +1.60 bps result into −0.92 bps net.
> - Adopted mutation testing as standard practice, deliberately re-breaking
>   every fix to verify test coverage; caught multiple tests that passed for
>   the wrong reason, including one where a floating-point serialisation
>   mismatch meant the assertion never executed.
> - Diagnosed a timezone defect where `tz_convert(None)` silently retained UTC,
>   causing the regular-hours filter to capture 36.3% of daily volume and
>   invalidating all intraday results.
> - Replaced `os.kill(pid, 0)` process-liveness detection on Windows with
>   `OpenProcess`/`GetExitCodeProcess` after measuring that it reported exited
>   processes as alive and live processes as dead — a failure that would have
>   permitted two trading loops on a single account.

---

## Framing guidance — read before submitting

**Do claim:** research rigour, statistical discipline, testing depth, systems
debugging, and the willingness to publish a negative result.

**Do not claim:** that you built a profitable trading system, generated
returns, or deployed to live markets. None of those are true, and a trading
interviewer will ask. The project never traded real money — by design, with a
hard guard in code.

**The negative result is the strongest part of this project, not a weakness.**
Most retail trading projects present a backtest with a rising equity curve and
no correction for multiple comparisons, no cost modelling, and no out-of-sample
discipline. This one ran the corrections, built null controls, found nothing,
and said so. Anyone doing quantitative work for a living will recognise that as
the harder and more honest outcome — and will be more interested in *how you
established it* than in a curve you can't defend.

If asked "so it doesn't make money?", the answer is: *"No — and demonstrating
that took more work than producing something that appeared to. The edge that
survived screening turned out to be the bid-ask spread. That's what an
efficient market predicts, and I could show it with a window-sensitivity
analysis rather than just asserting it."*

**Suggested title:** "Algorithmic Trading Research Platform" or "Quantitative
Strategy Evaluation Framework" — not "Trading Bot", which invites the wrong
question.
