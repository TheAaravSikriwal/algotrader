# Independent audit of `research/sources/01`–`11`

Produced by an independent reviewer given no prior context, no summary of earlier
analysis, and no steer on which strategies looked promising. It read the eleven
sources cold, checked 18 factual claims against primary sources, and backtested
every rule precise enough to implement.

All new code is in `research/audit/` (~60 scripts, 130+ result CSVs). No existing
repo file was modified; the 279 tests still pass; nothing was committed.

**Audit date: 2026-09-11.**

---

> ## Correction, 2026-09-12 — §4.2's recommendation does not hold
>
> The recommendation below was reimplemented in `core/daytrade.py` and re-run
> on freshly fetched Eastern-clock bars. **It does not reproduce, and the
> reason invalidates the recommendation rather than the finding.**
>
> This report measured a gross edge over the whole session, then separately
> recommended restricting to 10:30–15:30 because the spread is cheapest there.
> Those were two independent measurements, and nothing checked that the edge
> survived the restriction. It does not
> (`research/audit/window_sensitivity.py`, 2021–2026, ten symbols):
>
> | window | trades | gross bps | t | spread | **net** |
> |---|---|---|---|---|---|
> | 09:30–09:45 | 880 | 5.38 | 5.89 | 3.78 | **+1.60** |
> | 09:30–10:30 | 12,029 | 3.41 | 6.19 | 2.34 | **+1.07** |
> | full session | 140,879 | 0.91 | 5.80 | 2.10 | **−1.19** |
> | **10:30–15:30 (recommended)** | 121,631 | 0.67 | 3.85 | 1.59 | **−0.92** |
> | 11:00–14:00 | 74,349 | 0.75 | 3.22 | 1.59 | **−0.84** |
> | 15:00–15:55 | 19,328 | 1.31 | 5.31 | 1.79 | **−0.48** |
>
> The gross edge is real and strongly significant, but it is concentrated
> almost entirely in the opening minutes — exactly where the spread is widest.
> §3.4's 5.43 bps headline matches the first-fifteen-minutes figure almost
> exactly, which is the tell that this report's sample was dominated by opening
> trades. **Moving to the cheap window removes the edge along with the cost.**
>
> The honest reading is stronger than the original conclusion, not weaker: the
> edge *is* the spread. Providing liquidity pays roughly what providing
> liquidity is worth, which is what an efficient market predicts. Even the two
> positive rows clear a p90 opening spread of 10.6 bps by nothing — at the
> ninetieth percentile the first-fifteen-minutes variant nets about −5 bps.
>
> **What stands:** every finding in Part 3 about the eleven sources, the
> conclusion that none contains a profitable day-trading rule, the falsification
> protocol in §4.2, and the infrastructure list. The paper-trading protocol is
> now *only* an infrastructure and fill-quality test, which is what §4.2's
> stopping rules already said it was.
>
> **What is superseded:** the specific 10:30–15:30 configuration as a
> candidate for profit, and the "+7–9%/yr gross" expectation built on it.

---

## Executive summary

1. **Not one of the eleven contains a day-trading rule with a profitable edge at
   retail cost levels.**
2. **The most accurate sentence in the collection is source 01's throwaway
   warning** that "trading fees can wipe out small profits." Tested, that is not
   a caveat — it is the result.
3. **One instruction is actively dangerous** (source 03: "take on loans to
   invest") and **one expired claim is dangerous in the opposite direction**
   (source 02's $25,000 PDT minimum, repealed 2026-06-04 — the floor is now
   $2,000, so the accidental guardrail is gone).
4. **Sources 06, 07, 08 converge on low-cost indexing — the only claim that
   survives testing outright**, though several supporting numbers are stale or
   wrong.
5. **The recommendation is not to fund a day-trading strategy.** A pre-committed
   paper protocol follows, whose purpose is to test *infrastructure*, because no
   realistic paper window can test an edge this small.

---

# Part 4 — The recommendation

## 4.1 The honest headline

| Source | Rule | Result |
|---|---|---|
| 05 gap-and-go | top-5 gappers, first pullback to VWAP/MA9 | +0.73%/trade but **t=1.50 (p=0.13)**, and it sits at the **1.5th percentile of its own random-entry null** (null mean +1.16%, sd 0.19, z=−2.29). Random entry on the same days beats it. Negative at measured small-cap spreads. |
| 09 SLC | 4H structure + supply/demand zone + stochastic | **Max \|t\| across all 36 stochastic settings = 0.988** — nothing reaches even an unadjusted 1.96. Break-even **0.70 bps**. Randomly relocated zones beat real ones 6:1. |
| 10 FVG/CHoCH | BOS→CHoCH→limit at FVG midpoint | Robust (+5.43 bps/trade, t=7.14, persists OOS) **but the BOS/CHoCH machinery contributes nothing and ~90% of the edge is generic limit execution.** Dies between 2 and 5 bps of cost. |
| 01 swing momentum | "strong technical momentum", 2–5 days | **0 of 132 cells clear the adjusted bar** on excess over base rate. BH-FDR at 5% and 10%: zero discoveries. |
| 03 Fed quadrant | 4 ordinal predictions | **0/4 correct at 12 months** (1/4 at 3m; random expectation 1.0). Allocation rule 5.51% CAGR / −50.5% DD vs SPY 10.00% / −50.8%. |
| 04 DCA overlay | valuation-timing holdback | Loses under **every** implementable reading. **Inverting the rule beats it.** |
| 11 moon phases | new moon bullish | t=0.232, p=0.817 — the pipeline's null control. It passed. |

## 4.2 What is recommended

**Do not fund a day-trading strategy from this material.** But "do nothing" is
useless, so here is the best available next step — deliberately modest.

### The one rule worth paper-trading

Derived from source 10 **after stripping out everything that failed testing**.
Not called "source 10's strategy," because source 10's distinctive content adds
nothing:

> **Passive limit fade on liquid ETFs, mid-session only.**

| Parameter | Value | Why |
|---|---|---|
| **Universe** | **SPY and QQQ only** | Measured mid-day effective spread **1.59** and **2.12 bps** — the only instruments where a 5.43 bps edge has headroom |
| **Window** | **10:30–15:30 ET only** | SPY spread is **3.78 bps** in the first 15 min (p90 **10.6 bps**) vs 1.59 mid-day. The open alone eats the edge |
| **Signal** | 5-min three-candle FVG: bullish if `high[t−2] < low[t]` | Unambiguous, knowable at candle 3's close |
| **Entry** | Resting **limit at the gap's 50% midpoint**, live 12 bars, then cancel | |
| **Stop** | Most recent confirmed swing beyond the gap, else gap's far edge | Median risk measured at **0.363% of price** |
| **Target** | **2R**, market exit | 2R cells had the highest hit rate (21%) at equal break-even |
| **Hard exit** | **Flat at 15:55 ET unconditionally** | |
| **Frequency** | Max 1 trade/symbol/session (~107/symbol/year) | |
| **Sizing** | **Risk 0.5% of equity** (~1.4× notional) | **20× smaller than source 05's example** |

**On the record:** the FVG contributes only **0.55 of the 5.43 bps** (a limit at
an *arbitrary* level the same distance away earned 4.88 bps). The FVG is kept
because it is a concrete non-discretionary trigger, not because it is proven
better.

### Strength of the evidence

- **Strong:** n=**5,917 trades**, 14,210 symbol-sessions, 10 symbols,
  2021-01-04 → 2026-08-31. Gross **+5.43 bps/trade, t=7.14**. Break-even stable
  at **5.25–6.55 bps across all 18 grid cells**.
- **Genuinely out of sample:** IS (<2024-06) +0.094R (t=4.16, n=3,672);
  **OOS +0.145R (t=2.78, n=2,245)**. Best IS cell scored **OOS t=6.32**.
- **Weak, and this should worry him most:** entering the same signal **at market**
  instead of via a limit gives **−0.009R**. The edge is liquidity provision — the
  service market makers exist to compete away.
- **Weak:** the backtest fills any limit the bar's range covers. Real fill rates
  will be lower and adversely selected. **Most likely failure mode.**
- **Fragile:** 0 bps t=7.14 → 2 bps t=4.51 → **5 bps t=0.57 (p=0.57)** →
  10 bps t=−6.00. It lives in a 3 bps band.

### Falsification, pre-committed

**60 trading days**, SPY+QQQ, ~50 trades. **Stated in advance so it cannot be
rationalised later: this window has essentially zero power.** With measured
per-trade dispersion of **58.5 bps**, 50 trades detect only a ~23 bps edge at 80%
power — four times the gross edge.

- True edge **exactly zero** → **16.9% chance** the 60-day mean exceeds 5 bps
  (a 1-in-6 chance a worthless rule looks better than anything in this audit).
- True edge 3.4 bps → only **10% chance** of significance.
- Even **a full year on 10 symbols** (1,058 trades) gives **47.5% power**.

**So the stopping rule is about execution, not P&L:**

1. **Fill rate ≥60%** of backtest-predicted fills. Below 40% over 20 days →
   **stop; the backtest is invalid.**
2. **Entry slippage ≤1 bp; stop-exit slippage ≤3 bps.** Worse than 5 bps →
   **stop; the edge is gone by construction.**
3. **Mechanical:** zero overnight positions, zero orders outside 10:30–15:30,
   zero orders during halts.

**Immediate kill:** any overnight position; any out-of-window order; paper
drawdown >6% (12 consecutive full losses ≈ 1-in-4,000 — a bug, not bad luck).

**If it passes:** extend to 250 days (~200 trades, minimum detectable 11.6 bps —
still short). **He should go in knowing he will probably never get statistical
confirmation at retail scale.**

### Realistic expectation

Gross 5.43 bps; realistic all-in cost on SPY mid-day **2–3 bps** → net
**2.4–3.4 bps/trade**. ~214 trades/yr at 0.5% risk ≈ **+7–9%/yr gross of tax**.
**Probability of live failure is above 50%**: untested fill assumption, 90% of the
edge is liquidity provision, 3 bps of margin.

Every gain is a **short-term capital gain**. At a combined 40.8%, ~8% gross
becomes **~4.7% net — below the ~4.1% a high-yield savings account paid in
September 2026**, for vastly more work and risk. **That is the real conclusion:**
the best-supported active idea in eleven sources plausibly nets less than cash.

### Infrastructure the repo still needs

1. **The UTC clock bug (fix first).** `core/data.py:_normalise` calls
   `tz_convert(None)`, so the naive index for Alpaca *and* yfinance intraday is
   **UTC, not Eastern**. A naive `between_time("09:30","16:00")` captures
   **36.3%** of the day's volume. Not modified per instructions —
   `research/audit/verify_data.py` demonstrates it; `auditlib.to_eastern` shows
   the DST-correct fix (verified across the Nov 2025 boundary).
2. **Intraday cadence.** No streaming exists; needs websocket or bar-close-aligned
   polling.
3. **Limit + OCO bracket routing**, partial fills, cancel-after-12-bars. Note
   **`core/engine.py` cannot backtest this faithfully** — it fills at the next
   bar's open, a market-order convention. Hence `structure.py:simulate_bracket`
   rather than bypassing its contract.
4. **Calendar handling.** Half-days close 13:00 ET; the flatten must be relative
   to the *actual* close. Use Alpaca's calendar endpoint.
5. **A fill-quality log.** The whole test depends on it and nothing records it
   today. **Highest-value thing to build**, worth it even if he never trades this.

---

# Part 3 — Testing

## 3.1 Regulatory and factual claims

| Claim (source) | Verdict | Primary source |
|---|---|---|
| "$25,000 minimum … Pattern Day Trader" (02) | **REFUTED** | SEC Release **34-105226** (2026-04-14) approved SR-FINRA-2025-017 eliminating PDT; FINRA Notice 26-10, effective **2026-06-04**. Alpaca docs confirm the 4× intraday floor is now **$2,000**. Even pre-repeal it applied only to 4+ day trades in 5 days **in a margin account** |
| Wash sale "rebuy within 30 days" (02) | **PARTIALLY TRUE** | **61 days** (30 before + sale day + 30 after), IRC §1091(a) / IRS Pub 550. Also omits that the loss is added to basis, not destroyed |
| "0.5% profit per day" (02) | **REFUTED** | Compounds to **+251.4%/yr** (1.005²⁵²=3.514). Medallion ≈0.201%/day gross; top 500 of ~450,000 Taiwan day traders earned 37.9 bps/day after fees |
| "74–89% of CFD accounts lose" (07) | **PARTIALLY TRUE / stale** | Exact ESMA wording but a **2018** finding; current per-broker disclosures run ~46–76% |
| Barber/Lee/Liu/Odean ">80% lose" (07) | **VERIFIED** | 80–83% lose net of fees; "less than 1%… predictably and reliably earn positive abnormal returns net of fees" |
| ATTOM flipping "under 25%, 17-yr low" (07) | **PARTIALLY TRUE** | 17-yr low real; ROI was **25.5%**, not under 25 |
| "Oct 2025: savings 0.4%, inflation ~2.9%" (07) | **HALF NON-EXISTENT** | 0.40% exact (FDIC 2025-10-20). **There is no October 2025 CPI** — BLS never collected it (appropriations lapse) |
| "$4T capital gains 2021 > all wages" (07) | **REFUTED** | IRS SOI: gains **$2.03T**, wages **$9.02T**. Both halves fail |
| "~10%/yr since 1926, ~7% real" (07) | **VERIFIED** | 10.31% / 7.15% (Shiller, to 2024-09) |
| "$10k→$80k@30y, $150k@40y" (07) | **PARTIALLY TRUE** | At 7% real: $76,123 / $149,745. 40-yr near-exact, 30-yr ~5% high; neither matches the 10% quoted alongside |
| "10-yr S&P = 13.6%" (08) | **REFUTED / stale** | **14.82%** through 2025-12-31; 13.65% was right through 2025-06-30 |
| "60-yr inflation 3.8%" (08) | **PARTIALLY TRUE / stale** | **3.95%**/yr (CPIAUCSL Aug 1966 → Aug 2026) |
| "$250/mo @8% → $1M in 42y, $2M in 52y" (08) | **VERIFIED** | $1,030,124 and $2,332,240 — both exact |
| "No one lost money over 20 years" (08) | **PARTIALLY TRUE** | True nominal (worst +2.05%/yr, 1929–49). **Worst post-1926 real: +0.35%/yr, Mar 1962 – Mar 1982.** And index funds didn't exist before **1976-08-31** (VFINX inception) |
| "Active funds lag by 2%/yr" (08) | **PARTIALLY TRUE** | 2.1–2.8pp equal-weighted; **1.2–1.4pp asset-weighted** |
| "VUG 13.2% since inception" (04) | **REFUTED** | Vanguard: **12.09%** (inception 2004-01-26). VFINX 11.2% vs actual 11.72% |
| SPIVA / stock-picking difficulty (06) | **VERIFIED** | **85.59% / 89.93% / 92.89%** of US large-cap active funds lag the S&P 500 over 10/15/20yr (SPIVA YE2025) |

## 3.2 Data integrity (done before any backtest)

`research/audit/verify_data.py` — three checks, all passed:

- **Clock:** raw naive index is UTC. The `between_time` trap captures **36.3%** of
  volume vs **71.4%** correctly. DST-correct across Nov 2025.
- **Feed:** RTH minute volume / daily volume = **0.84** → consolidated SIP.
  IEX-only would be ~0.02.
- **Session shape:** exactly **78** 5-min / **390** 1-min bars per session, 100% of
  sessions; RTH open matches the daily open to **0.00 bps**.

`research/audit/selftest_structure.py` — **47/47 pass** on synthetic data where
the answer is known.

## 3.3 Source 09 (SLC) — comprehensively null

41 specifications, Bonferroni bar **|t| ≥ 3.234**.

- **Max |t_R| across all 36 stochastic settings = 0.988.** The entire parameter
  space is null. (The source never states its settings — it shows them on screen
  and tells you to pause. That forced the sweep.)
- Headline: n=355, mean R **+0.031, t=0.49, p=0.63**. Frictionless **+0.0070%** of
  notional → break-even **0.70 bps**.
- Cost sweep: 2 bps → −0.013% (t=−0.57); 5 bps → −0.043% (t=−1.89); 10 bps →
  −0.093% (t=−4.08).
- **Component ablation:** drop **Structure** → +0.034R (n=1,062) — the 4H filter
  adds *nothing*, it only cuts n from 1,062 to 355. Drop **Confirmation** →
  −0.034R (n=1,865).
- **Placebo:** zones relocated at random → **+0.173R (t=3.74)**, break-even
  **4.35 bps** vs the real zones' 0.70. **The placebo is 6× better on the
  cost-comparable measure.**
- Self-test: inverted → −0.180R (t=−4.09), sign flips as it must.
- OOS: IS −0.034R (t=−0.45), OOS +0.130R (t=1.23). Best IS cell OOS t=0.55.

## 3.4 Source 10 (FVG/CHoCH) — real effect, wrong attribution

22 specifications, Bonferroni bar **|t| ≥ 3.052**.

**The effect is real:** +5.43 bps/trade, t=7.14, n=5,917; **OOS t=2.78**;
break-even stable 5.25–6.55 bps across all 18 cells.

**The attribution is entirely wrong:**

| Variant | n | mean R | break-even bps | Reading |
|---|---|---|---|---|
| Model (BOS→CHoCH→FVG limit) | 5,917 | +0.1135 | 5.43 | as written |
| **`fvg_only` — drop ALL BOS/CHoCH** | 14,210 | +0.3294 | **5.45** | **the trend machinery contributes nothing**, on 2.4× more trades |
| **`market_entry` — same signal, no limit** | 6,553 | **−0.0090** | **−0.80** | **the entire edge is the limit order** |
| **`random_level` — limit at a random offset** | 5,930 | +0.0850 | **4.88** | **90% of the edge survives with no FVG at all** |
| `strict_fill` (low < mid) | 5,908 | +0.1038 | 5.29 | fill-at-extreme concern is minor |

- Cost sweep: 2 bps t=4.51; **5 bps t=0.57 (p=0.57)**; 10 bps t=−6.00.
- Random-entry null: real +0.1135R at the 100th percentile (null mean −0.171R).
- **Bug found and fixed:** resolving the bracket from the bar *after* an intrabar
  limit fill lets a fill-and-blow-through candle book a fill and no loss — worth
  **5R on a single constructed trade**. Quantified on real data: same-bar
  resolutions are 5.4% of trades and 7.3% of total R; dropping them entirely still
  leaves +0.0605%/trade, so the bug was not the driver. The self-test for it is
  now permanent.

## 3.5 Source 05 (gap-and-go) — beaten by random entry

Universe: **3,786 distinct symbols, 1,418 trading days**, 2021–2026; 8,117
symbol-days with gap ≥10%.

- Best variant: n=1,924, mean **+0.732%/trade**, **median −2.35%**, win rate
  **25.6%**, **t=1.50, p=0.134** — not significant even unadjusted. The mean is
  carried by a handful of outliers (best +604%).
- **Random-entry null: the real rule sits at the 1.5th percentile** (null mean
  +1.160%, sd 0.187, **z = −2.29**). Entering at random on the same gap days is
  *better* than the "first pullback" rule.
- Costs: at 80 bps round trip (realistic for $2–20 low-float names) mean =
  **−0.068%**; at 160 bps **−0.868%**.
- OOS: best IS variants score OOS t = 0.50–1.35.
- **The float <20M filter is not testable** — no point-in-time float series exists
  in this repo, Alpaca or yfinance. A relaxed version was tested and labelled as
  such.
- **Survivorship bias runs in the strategy's favour** (universe built from
  currently-listed assets) and it still failed.

## 3.6 Source 01 (2–5 day momentum) — 0 of 132

- **Zero of 132 cells clear the Bonferroni bar (3.554)** on excess over the base
  rate. **BH-FDR at 5% and 10%: zero discoveries.**
- **Base rate:** 3-day mean +0.199%, and it is **linear at 0.066%/day from 1 to 10
  days**. There is no bulge at 2–5 days — **the window is arbitrary**.
- **31 of 44 momentum cells do worse than picking at random.** The source's most
  literal reading (high relative volume + up day, 3d) returns **−0.005%** vs a
  +0.199% base rate; break-even **−0.5 bps**.
- OOS: best IS cell in the 2–5 day range → OOS excess t=1.895, p=0.058 (fails even
  unadjusted).
- **Cost arithmetic is decisive:** gross/year is *identical* (16.7%) at every hold
  from 1 to 10 days — only cost changes. A 3-day hold pays 4.2%/yr at 5 bps and
  16.8%/yr at 20 bps for zero gross benefit.
- Self-tests: 8/8 sign flips on inversion; one day of deliberate lookahead takes
  excess t from the ±0.7 range to **+6.7 to +25.9**.

## 3.7 Source 03 (Fed quadrant) — 0/4

- **0/4 ordinal predictions correct at 12 months, 1/4 at 3 months** (random
  expectation 1.0). Q4 — its most emphatic defensive claim — is **dead last**,
  with the preferred archetype underperforming by **7.00pp/yr (t_NW=−3.80)**.
- Allocation rule: CAGR **5.51%**, Sharpe 0.432, max DD **−50.5%** vs SPY
  10.00%/0.719/−50.8% and an exposure-matched 59% SPY sleeve 6.85%/0.798/−33.1%.
  It took the full equity drawdown for 55% of the return.
- **Q1 rests on two macro events** — post-GFC QE and COVID QE; the top 3 episodes
  are 48.7% of Q1 months. Dropping the largest moves archetype D from 3rd to
  **4th**.
- Block-shuffle null (1,000 draws, preserving episode structure): the real
  labelling sits at the **7.5th percentile** on Sharpe and the **10.7th** on
  12-month ordinal hits — **worse than random relabelling**.
- Self-tests: an oracle signal gives 4/4 and Sharpe 1.514, and inverting the
  oracle collapses it — so the pipeline *can* detect a real signal. Inverting the
  real mapping does **not** flip the sign, which is what "no information" looks
  like.
- What actually happened: **QQQ > SPY > IWO > IWD in almost every quadrant.** The
  label adds nothing.

## 3.8 Source 04 (DCA overlay) — loses under every implementable reading

- Terminal-wealth ratio vs plain DCA (1980–2026, $561,000 contributed in **both**
  arms): **0.942** (trailing 50-yr log-linear), **0.522** (50-yr log MA —
  degenerate, invests $500 for 560 of 561 months), **0.952** (10-yr trend). With a
  T-bill reserve: 0.982 / 0.541 / 0.991.
- The **lookahead** version (full-sample trend fit, not implementable) reaches only
  **1.016** — that is the ceiling with perfect foreknowledge.
- **Signal-shuffle null (200 draws): the real overlay sits at the 62nd percentile.
  Inverting the rule beats the rule as written** ($15,643,289 vs $15,620,786).
- Independent observations: **2** non-overlapping 20-year periods, **1** 30-year
  period. The 322 "rolling start dates" are not 322 observations.
- **The GFC claim is not reproducible:** the claimed ratio 247/223 = **1.108** is
  *below the minimum* across all 774 specifications tested (min 1.152). The
  July-2007 investor contributes **$19,000 more** (19 extra months), and on IRR the
  **Feb-2009 investor wins at every end date** (14.81% vs 14.21% to 2026-08).
  2007-07 ranks **66th of 96** start months by IRR.

## 3.9 Source 11 — the null control and the Fibonacci contradiction

**Moon phases (the pipeline's health check):** NEW−FULL = +0.0093%/day,
**t=0.232, p=0.817**, n=3,421 days / 416 lunar cycles. The trap the source falls
into: *both* legs are individually "significant" (new t=2.575, full t=2.038)
because the market drifts up, so every subset of days is positive. Tradeable
version: 2.69% CAGR vs buy-and-hold 10.80%; **the inverted rule earns 7.90%**;
long/short loses 92.5% frictionlessly. The real lunar calendar sits at the **6th
percentile of 50 random-phase calendars** — the actual moon is a *worse* trading
calendar than a random one. **Pipeline passed: handed astrology, it returned
nothing.**

**Fibonacci, 7,819 confirmed upswings across 46 symbols, 2015–2026:**

- P(resume | touched) is **linear in retracement depth: slope −0.606,
  R² = 0.9907.** Residuals at 0.382 (+0.0099) and 0.618 (+0.0097) are *smaller*
  than arbitrary controls at 0.450 (+0.0122) and 0.550 (+0.0114). **There is no
  Fibonacci bump.**
- Drift-adjusted forward returns across 11 levels × 2 horizons: **max |t| = 1.023**
  against a Bonferroni bar of 3.172.
- **Source 11's claim is a base-rate inversion.** 0.382 is touched 81.3% of the
  time, 0.786 only 55.0%, so shallow levels dominate any tally of "reversals from"
  a level. The decision-relevant quantity runs the *opposite* way: P(resume|touched)
  is 0.442 at 0.382 and 0.299 at 0.618, both **below** the unconditional 0.546.
- Placebo (random bar in the same episode) gives excess +0.117%, **t=2.97 — larger
  than any Fibonacci level.**

## 3.10 Baselines the sources should be judged against

**Measured spreads** (Corwin–Schultz, 1,421 sessions) — this is what every marginal
case turns on:

| | 09:30–09:45 | 09:45–10:30 | 10:30–15:30 | 15:30–16:00 |
|---|---|---|---|---|
| SPY | **3.78** (p90 10.6) | 2.34 | **1.59** | 1.79 |
| QQQ | 6.00 | 3.51 | **2.12** | 2.21 |
| IWM | **7.85** | 3.96 | 2.10 | 2.18 |

Spreads widen with volatility: SPY 3.07 bps in 2022 vs 1.56 in 2024.

**Overnight vs intraday decomposition:** the asymmetry is large but *not*
universal. AAPL overnight Sharpe **0.837** vs intraday 0.221; AMD **1.130** vs
**−0.138**; but ADBE runs the other way (intraday 0.486 vs overnight 0.258). The
tradeable version needs 504 round trips/yr: SPY break-even **4.01 bps** full
sample, **3.28 bps** for 2021–2026. At 3 bps t=1.39 (insignificant); at 5 bps
negative. **The effect is real and has decayed to roughly the width of its own
transaction cost.**

**Opening-range breakout** (not from the sources — the best-documented retail
intraday rule, used as the benchmark): **1,728 specifications**, Bonferroni bar
**3.925**. Three of four symbols' in-sample picks **die completely** out of
sample — SPY IS t=2.80 → **OOS t=−0.19**; QQQ 2.78 → **0.14**; NVDA 3.58 →
**0.24**. Only TSLA survived (IS 2.94 → OOS 2.29, break-even 20.1 bps), and it
does not clear the bar for a 1,728-cell search. **ORB is not a free lunch either.**

## 3.11 Position sizing and statistical power

**Risk of ruin** (Monte Carlo on source 05's *own* numbers — $200 risk, 50% win
rate):

- $2,000 account (the post-repeal floor) = **10% risk per trade**, with a *genuine*
  +0.25R edge: **18.1%** chance of being down 50% within a year; average worst
  drawdown **68.4%**.
- **With no edge** (coin flip minus 5 bps cost) at 10% risk: **95.7%** chance of
  being down 50%; median outcome **$393 from $5,000**.
- At 1% risk with no edge: a slow bleed (median $4,353) rather than a blowup.
  **Sizing, not the strategy, is what kills the account.**

**Power** (per-trade sd = 58.5 bps, measured): minimum detectable edge at 80%
power — 60 days × 5 symbols (126 trades) = **14.6 bps**; 252 days × 10 symbols
(1,058 trades) = **5.0 bps**. To detect a 3.43 bps net edge needs **2,284 trades
≈ 2.1 years** on ten symbols.

---

# Part 1 — Per-source reading

**01 — 1-to-7 day list.** No monetisation, no performance claims, volunteers its
own risks. Its only tradeable item is **unfalsifiable**: "strong technical
momentum" names no indicator, threshold or lookback. 11 readings were tested
rather than inventing one and attributing it. Its warning about fees is verified
and is its most accurate statement.

**02 — Gemini "make money every day."** Intraday. **Internally contradictory**:
presents 0.5%/day (+251%/yr) alongside "the vast majority of retail day traders
lose money." Nothing implementable — "extract small profits from minor
fluctuations" has no entry or exit. Its two checkable facts (PDT, wash sale) are
both wrong. No author accountability, and it ends by asking for the reader's
capital and hours, i.e. it is optimised to continue a conversation.

**03 — Fed chessboard.** 3 months–2 years. The **most testable macro model** in the
collection: quadrants from FRED `FEDFUNDS`/`WALCL`, four ordinal predictions,
stated cash weights. The WACC mechanism is textbook-correct as theory; the
*empirical* prediction fails. The exit rule — "when you **feel** like you gained a
lot of profit" — is pure hindsight and is where all claimed performance would
live. Only channel monetisation; the author disclaims it himself.

**04 — Advanced DCA.** Decades. Base claim (keep buying through a crash) is sound.
The "advanced" overlay is the part being sold and its only input — "the 50-year
mean growth of the market" — is **never defined**. Methodologically disqualifying:
figures "sourced by asking ChatGPT" and a backtest on "ChatGPT-generated data."
Also falsely describes the S&P 500 as "the top 500 performing companies in the
world." Monetised via a wealth-training funnel; opens with "No BS, no hype," which
is itself hype.

**05 — Warrior Trading gap-and-go.** Intraday. **Highest conflict of interest** —
paid curriculum, daily show, repeated CTAs — though it does disclose "my results
are not typical" and "most beginner traders do lose money." Codeable: gap
definition, $2–$20 band, top-5 gappers, VWAP/MA9, "first candle makes a new high."
**Not testable:** float <20M. **Unfalsifiable:** "most obvious," "trending on
social media," a second pullback being "a bit extended." Evidence is **one case
study** (n=1) selected for being spectacular.

**06 — The Plain Bagel research process.** Years. The most professionally sound
source, and the one that most explicitly tells the reader not to do what it
describes. Its opening caveat is backed by the strongest evidence of any claim in
all eleven files. **Not codeable in this repo** — there is no fundamentals data
source at all; testing it would need an EDGAR/XBRL pipeline and would not answer a
day-trading question anyway. It also relies on a paid terminal and direct
executive access, which he flags himself. Its warning that adjusted figures are
adjusted *because* they fail accounting rules is unusually honest.

**07 — Alux tier list.** Mixed. Nothing to implement; its value is factual claims
plus the leverage arithmetic, which is **correct and is the most useful thing in
it**. Direct conflict: it ranks private equity at S tier while selling access to a
"private equity insider" collection, then concedes S tier is "out of reach for 99%
of people." Its statistics are unreliable about half the time.

**08 — Mark Tilbury index funds.** Decades. Prerequisites (clear high-interest
debt, then a 3–6 month emergency fund) are correct and well-evidenced, as is the
core recommendation. Return statistics are stale in a consistently *flattering*
direction, and it is **internally inconsistent** — "8–10%" on one page and "13.6%"
on another. Sponsored by Trading 212 with a referral code, and claims investments
"grow by around $117,000 a week."

**09 — SLC.** Intraday. Better specified than most — the zone-drawing rule is
unusually precise ("the last candle before the aggressive move"). But the
**confirmation step's parameters are shown on screen and never stated**, which
makes the strategy unevaluable as written and forced a 36-cell sweep. "Works every
single day" is unfalsifiable as stated and false as written. To its credit it warns
that "supply levels aren't magic."

**10 — BOS/CHoCH/FVG.** Intraday. **The most precisely specified rules in the
collection** — the three-candle FVG, the close-confirmed BOS, the 50% entry and 4R
target are all exact. Its **best passages are epistemic, not technical**: "a
perfect setup is not a trade that is guaranteed to win," don't size up on
conviction, "what makes a good trade is when I'm following my actual rules." That
is genuinely good process discipline and is the part worth keeping. Its take-profit
refinement is conceded by the author to be "reverse engineering your success" — a
description of curve-fitting, volunteered. One load-bearing claim is unsupported:
"an edge that we know is there from data testing," with no testing shown.

**11 — TA catalogue.** No performance claims, no backtests, minimal monetisation —
the most epistemically honest file, even though much of its content is
unsupported. **The moon-phase entry is the tell**: presented in the same register
as VWAP with no distinguishing caveat. A catalogue that cannot separate astrology
from volume-weighted average price is not exercising quality control on any entry.
Its **Elliott wave and Gann material are unfalsifiable by construction** —
Elliott's rules only tell you when a count is *invalid*, and the count is assigned
in hindsight; Gann's setup says "mark the swing low and high of a range" without
saying which range. These were judged rather than tested at length. Two statements
are notably accurate: Heikin Ashi and Renko "do not display the real market price."

---

# Part 2 — Cross-source adjudication

## Where they agree

| Agreement | Sources | Supported? |
|---|---|---|
| Most retail day traders lose money | 02, 05, 07 | **YES** — 80–83% lose; <1% predictably profitable |
| Low-cost indexing beats active for most | 06, 07, 08 | **YES** — SPIVA 85.6/89.9/92.9% at 10/15/20yr |
| Costs and taxes erode short-horizon returns | 01, 02 | **YES**, and by more than either implies |
| Supply/demand zones drawn the same way | 09, 11 | Definitions agree; **the rule doesn't work** |
| BOS/CHoCH definitions | 10, 11 | Agree, except 10 requires a candle **close**. 10's is stricter and is what was implemented |
| Paper trade first | 02, 05 | **YES** — and §4.2 quantifies for how long |

**Every agreement that survives testing is a warning. Every agreement about how to
make money does not.**

## Direct contradictions

1. **0.382 or 0.618?** Source 11 says 0.382 is "the most common level where price
   tends to reverse from"; source 10 calls 0.618 the important one. **Both wrong** —
   P(resume|touch) is linear in depth (R²=0.991) with no bump at either, and
   arbitrary controls at 0.45/0.55 have *larger* residuals. Source 11's version is
   a base-rate inversion.
2. **Is high RSI a buy or a sell?** Source 01 buys momentum; source 11 reads
   overbought as a reversal down. **Source 11's reading is the one the data
   supports** — the mirror beats momentum in **34 of 44** head-to-head cells
   (RSI<30 held 3 days: +0.451% vs RSI>70's +0.149%). **Caveat: neither side clears
   the adjusted bar.** Source 11 is less wrong, not right.
3. **Leverage?** Source 03 says borrow to invest; source 07's F tier shows −5% wipes
   a 20× account; source 08 refuses margin outright. **07 and 08 are right; 03 is
   dangerous** — its own rule took −50.5% unlevered.
4. **Does entry timing matter?** Source 04's base claim says no; source 04's overlay
   says yes. **The same source contradicts itself**, and the overlay is the half
   that loses.
5. **When is a stochastic useful?** Source 11 says oscillators suit choppy markets;
   source 09 uses one *only* when the 4H chart is trending and forbids trading in
   consolidation — exactly inverting 11's guidance for that tool. Moot, since
   neither configuration worked, but it shows no shared theory of the indicator.
6. **How many trades a day?** Source 05: "one good trade a day." Source 02: many
   small ones. **05 is right, for a reason neither states** — gross return per year
   is identical at every holding period; only cost scales with frequency.
7. **Is stock picking worth doing?** Source 06 (a working portfolio manager) says
   most people shouldn't and cites the evidence; sources 03/05/09/10 assume they
   should. **06 wins.**
8. **What return should the S&P 500 give?** 07 says ~10% since 1926 (**correct**);
   08 says 8–10% *and* 13.6%; 04 says 11.2%, 13.2%, or "16.24 averaged down to
   13%." Source 08's figure is a stale trailing-decade number offered as a forward
   expectation — extrapolating a bull decade.

## Same word, different thing — more dangerous than the open contradictions

- **"Momentum"** means three different things: time-series technical (01),
  hype-chasing (07), and cross-sectional ranking (academic). Only the third showed
  anything in testing (+0.117% over 3 days, top-decile 20-day return) — **and no
  source in the collection actually describes it.**
- **"Fair value gap"**: source 10 uses it as an **entry**; source 11 as a
  **target/magnet**. Same construction, opposite trade role. Combining them would
  have you enter and target the same object.
- **"Change of character"**: 10 requires a candle close; 11 doesn't. Different,
  noisier signal.
- **"Dollar-cost averaging"**: 08 means a fixed direct debit; 04 means a monthly
  deposit **with a timing overlay**, which is market timing wearing DCA's name.
- **"Value investing"**: 07 means undervalued companies; 03's "stock A" is a
  quantitative screen; 06 means a fundamental process.

## What cannot be combined

The brief asked for "the best combination from each." **Most of these cannot be
combined**, for structural reasons:

- **Horizons don't compose.** Source 06 takes *weeks per company* for a *multi-year*
  hold; source 05 wants a decision in the first 15 minutes. A $2–20 low-float shell
  with negative EBITDA fails 06's screen at step 1, by construction.
- **Source 03's Q1 and source 07's F tier are mutually exclusive** — borrow to buy
  loss-making growth vs leverage is how retail accounts get wiped out.
- **Sources 09 and 10 are incompatible intraday reversal models** that both fail
  independently; their intersection has no reason to work and would have almost no
  sample.
- **One genuine combination survives, and it is not a trading strategy:** sources
  06, 07 and 08 agree on low-cost broad indexing with a long horizon, with source
  08's prerequisites in front of it.

---

## Bottom line

The request was a "super algo" built from the best of eleven sources. The honest
answer is that **the material does not contain the parts for one.** What it does
contain is a reliable set of warnings, one dangerous leverage instruction, one
expired regulation that now cuts against him, and — buried in source 10 and
stripped of everything that source teaches — a small, fragile, execution-driven
effect that plausibly nets less after tax than a savings account.

The most valuable thing to build next is not a strategy. It is the **fill-quality
log and the UTC clock fix** — because without those, a real edge cannot be told
from a backtest artefact, and that distinction is what separates this project from
the eleven sources it started with.
