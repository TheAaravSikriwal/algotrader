"""Measure every registered strategy against the same honest standard.

The question is not "which one made the most money" -- on any sample, some
rule always did. The question is whether any of them beat **buying and
holding the same instrument**, after costs, out of sample, once you account
for having looked at dozens of them.

Four things are done here that a naive comparison skips, each of which
normally flips the conclusion:

1. **Costs on every trade.** Positions change, and each change pays the
   spread. A rule that trades daily pays roughly 250x what a rule that trades
   yearly pays for the same gross edge.
2. **The benchmark is buy-and-hold of the same symbol over the same bars**,
   not zero. A strategy that made 8% a year during a decade when the index
   made 13% lost to doing nothing.
3. **Out of sample.** Everything is split at a fixed date. The in-sample half
   is where a strategy is allowed to look good; the out-of-sample half is
   where it has to.
4. **Multiple comparisons.** Roughly thirty strategies are tested on each of
   several symbols. At an unadjusted 5% threshold you expect false positives
   by construction, so both a Bonferroni bar and Benjamini-Hochberg FDR are
   applied to the out-of-sample excess returns.

Usage:
    python research/evaluate_all.py                 # default universe
    python research/evaluate_all.py --quick         # three symbols
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.data import DataError, bars_per_year, load_bars  # noqa: E402
from core.engine import BacktestConfig, run_backtest  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402
from core.strategy import REGISTRY  # noqa: E402
import strategies  # noqa: F401,E402

OUT = REPO / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "JPM", "XOM",
            "JNJ", "TLT", "GLD", "KO"]
QUICK = ["SPY", "AAPL", "TLT"]

START, END = "2010-01-01", "2026-09-01"
SPLIT = pd.Timestamp("2020-01-01")     # fixed in advance, not chosen from results
COST_BPS = 5.0                         # round trip, per change of position


def load(symbols: list[str]) -> dict[str, pd.DataFrame]:
    bars = {}
    for s in symbols:
        try:
            df = load_bars(s, START, END, "1Day", "yfinance")
            if len(df) > 800:
                bars[s] = df
        except (DataError, Exception) as exc:
            print(f"  {s}: skipped ({type(exc).__name__})")
    return bars


def evaluate(df: pd.DataFrame, name: str, cost_bps: float) -> dict | None:
    """One strategy on one symbol. Returns per-bar excess over buy-and-hold."""
    try:
        sig = REGISTRY[name]().generate_signals(df)
    except Exception:
        return None
    if sig is None or sig.isna().all():
        return None

    # slippage_bps is charged per side, so half the round-trip figure.
    cfg = BacktestConfig(slippage_bps=cost_bps / 2.0, allow_short=True)
    try:
        res = run_backtest(df, sig, cfg)
    except Exception:
        return None

    strat_ret = res.equity.pct_change()
    hold_ret = df["close"].pct_change()

    excess = (strat_ret - hold_ret).dropna()
    if len(excess) < 250:
        return None

    turnover = float(sig.diff().abs().sum())
    return {"excess": excess, "strat": strat_ret.reindex(excess.index),
            "hold": hold_ret.reindex(excess.index), "turnover": turnover,
            "exposure": float(sig.abs().mean())}


def sharpe(r: pd.Series, ppy: float) -> float:
    """Risk-free rate is zero here, so this overstates everything equally.

    It is used only to compare a strategy against buy-and-hold on the same
    bars, and a constant added to both sides cancels out of the comparison.
    """
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def summarise(excess: pd.Series, ppy: float) -> dict:
    n = len(excess)
    mu, sd = excess.mean(), excess.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return {
        "bars": n,
        "excess_ann_pct": float(mu * ppy * 100),
        "t_stat": float(t),
        # Normal approximation. With n in the hundreds the difference from a
        # t-distribution is in the fourth decimal, and it is the same function
        # the research journal already uses, so bars stay comparable.
        "p_value": float(two_sided_p(t)) if np.isfinite(t) else np.nan,
    }


def run(symbols: list[str], cost_bps: float) -> pd.DataFrame:
    bars = load(symbols)
    print(f"{len(bars)} symbols loaded\n")

    names = [n for n, c in REGISTRY.items() if not c.requires_features]
    rows = []
    for i, name in enumerate(sorted(names), 1):
        print(f"[{i:>2}/{len(names)}] {name}")
        for sym, df in bars.items():
            r = evaluate(df, name, cost_bps)
            if r is None:
                continue
            ppy = bars_per_year(df.index)
            ex = r["excess"]
            is_ex = ex[ex.index < SPLIT]
            oos_ex = ex[ex.index >= SPLIT]
            if len(is_ex) < 250 or len(oos_ex) < 250:
                continue

            oos_mask = r["strat"].index >= SPLIT
            s_oos, h_oos = r["strat"][oos_mask], r["hold"][oos_mask]
            row = {"strategy": name, "symbol": sym,
                   "trades": r["turnover"] / 2.0,
                   "exposure": r["exposure"],
                   "full_ann_pct": summarise(ex, ppy)["excess_ann_pct"],
                   "oos_sharpe": sharpe(s_oos, ppy),
                   "oos_hold_sharpe": sharpe(h_oos, ppy),
                   # A rule holding 20% of the time cannot match a fully
                   # invested benchmark on raw return, and condemning it for
                   # that measures exposure rather than skill. Sharpe is the
                   # comparison that survives the difference.
                   "oos_sharpe_gap": sharpe(s_oos, ppy) - sharpe(h_oos, ppy)}
            for label, part in [("is", is_ex), ("oos", oos_ex)]:
                s = summarise(part, ppy)
                row[f"{label}_excess_ann_pct"] = s["excess_ann_pct"]
                row[f"{label}_t"] = s["t_stat"]
                row[f"{label}_p"] = s["p_value"]
            rows.append(row)

    return pd.DataFrame(rows)


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Bonferroni bar and Benjamini-Hochberg FDR on the out-of-sample tests."""
    n = len(df)
    df = df.sort_values("oos_p").reset_index(drop=True)
    df["bonferroni_alpha"] = 0.05 / max(n, 1)
    df["bonferroni_t_bar"] = bonferroni_bar(max(n, 1))
    df["beats_bonferroni"] = (df["oos_p"] < df["bonferroni_alpha"]) & (df["oos_t"] > 0)

    ranks = np.arange(1, n + 1)
    crit = 0.05 * ranks / n
    passing = df["oos_p"].to_numpy() <= crit
    cutoff = np.flatnonzero(passing).max() + 1 if passing.any() else 0
    df["beats_fdr"] = False
    if cutoff:
        df.loc[:cutoff - 1, "beats_fdr"] = df.loc[:cutoff - 1, "oos_t"] > 0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    symbols = QUICK if args.quick else UNIVERSE
    print(f"Universe: {', '.join(symbols)}")
    print(f"Cost: {args.cost_bps} bps round trip   Split: {SPLIT.date()}\n")

    df = run(symbols, args.cost_bps)
    if df.empty:
        print("nothing evaluated")
        return

    dead = df[df["exposure"] <= 0.001]["strategy"].unique()
    if len(dead):
        print(f"\nNot evaluated -- never took a position, so their "
              f"numbers only measure sitting in cash: "
              f"{', '.join(sorted(dead))}")
        df = df[df["exposure"] > 0.001]

    # Buy-and-hold is the benchmark, not a candidate. Its "excess" is
    # 0.0004%/yr -- economically zero -- but the engine enters on the first
    # bar's open rather than its close, which leaves a tiny constant offset
    # with almost no variance. That produces t-stats near 2.7 on nothing at
    # all, and including it would both pollute the multiple-comparison count
    # and let the benchmark beat itself.
    candidates = df[df["strategy"] != "Buy and hold"].copy()
    bench = df[df["strategy"] == "Buy and hold"].copy()
    candidates = adjust(candidates)
    df = pd.concat([candidates, bench], ignore_index=True)
    df.to_csv(OUT / "evaluate_all.csv", index=False)
    df_tested = candidates

    print("\n" + "=" * 88)
    print(f"{len(df_tested)} strategy-symbol pairs tested "
          f"(buy-and-hold excluded -- it is the benchmark). "
          f"Bonferroni bar: p < {df_tested['bonferroni_alpha'].iloc[0]:.6f}, "
          f"|t| > {df_tested['bonferroni_t_bar'].iloc[0]:.2f}")
    print("=" * 88)

    per = (df.groupby("strategy")
             .agg(pairs=("symbol", "count"),
                  median_oos=("oos_excess_ann_pct", "median"),
                  best_oos=("oos_excess_ann_pct", "max"),
                  beat_hold=("oos_excess_ann_pct", lambda s: (s > 0).mean()),
                  median_sharpe_gap=("oos_sharpe_gap", "median"),
                  beat_sharpe=("oos_sharpe_gap", lambda s: (s > 0).mean()),
                  median_exposure=("exposure", "median"),
                  median_trades=("trades", "median"),
                  bonf=("beats_bonferroni", "sum"),
                  fdr=("beats_fdr", "sum"))
             .sort_values("median_oos", ascending=False))
    per.to_csv(OUT / "evaluate_by_strategy.csv")

    print(f"\n{'strategy':<32}{'medOOS':>8}{'beat':>6}{'ShrpGap':>9}{'beat':>6}{'expo':>6}{'trades':>8}{'Bonf':>6}{'FDR':>5}")
    print("-" * 88)
    for name, r in per.iterrows():
        print(f"{name:<32}{r['median_oos']:>7.2f}%{r['beat_hold']:>6.0%}"
              f"{r['median_sharpe_gap']:>9.2f}{r['beat_sharpe']:>6.0%}"
              f"{r['median_exposure']:>6.2f}{r['median_trades']:>8.0f}"
              f"{int(r['bonf']):>6}{int(r['fdr']):>5}")
    print("\nmedOOS  = annualised excess over buy-and-hold, after costs.")
    print("ShrpGap = strategy Sharpe minus buy-and-hold Sharpe, same bars.")
    print("expo    = fraction of the time a position was held.")

    print("\n" + "=" * 88)
    winners = df_tested[df_tested["beats_bonferroni"]]
    print(f"Cleared the corrected bar: {len(winners)} of {len(df_tested)}")
    if len(winners):
        print(winners[["strategy", "symbol", "oos_excess_ann_pct",
                       "oos_t", "oos_p"]].to_string(index=False))
    fdr = df_tested[df_tested["beats_fdr"]]
    print(f"Cleared Benjamini-Hochberg FDR at 5%: {len(fdr)}")
    if len(fdr):
        print(fdr[["strategy", "symbol", "oos_excess_ann_pct",
                   "oos_t"]].head(15).to_string(index=False))

    print(f"\nPairs beating buy-and-hold out of sample at all: "
          f"{(df_tested['oos_excess_ann_pct'] > 0).mean():.1%}")

    # The decisive diagnostic. If these rules predicted anything their wins
    # would be spread across instruments. If they merely hold less of the
    # time, they win exactly where the instrument fell and lose everywhere it
    # rose -- which is dilution, not prediction.
    by_sym = (df_tested.groupby("symbol")
              .agg(beat=("oos_excess_ann_pct", lambda x: (x > 0).mean()),
                   median_excess=("oos_excess_ann_pct", "median"),
                   hold_sharpe=("oos_hold_sharpe", "first"))
              .sort_values("beat", ascending=False))
    by_sym.to_csv(OUT / "evaluate_by_symbol.csv")
    print(f"\nWhere did strategies beat buy-and-hold?")
    print(f"{'symbol':<9}{'beat rate':>11}{'med excess':>12}{'B&H Sharpe':>12}")
    print("-" * 44)
    for sym, r in by_sym.iterrows():
        print(f"{sym:<9}{r['beat']:>10.0%}{r['median_excess']:>11.2f}%"
              f"{r['hold_sharpe']:>12.2f}")
    print(f"\nRead that against the B&H Sharpe column: the rules win on "
          f"the instruments that fell and lose on the ones that rose. "
          f"That is\nlower exposure, not prediction.")
    print(f"Written to {OUT / 'evaluate_all.csv'}")


if __name__ == "__main__":
    main()
