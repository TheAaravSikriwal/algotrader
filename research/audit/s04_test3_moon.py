"""TEST 3 -- NULL CONTROL: lunar phase (Source 11).

Claim: "A new moon means the market tends to be bullish and a full moon means
the market tends to be bearish."

This is the pipeline's null control. If this comes out significant, the
pipeline is broken, not the moon.

Lunar phase method (no ephem/astral in this venv -- stated explicitly):
mean synodic phase from the reference new moon 2000-01-06 18:14 UTC with a
synodic month of 29.530588853 days. Phase angle = 360 * frac((t - t0)/P).
0 deg = new moon, 180 deg = full moon. This mean-phase approximation is
accurate to roughly +/- 0.5 day against true syzygy -- immaterial at a 7-day
window width, and it uses NO market data whatsoever, so there is no lookahead
by construction.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from core.engine import BacktestConfig, run_backtest  # noqa: E402
from core.journal import bonferroni_bar, two_sided_p  # noqa: E402
from core.metrics import equity_stats  # noqa: E402

EPOCH = pd.Timestamp("2000-01-06 18:14:00")
SYNODIC = 29.530588853


def moon_age_days(idx: pd.DatetimeIndex) -> np.ndarray:
    """Days since the most recent mean new moon, in [0, 29.53)."""
    delta = (pd.DatetimeIndex(idx) - EPOCH).total_seconds() / 86400.0
    return np.mod(delta.to_numpy(dtype=float), SYNODIC)


def signed_distance(age: np.ndarray, target_age: float) -> np.ndarray:
    """Signed days from the nearest occurrence of `target_age`, in
    (-SYNODIC/2, SYNODIC/2]."""
    d = age - target_age
    return (d + SYNODIC / 2.0) % SYNODIC - SYNODIC / 2.0


def main():
    specs = 0
    df = A.bars("SPY", "1993-01-01", "2026-09-11", "1Day",
                source="yfinance", eastern=False)
    px = df["close"].astype(float)
    ret = px.pct_change()
    idx = df.index

    age = moon_age_days(idx)
    d_new = signed_distance(age, 0.0)          # 0 at new moon
    d_full = signed_distance(age, SYNODIC / 2) # 0 at full moon

    print(f"SPY daily bars: n={len(df)} {idx[0].date()} .. {idx[-1].date()} "
          f"(yfinance auto_adjust -> total return)")
    print(f"lunar cycles covered: {(idx[-1] - idx[0]).days / SYNODIC:.0f} "
          f"(= independent new-moon events)")

    # ---- 1. window means ---------------------------------------------------
    rows = []
    for half in (3, 5):   # 7-day and 11-day windows centred on the event
        for label, dist in (("new_moon", d_new), ("full_moon", d_full)):
            m = np.abs(dist) <= half
            r = ret[m].dropna()
            t = A.tstat(r)
            rows.append({"window_days": 2 * half + 1, "phase": label,
                         "n_days": len(r), "mean_daily_pct": r.mean() * 100,
                         "std_daily_pct": r.std(ddof=1) * 100,
                         "ann_mean_pct": r.mean() * 252 * 100,
                         "t": t, "p": two_sided_p(t)})
            specs += 1
        # difference of means, new vs full
        a = ret[np.abs(d_new) <= half].dropna()
        b = ret[np.abs(d_full) <= half].dropna()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (a.mean() - b.mean()) / se
        rows.append({"window_days": 2 * half + 1, "phase": "NEW_minus_FULL",
                     "n_days": len(a) + len(b),
                     "mean_daily_pct": (a.mean() - b.mean()) * 100,
                     "std_daily_pct": np.nan,
                     "ann_mean_pct": (a.mean() - b.mean()) * 252 * 100,
                     "t": float(t), "p": two_sided_p(float(t))})
        specs += 1
    stats = pd.DataFrame(rows)
    print("\n=== mean daily return by lunar window (SPY total return) ===")
    print(stats.to_string(index=False))
    A.save(stats, "s04_t3_window_means.csv")

    # block bootstrap on the difference, 20-day blocks, to kill the
    # "overlapping days are not independent" objection
    half = 3
    lab = np.where(np.abs(d_new) <= half, 1, np.where(np.abs(d_full) <= half, -1, 0))
    r = ret.to_numpy()
    ok = ~np.isnan(r)
    rng = np.random.default_rng(11)
    obs = np.nanmean(r[(lab == 1) & ok]) - np.nanmean(r[(lab == -1) & ok])
    n = len(r)
    boot = []
    for _ in range(2000):
        shift = rng.integers(1, n)              # circular shift of the CALENDAR
        lab_s = np.roll(lab, shift)
        a = r[(lab_s == 1) & ok]
        b = r[(lab_s == -1) & ok]
        if len(a) > 10 and len(b) > 10:
            boot.append(np.nanmean(a) - np.nanmean(b))
    boot = np.array(boot)
    pboot = float((np.abs(boot) >= abs(obs)).mean())
    print(f"\ncircular-shift null (2000 draws) on NEW-minus-FULL (7d windows): "
          f"observed {obs*100:+.5f}%/day, null sd {boot.std()*100:.5f}%/day, "
          f"p = {pboot:.3f}")

    # ---- 2. tradeable versions --------------------------------------------
    # signal known at the close of bar t (the moon's phase tomorrow is known
    # centuries in advance), filled at the open of bar t+1 by the engine.
    sig_long = pd.Series(np.where(d_new >= 0, 1.0, 0.0), index=idx)   # new -> full
    sig_ls = pd.Series(np.where(d_new >= 0, 1.0, -1.0), index=idx)
    sig_inv = pd.Series(np.where(d_new >= 0, 0.0, 1.0), index=idx)    # full -> new
    bh = pd.Series(1.0, index=idx)

    variants = {"long_new_to_full": (sig_long, False),
                "SELFTEST_inverted_full_to_new": (sig_inv, False),
                "longshort_new_to_full": (sig_ls, True),
                "buy_and_hold": (bh, False)}

    out = []
    for bps in (0.0, 5.0):
        for name, (s, short) in variants.items():
            cfg = BacktestConfig(initial_cash=10_000.0, slippage_bps=bps,
                                 allow_short=short)
            res = run_backtest(df, s, cfg)
            st = equity_stats(res.equity)
            out.append({"variant": name, "slippage_bps_per_side": bps,
                        "trades": len(res.trades),
                        "CAGR": st["CAGR"], "Sharpe": st["Sharpe"],
                        "MaxDD": st["Max drawdown"],
                        "final": st["Final equity"],
                        "exposure": float((res.exposure != 0).mean())})
            if name != "buy_and_hold":
                specs += 1
    bt = pd.DataFrame(out)
    print("\n=== tradeable lunar backtests (fills at next open, core.engine) ===")
    print(bt.to_string(index=False))
    A.save(bt, "s04_t3_backtests.csv")

    # ---- 3. randomisation self-test ---------------------------------------
    print("\n=== SELF-TEST: randomise the lunar calendar ===")
    rng = np.random.default_rng(3)
    cagrs = []
    for _ in range(50):
        shift = int(rng.integers(1, len(idx)))
        s = pd.Series(np.roll(sig_long.to_numpy(), shift), index=idx)
        res = run_backtest(df, s, BacktestConfig(initial_cash=10_000.0, slippage_bps=0.0))
        cagrs.append(equity_stats(res.equity)["CAGR"])
    real = bt.loc[(bt.variant == "long_new_to_full") & (bt.slippage_bps_per_side == 0),
                  "CAGR"].iloc[0]
    cagrs = np.array(cagrs)
    print(f" real lunar CAGR {real:.4%}; 50 random-phase CAGRs: "
          f"min {cagrs.min():.4%} median {np.median(cagrs):.4%} max {cagrs.max():.4%}; "
          f"percentile of real = {(cagrs < real).mean():.2f}")

    # ---- 4. ^GSPC price-only cross-check over a longer window -------------
    g = A.bars("^GSPC", "1950-01-01", "2026-09-11", "1Day",
               source="yfinance", eastern=False)
    gr = g["close"].astype(float).pct_change()
    ga = moon_age_days(g.index)
    gd_new, gd_full = signed_distance(ga, 0.0), signed_distance(ga, SYNODIC / 2)
    a = gr[np.abs(gd_new) <= 3].dropna()
    b = gr[np.abs(gd_full) <= 3].dropna()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = float((a.mean() - b.mean()) / se)
    print(f"\n^GSPC PRICE ONLY 1950-2026 cross-check (7d windows): "
          f"new {a.mean()*100:+.4f}%/day (n={len(a)}), full {b.mean()*100:+.4f}%/day "
          f"(n={len(b)}), diff t={t:+.3f} p={two_sided_p(t):.3f}, "
          f"cycles={(g.index[-1]-g.index[0]).days/SYNODIC:.0f}")
    specs += 1

    print(f"\nspecifications tried in TEST 3: {specs}")
    print(f"Bonferroni |t| bar at alpha=0.05 for {specs} looks: "
          f"{bonferroni_bar(specs):.3f}")


if __name__ == "__main__":
    main()
