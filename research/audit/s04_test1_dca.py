"""TEST 1 -- Source 04's "Advanced DCA" valuation overlay.

Baseline: $1,000/month into an S&P 500 index fund.
Overlay:  dev = price / "50-year mean growth of the market" - 1
            dev > +20%  -> invest 500, save 500
            dev > +10%  -> invest 750, save 250
            dev < -10%  -> invest 1000 AND deploy the whole reserve evenly
                           over the next 6 monthly deposits.

"50-year mean growth" is never defined by the source. Four readings are tested;
(d) is deliberately non-implementable and is reported only as an upper bound.

Causality: the decision made for month m uses the trend/price observed at the
close of month m-1 (dev is lagged one month). The deposit is made at the close
of month m. Nothing in the signal uses a price from month m or later.

Total cash OUT of the investor's pocket is $1,000/month in BOTH arms -- the
overlay does not contribute less, it parks part of the contribution in cash.
So terminal wealth and IRR are ranked identically and the comparison is
apples-to-apples on contributions.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

END = "2026-09-11"
DEPOSIT = 1000.0


# ---------------------------------------------------------------- data -----
def monthly_close(symbol: str, start: str) -> pd.Series:
    df = A.bars(symbol, start, END, "1Day", source="yfinance", eastern=False)
    s = df["close"].resample("ME").last().dropna()
    s.name = symbol
    return s


def tbill_monthly_rate() -> pd.Series:
    """Monthly simple rate on 3-month T-bills. DTB3 (daily, 1990+) spliced onto
    TB3MS (monthly, 1934+) for the earlier years."""
    dtb3 = pd.read_csv(A.CACHE / "fred_DTB3.csv", index_col=0, parse_dates=True).iloc[:, 0]
    tb3ms = pd.read_csv(A.CACHE / "fred_TB3MS.csv", index_col=0, parse_dates=True).iloc[:, 0]
    d_m = dtb3.resample("ME").mean()
    t_m = tb3ms.resample("ME").mean()
    ann = d_m.combine_first(t_m)        # DTB3 wins wherever it exists
    ann = ann.sort_index().dropna()
    return ann / 100.0 / 12.0


# --------------------------------------------------------------- trends ----
def trend_loglin(prices: pd.Series, window: int, min_obs: int = 120) -> pd.Series:
    """Exponential trend fitted to log price over a TRAILING window, refit each
    month, evaluated at the last point of that window. Strictly causal."""
    lp = np.log(prices.to_numpy(dtype=float))
    n = len(lp)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        y = lp[lo:i + 1]
        k = len(y)
        if k < min_obs:
            continue
        x = np.arange(k, dtype=float)
        b, a = np.polyfit(x, y, 1)
        out[i] = np.exp(a + b * (k - 1))
    return pd.Series(out, index=prices.index)


def trend_logma(prices: pd.Series, window: int, min_obs: int = 120) -> pd.Series:
    """exp(trailing mean of log price). Causal."""
    lp = np.log(prices.astype(float))
    m = lp.rolling(window, min_periods=min_obs).mean()
    return np.exp(m)


def trend_fullsample(prices: pd.Series) -> pd.Series:
    """LOOKAHEAD -- not implementable. Full-sample log-linear fit."""
    lp = np.log(prices.to_numpy(dtype=float))
    x = np.arange(len(lp), dtype=float)
    b, a = np.polyfit(x, lp, 1)
    return pd.Series(np.exp(a + b * x), index=prices.index)


# ------------------------------------------------------------ simulation ---
def simulate(prices: pd.Series, dev_lag: pd.Series | None,
             rate: pd.Series | None) -> dict:
    """prices: month-end total-return NAV of the fund actually bought.
    dev_lag: deviation-from-mean KNOWN BEFORE the month's deposit (already
             shifted). None => plain DCA.
    rate:    monthly simple interest on the cash reserve. None => 0%."""
    shares = 0.0
    reserve = 0.0
    deploy_left = 0
    deploy_amt = 0.0
    wealth = []
    contributed = 0.0
    months_in_reserve = 0
    n_trig = 0
    reserve_path = []

    for ts, px in prices.items():
        if rate is not None and reserve > 0:
            reserve *= 1.0 + float(rate.get(ts, 0.0))

        invest, save = DEPOSIT, 0.0
        d = np.nan if dev_lag is None else float(dev_lag.get(ts, np.nan))
        if dev_lag is not None and not np.isnan(d):
            if d > 0.20:
                invest, save = 500.0, 500.0
            elif d > 0.10:
                invest, save = 750.0, 250.0
            if d < -0.10 and deploy_left == 0 and reserve > 1e-9:
                deploy_amt = reserve / 6.0
                deploy_left = 6
                n_trig += 1

        extra = 0.0
        if deploy_left > 0:
            extra = min(deploy_amt, reserve)
            reserve -= extra
            deploy_left -= 1

        reserve += save
        if save > 0:
            months_in_reserve += 1
        contributed += DEPOSIT
        shares += (invest + extra) / float(px)
        w = shares * float(px) + reserve
        wealth.append(w)
        reserve_path.append(reserve)

    w = pd.Series(wealth, index=prices.index)
    res = pd.Series(reserve_path, index=prices.index)
    dd = (w / w.cummax() - 1.0).min()
    return {
        "terminal": float(w.iloc[-1]),
        "contributed": contributed,
        "months": len(prices),
        "irr_ann": irr_annual(len(prices), float(w.iloc[-1])),
        "max_dd": float(dd),
        "wealth": w,
        "reserve_end": float(res.iloc[-1]),
        "reserve_max": float(res.max()),
        "months_saving": months_in_reserve,
        "n_triggers": n_trig,
        "mult": float(w.iloc[-1]) / contributed,
    }


def irr_annual(n_months: int, terminal: float) -> float:
    """Money-weighted return: n_months deposits of $1,000, terminal value paid
    out with the last one. Bisection on the monthly rate."""
    def f(r):
        """terminal minus the future value of the deposit annuity. Strictly
        decreasing in r, so a plain bisection is safe."""
        if abs(r) < 1e-12:
            fv = DEPOSIT * n_months
        else:
            fv = DEPOSIT * ((1.0 + r) ** n_months - 1.0) / r
        return terminal - fv

    lo, hi = -0.90, 0.20
    if f(lo) < 0 or f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    r = (lo + hi) / 2
    return float((1.0 + r) ** 12 - 1.0)


# ------------------------------------------------------------------ main ---
def build_devs(sig: pd.Series) -> dict[str, pd.Series]:
    return {
        "a_loglin600": sig / trend_loglin(sig, 600) - 1.0,
        "b_logMA600": sig / trend_logma(sig, 600) - 1.0,
        "c_loglin120": sig / trend_loglin(sig, 120) - 1.0,
        "d_LOOKAHEAD_fullsample": sig / trend_fullsample(sig) - 1.0,
    }


def main():
    rate = tbill_monthly_rate()

    # investable series: VFINX total return (yfinance auto_adjust) from 1980
    inv = monthly_close("VFINX", "1980-01-01")
    # signal series: ^GSPC PRICE from 1927 so a genuine trailing 600-month
    # window exists for every investing month. Price-only is the right series
    # for a "distance above the mean of the chart" rule; it is also the only
    # one with 50 years of history before 1980.
    sig = monthly_close("^GSPC", "1927-12-01")

    devs = build_devs(sig)
    idx = inv.index
    print(f"investable VFINX months: {len(inv)} {idx[0].date()} .. {idx[-1].date()}")
    for k, v in devs.items():
        vv = v.reindex(idx).dropna()
        print(f"  dev {k}: {len(vv)} usable months, "
              f"min={vv.min():+.3f} med={vv.median():+.3f} max={vv.max():+.3f}")

    rows = []
    # --- full history ------------------------------------------------------
    base_free = simulate(inv, None, None)
    for cash_label, rt in (("0pct", None), ("tbill", rate)):
        base = simulate(inv, None, rt)   # identical to base_free; no reserve
        rows.append(dict(test="full_history", reading="plain_DCA", cash=cash_label,
                         **{k: base[k] for k in
                            ("terminal", "contributed", "months", "irr_ann",
                             "max_dd", "mult", "reserve_end", "reserve_max",
                             "months_saving", "n_triggers")}))
        for name, dv in devs.items():
            r = simulate(inv, dv.reindex(idx).shift(1), rt)
            r["beat"] = r["terminal"] > base["terminal"]
            rows.append(dict(test="full_history", reading=name, cash=cash_label,
                             beat_plain=bool(r["beat"]),
                             ratio=r["terminal"] / base["terminal"],
                             **{k: r[k] for k in
                                ("terminal", "contributed", "months", "irr_ann",
                                 "max_dd", "mult", "reserve_end", "reserve_max",
                                 "months_saving", "n_triggers")}))
    full = pd.DataFrame(rows)
    print("\n=== FULL HISTORY (VFINX total return, 1980-01 .. 2026-09) ===")
    print(full[["reading", "cash", "terminal", "irr_ann", "max_dd", "mult",
                "months_saving", "n_triggers", "ratio"]].to_string(index=False))
    A.save(full, "s04_t1_full_history.csv")

    # --- rolling windows ---------------------------------------------------
    roll_rows = []
    for L, tag in ((240, "20y"), (360, "30y")):
        for cash_label, rt in (("0pct", None), ("tbill", rate)):
            for name, dv in devs.items():
                recs = []
                dvl = dv.reindex(idx).shift(1)
                for s in range(0, len(idx) - L + 1):
                    sl = idx[s:s + L]
                    b = simulate(inv.loc[sl], None, rt)
                    o = simulate(inv.loc[sl], dvl.loc[sl], rt)
                    recs.append({
                        "start": str(sl[0].date()), "end": str(sl[-1].date()),
                        "plain_terminal": b["terminal"], "ovl_terminal": o["terminal"],
                        "plain_irr": b["irr_ann"], "ovl_irr": o["irr_ann"],
                        "plain_dd": b["max_dd"], "ovl_dd": o["max_dd"],
                        "ratio": o["terminal"] / b["terminal"],
                        "beat": o["terminal"] > b["terminal"],
                        "n_triggers": o["n_triggers"],
                        "months_saving": o["months_saving"],
                    })
                d = pd.DataFrame(recs)
                if d.empty:
                    continue
                d.to_csv(A.RESULTS / f"s04_t1_roll_{tag}_{name}_{cash_label}.csv", index=False)
                roll_rows.append({
                    "window": tag, "reading": name, "cash": cash_label,
                    "n_start_months": len(d),
                    "n_independent": len(idx) // L,
                    "frac_beat": float(d["beat"].mean()),
                    "ratio_p05": float(d["ratio"].quantile(0.05)),
                    "ratio_median": float(d["ratio"].median()),
                    "ratio_mean": float(d["ratio"].mean()),
                    "ratio_p95": float(d["ratio"].quantile(0.95)),
                    "ratio_min": float(d["ratio"].min()),
                    "ratio_max": float(d["ratio"].max()),
                    "irr_diff_median_bps": float((d["ovl_irr"] - d["plain_irr"]).median() * 1e4),
                    "dd_plain_median": float(d["plain_dd"].median()),
                    "dd_ovl_median": float(d["ovl_dd"].median()),
                })
    roll = pd.DataFrame(roll_rows)
    print("\n=== ROLLING START DATES (overlay / plain terminal wealth) ===")
    print(roll.to_string(index=False))
    A.save(roll, "s04_t1_rolling_summary.csv")

    # --- non-overlapping (genuinely independent) ---------------------------
    ind_rows = []
    for L, tag in ((240, "20y"), (360, "30y")):
        starts = list(range(0, len(idx) - L + 1, L))
        for name, dv in devs.items():
            dvl = dv.reindex(idx).shift(1)
            for s in starts:
                sl = idx[s:s + L]
                b = simulate(inv.loc[sl], None, rate)
                o = simulate(inv.loc[sl], dvl.loc[sl], rate)
                ind_rows.append({"window": tag, "reading": name,
                                 "start": str(sl[0].date()), "end": str(sl[-1].date()),
                                 "plain": b["terminal"], "overlay": o["terminal"],
                                 "ratio": o["terminal"] / b["terminal"]})
    ind = pd.DataFrame(ind_rows)
    print("\n=== NON-OVERLAPPING PERIODS (T-bill reserve) ===")
    print(ind.to_string(index=False))
    A.save(ind, "s04_t1_nonoverlapping.csv")

    # --- self-test: invert the rule ----------------------------------------
    print("\n=== SELF-TEST: invert the overlay (buy MORE when expensive) ===")
    dv = devs["c_loglin120"].reindex(idx).shift(1)
    base = simulate(inv, None, rate)
    normal = simulate(inv, dv, rate)
    inverted = simulate(inv, -dv, rate)
    shuffled = simulate(inv, pd.Series(
        np.random.default_rng(7).permutation(dv.to_numpy()), index=idx), rate)
    print(f" plain    terminal = {base['terminal']:,.0f}")
    print(f" overlay  terminal = {normal['terminal']:,.0f}  ({normal['terminal']/base['terminal']:.4f}x)")
    print(f" INVERTED terminal = {inverted['terminal']:,.0f}  ({inverted['terminal']/base['terminal']:.4f}x)")
    print(f" SHUFFLED terminal = {shuffled['terminal']:,.0f}  ({shuffled['terminal']/base['terminal']:.4f}x)")

    # --- long price-only history (upper bound on sample size) --------------
    print("\n=== LONG HISTORY, ^GSPC PRICE ONLY (no dividends; both arms) ===")
    long_inv = sig.loc["1933-01-01":]
    long_devs = build_devs(sig)
    lrows = []
    for L, tag in ((240, "20y"), (360, "30y")):
        for name in ("a_loglin600", "c_loglin120"):
            dvl = long_devs[name].reindex(long_inv.index).shift(1)
            recs = []
            li = long_inv.index
            for s in range(0, len(li) - L + 1):
                sl = li[s:s + L]
                if dvl.loc[sl].isna().all():
                    continue
                b = simulate(long_inv.loc[sl], None, rate)
                o = simulate(long_inv.loc[sl], dvl.loc[sl], rate)
                recs.append({"start": str(sl[0].date()), "ratio": o["terminal"] / b["terminal"],
                             "beat": o["terminal"] > b["terminal"]})
            d = pd.DataFrame(recs)
            if d.empty:
                continue
            lrows.append({"window": tag, "reading": name, "n_start_months": len(d),
                          "n_independent": len(li) // L,
                          "frac_beat": float(d["beat"].mean()),
                          "ratio_median": float(d["ratio"].median()),
                          "ratio_p05": float(d["ratio"].quantile(0.05)),
                          "ratio_p95": float(d["ratio"].quantile(0.95))})
    lf = pd.DataFrame(lrows)
    print(lf.to_string(index=False))
    A.save(lf, "s04_t1_longhistory_priceonly.csv")


if __name__ == "__main__":
    main()
