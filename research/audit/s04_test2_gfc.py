"""TEST 2 -- Source 04's headline back-test claim.

Claim (verbatim): "$1,000/month starting 1 July 2007 (just before the GFC) ends
at $247,000; starting February 2009 (timing the bottom perfectly) ends at
$223,000 ... almost a 25 grand difference."

The 2007 investor makes 19 more deposits than the 2009 investor. The whole
exercise is to separate "starting before the crash was fine" from "I put in
$19,000 more money".
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from s04_test1_dca import irr_annual  # noqa: E402

DEPOSIT = 1000.0
END = "2026-09-11"


def load(symbol: str, start: str) -> pd.DataFrame:
    return A.bars(symbol, start, END, "1Day", source="yfinance", eastern=False)


def monthly(df: pd.DataFrame, when: str) -> pd.Series:
    """when='close' -> last trading day of the month; 'open' -> first trading
    day of the month, bought at that day's open (the literal '1 July 2007')."""
    if when == "close":
        return df["close"].resample("ME").last().dropna()
    first = df["open"].resample("MS").first().dropna()
    return first


def dca(px: pd.Series, start: str, end: str, n_max: int | None = None) -> dict:
    s = px.loc[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    if n_max is not None:
        s = s.iloc[:n_max]
    if s.empty:
        return {}
    shares = (DEPOSIT / s.astype(float)).cumsum()
    # value the book at the LAST price in the window
    last = float(s.iloc[-1])
    terminal = float(shares.iloc[-1]) * last
    n = len(s)
    return {"n_deposits": n, "contributed": DEPOSIT * n, "terminal": terminal,
            "per_dollar": terminal / (DEPOSIT * n),
            "irr_ann": irr_annual(n, terminal),
            "first": str(s.index[0].date()), "last": str(s.index[-1].date())}


def dca_stop(px: pd.Series, start: str, end: str, n_deposits: int) -> dict:
    """Deposit for n_deposits months from `start`, then STOP contributing and
    stay invested until `end`. Equalises capital AND the valuation date."""
    s = px.loc[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    dep = s.iloc[:n_deposits]
    shares = float((DEPOSIT / dep.astype(float)).sum())
    terminal = shares * float(s.iloc[-1])
    return {"n_deposits": len(dep), "contributed": DEPOSIT * len(dep),
            "terminal": terminal, "per_dollar": terminal / (DEPOSIT * len(dep)),
            "last": str(s.index[-1].date())}


def main():
    series = {
        "VFINX_TR": load("VFINX", "2006-01-01"),
        "SPY_TR": load("SPY", "2006-01-01"),
        "GSPC_PRICE_ONLY": load("^GSPC", "2006-01-01"),
    }

    # ---- 1. hunt for the end date that reproduces 247k / 223k --------------
    hunt = []
    for name, df in series.items():
        for when in ("close", "open"):
            px = monthly(df, when)
            for end in pd.date_range("2016-01-31", "2026-09-30", freq="ME"):
                a = dca(px, "2007-07-01", str(end.date()))
                b = dca(px, "2009-02-01", str(end.date()))
                if not a or not b:
                    continue
                hunt.append({
                    "series": name, "deposit_at": when, "end": str(end.date())[:7],
                    "t2007": a["terminal"], "t2009": b["terminal"],
                    "diff": a["terminal"] - b["terminal"],
                    "n2007": a["n_deposits"], "n2009": b["n_deposits"],
                    "ratio": a["terminal"] / b["terminal"],
                    "err": abs(a["terminal"] - 247000) + abs(b["terminal"] - 223000),
                })
    h = pd.DataFrame(hunt)
    h["err_2007_only"] = (h["t2007"] - 247000).abs()
    h["err_ratio"] = (h["ratio"] - 247.0 / 223.0).abs()
    A.save(h, "s04_t2_enddate_hunt.csv")
    pd.set_option("display.width", 250)
    print("=== closest joint reproductions of (247k, 223k) ===")
    print(h.nsmallest(10, "err").to_string(index=False))
    print("\n=== end dates where the 2007 leg alone hits ~247k ===")
    print(h.nsmallest(10, "err_2007_only").to_string(index=False))
    print(f"\n=== closest to the claimed RATIO 247/223 = {247/223:.4f} ===")
    print(h.nsmallest(8, "err_ratio").to_string(index=False))
    print(f"\nratio t2007/t2009 across ALL specs: min={h['ratio'].min():.4f} "
          f"median={h['ratio'].median():.4f} max={h['ratio'].max():.4f}")

    best = h.nsmallest(1, "err").iloc[0]
    print(f"\nbest match: {best['series']} deposit_at={best['deposit_at']} "
          f"end={best['end']}  2007->{best['t2007']:,.0f}  2009->{best['t2009']:,.0f}")

    # ---- 2. the three comparisons at a set of plausible end dates ---------
    rows = []
    for name, df in series.items():
        px = monthly(df, "close")
        for end in ("2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31",
                    "2024-12-31", "2025-12-31", "2026-08-31",
                    str(best['end']) + "-28"):
            a = dca(px, "2007-07-01", end)
            b = dca(px, "2009-02-01", end)
            if not a or not b:
                continue
            # fair comparison A: 2007 investor makes only as many deposits as
            # the 2009 investor, then holds to the SAME end date. Equal capital,
            # equal valuation date; the 2007 investor still gets more duration.
            a_trunc = dca_stop(px, "2007-07-01", end, b["n_deposits"])
            # fair comparison B: 2009 investor keeps paying 19 months longer
            # (same number of payments, later start, later finish)
            px_ext = px.loc[px.index >= pd.Timestamp("2009-02-01")]
            b_ext = dca(px_ext, "2009-02-01", "2026-09-30", n_max=a["n_deposits"])
            rows.append({
                "series": name, "end": end,
                "n2007": a["n_deposits"], "n2009": b["n_deposits"],
                "extra_capital_2007": a["contributed"] - b["contributed"],
                "term2007": a["terminal"], "term2009": b["terminal"],
                "diff": a["terminal"] - b["terminal"],
                "perdollar2007": a["per_dollar"], "perdollar2009": b["per_dollar"],
                "irr2007": a["irr_ann"], "irr2009": b["irr_ann"],
                "fairA_2007_trunc_term": a_trunc["terminal"] if a_trunc else np.nan,
                "fairA_2007_trunc_n": a_trunc["n_deposits"] if a_trunc else np.nan,
                "fairB_2009_ext_term": b_ext["terminal"] if b_ext else np.nan,
                "fairB_2009_ext_n": b_ext["n_deposits"] if b_ext else np.nan,
                "fairB_2009_ext_last": b_ext["last"] if b_ext else "",
            })
    r = pd.DataFrame(rows)
    A.save(r, "s04_t2_comparisons.csv")
    pd.set_option("display.width", 250)
    print("\n=== terminal wealth / per-dollar / IRR ===")
    print(r[["series", "end", "n2007", "n2009", "term2007", "term2009", "diff",
             "perdollar2007", "perdollar2009", "irr2007", "irr2009"]].to_string(index=False))
    print("\n=== fair (equal-deposit-count) comparisons ===")
    print(r[["series", "end", "term2009", "fairA_2007_trunc_n", "fairA_2007_trunc_term",
             "fairB_2009_ext_n", "fairB_2009_ext_term", "fairB_2009_ext_last",
             "term2007"]].to_string(index=False))

    # ---- 3. how much of the gap is just the extra 19,000? -----------------
    print("\n=== decomposition at each end date (VFINX_TR) ===")
    px = monthly(series["VFINX_TR"], "close")
    dec = []
    for end in ("2020-12-31", "2022-12-31", "2024-12-31", "2026-08-31"):
        a = dca(px, "2007-07-01", end)
        b = dca(px, "2009-02-01", end)
        at = dca_stop(px, "2007-07-01", end, b["n_deposits"])
        dec.append({"end": end, "gap_headline": a["terminal"] - b["terminal"],
                    "extra_capital": a["contributed"] - b["contributed"],
                    "gap_same_capital_same_date": at["terminal"] - b["terminal"],
                    "value_of_19_extra_deposits": a["terminal"] - at["terminal"],
                    "perdollar_2007_trunc": at["per_dollar"],
                    "perdollar_2009": b["per_dollar"]})
    d = pd.DataFrame(dec)
    print(d.to_string(index=False))
    A.save(d, "s04_t2_decomposition.csv")

    # ---- 3b. every start month 2005-2012, held to 2026-08 -----------------
    print("\n=== IRR by start month, all held to 2026-08 (VFINX TR) ===")
    px_long = monthly(load("VFINX", "2000-01-01"), "close")
    sw = []
    for st in pd.date_range("2005-01-01", "2012-12-01", freq="MS"):
        a = dca(px_long, str(st.date()), "2026-08-31")
        sw.append({"start": str(st.date())[:7], "n": a["n_deposits"],
                   "terminal": a["terminal"], "per_dollar": a["per_dollar"],
                   "irr": a["irr_ann"]})
    s = pd.DataFrame(sw)
    A.save(s, "s04_t2_startmonth_sweep.csv")
    print(s.iloc[::3].to_string(index=False))
    print(f"\nIRR rank of 2007-07 among the 96 start months: "
          f"{int((s['irr'] > s.loc[s['start'] == '2007-07', 'irr'].iloc[0]).sum()) + 1} "
          f"(1 = best). IRR rank of 2009-02: "
          f"{int((s['irr'] > s.loc[s['start'] == '2009-02', 'irr'].iloc[0]).sum()) + 1}")
    print(f"terminal-wealth rank of 2007-07: "
          f"{int((s['terminal'] > s.loc[s['start'] == '2007-07', 'terminal'].iloc[0]).sum()) + 1}")

    # ---- 4. self-test: swap the start dates, answer must flip -------------
    print("\n=== SELF-TEST ===")
    px = monthly(series["VFINX_TR"], "close")
    a = dca(px, "2007-07-01", "2026-08-31")
    b = dca(px, "2009-02-01", "2026-08-31")
    print(f" 2007 start: n={a['n_deposits']} term={a['terminal']:,.0f} irr={a['irr_ann']:.4%}")
    print(f" 2009 start: n={b['n_deposits']} term={b['terminal']:,.0f} irr={b['irr_ann']:.4%}")
    # a start with NO crash in it at all, same deposit count as 2009
    c = dca(px, "2013-01-01", "2026-08-31", n_max=b["n_deposits"])
    print(f" 2013 start (no crash), n capped to {b['n_deposits']}: "
          f"term={c['terminal']:,.0f} irr={c['irr_ann']:.4%}")
    # sanity: doubling the deposit must double terminal exactly
    shares = (2 * DEPOSIT / px.loc["2007-07-01":"2026-08-31"].astype(float)).cumsum()
    dbl = float(shares.iloc[-1]) * float(px.loc[:"2026-08-31"].iloc[-1])
    print(f" doubling the deposit doubles terminal? {abs(dbl - 2*a['terminal']) < 1e-6}")


if __name__ == "__main__":
    main()
