"""Translate a per-trade edge into account-level expectations and a stopping rule.

Two questions the eleven sources never ask and that decide everything:

  1. What is a per-trade edge of X bps actually WORTH on a small account, once
     position sizing, leverage and cost are taken into account?
  2. How many trades before you can tell that edge from zero? A strategy whose
     edge needs 4,000 trades to become significant cannot be validated by a
     retail trader in a paper-trading window, no matter how real it is.

(2) is the number that should govern the paper-trading plan, and it is derived
from the observed per-trade standard deviation, not from optimism.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auditlib as A  # noqa: E402


def required_n(mean: float, sd: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Trades needed for a two-sided test at `alpha` to detect `mean` with `power`."""
    if mean == 0:
        return float("inf")
    z_a, z_b = 1.959964, 0.841621          # 0.975 and 0.80 normal quantiles
    return ((z_a + z_b) * sd / abs(mean)) ** 2


def account_view(mean_ret_pct: float, sd_ret_pct: float, median_risk_pct: float,
                 trades_per_year: float, risk_frac: float, cost_bps: float,
                 label: str) -> dict:
    """mean_ret_pct / median_risk_pct are % of NOTIONAL; risk_frac is % of EQUITY."""
    # notional you must carry to put `risk_frac` of equity at risk
    leverage = risk_frac / (median_risk_pct / 100.0)
    gross_per_trade_eq = mean_ret_pct / 100.0 * leverage
    cost_per_trade_eq = cost_bps / 1e4 * leverage
    net_per_trade_eq = gross_per_trade_eq - cost_per_trade_eq
    return {
        "label": label,
        "mean_bps_notional": mean_ret_pct * 100,
        "breakeven_bps": mean_ret_pct * 100,
        "cost_bps": cost_bps,
        "median_risk_pct_notional": median_risk_pct,
        "risk_pct_equity": risk_frac * 100,
        "implied_leverage": leverage,
        "gross_pct_equity_per_trade": gross_per_trade_eq * 100,
        "net_pct_equity_per_trade": net_per_trade_eq * 100,
        "trades_per_year": trades_per_year,
        "net_annual_pct": net_per_trade_eq * trades_per_year * 100,
        "required_n_80pct_power": required_n(mean_ret_pct - cost_bps / 100.0, sd_ret_pct),
    }


def main():
    # Source 10's best-populated cell, measured: see results/s10_grid_5Min.csv
    trades = pd.read_csv(A.RESULTS / "s10_trades_diag_5Min.csv")
    m = trades["ret_pct"].mean()
    sd = trades["ret_pct"].std(ddof=1)
    risk = trades["risk_pct_of_px"].median()
    n = len(trades)
    per_year = n / 10 / (1389 / 252)      # per symbol per year

    print(f"source-10 model, measured: n={n}, mean={m*100:.2f} bps of notional, "
          f"sd={sd*100:.1f} bps, median risk={risk:.3f}% of notional")
    print(f"trades per symbol per year = {per_year:.0f}\n")

    rows = []
    for cost in (0.0, 1.0, 2.0, 3.0, 5.0):
        rows.append(account_view(m, sd, risk, per_year, 0.01, cost,
                                 f"risk 1% of equity, cost {cost:.0f} bps"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out[["label", "mean_bps_notional", "cost_bps", "implied_leverage",
               "net_pct_equity_per_trade", "trades_per_year", "net_annual_pct"]]
          .to_string(index=False))

    print("\n--- how long before you could TELL, at 80% power, two-sided 0.05 ---")
    for cost in (0.0, 1.0, 2.0, 3.0):
        need = required_n(m - cost / 100.0, sd)
        print(f"  cost {cost:.0f} bps: edge {m*100 - cost:+.2f} bps, "
              f"sd {sd*100:.0f} bps -> need {need:,.0f} trades "
              f"= {need/per_year:,.1f} years on one symbol, "
              f"{need/(per_year*10):,.1f} years on ten")
    A.save(out, "economics_s10.csv")


if __name__ == "__main__":
    main()
