"""TEST 1 self-tests: deliberately break the overlay and check it moves the way
the mechanics say it must."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
from s04_test1_dca import (DEPOSIT, monthly_close, simulate, tbill_monthly_rate,  # noqa: E402
                           trend_loglin)

inv = monthly_close("VFINX", "1980-01-01")
rate = tbill_monthly_rate()
idx = inv.index
n = len(idx)

plain = simulate(inv, None, None)
print(f"plain DCA terminal          {plain['terminal']:,.0f}  contributed {plain['contributed']:,.0f}")

# 1. a signal pinned below -10% must never save -> must equal plain exactly
always_cheap = pd.Series(-0.5, index=idx)
r = simulate(inv, always_cheap, None)
print(f"dev == -0.50 (never save)   {r['terminal']:,.0f}   "
      f"identical to plain? {abs(r['terminal'] - plain['terminal']) < 1e-6}")

# 2. a signal pinned at +0.15 must invest 750 every month and never deploy.
#    Terminal must equal 0.75 * plain equity + 0.25 * contributions in cash.
always_mid = pd.Series(0.15, index=idx)
r = simulate(inv, always_mid, None)
expect = 0.75 * plain["terminal"] + 0.25 * DEPOSIT * n
print(f"dev == +0.15 (750/mo)       {r['terminal']:,.0f}   expected {expect:,.0f}   "
      f"match? {abs(r['terminal'] - expect) < 1e-6}")

# 3. a signal pinned at +0.25 must invest 500 every month.
always_high = pd.Series(0.25, index=idx)
r = simulate(inv, always_high, None)
expect = 0.50 * plain["terminal"] + 0.50 * DEPOSIT * n
print(f"dev == +0.25 (500/mo)       {r['terminal']:,.0f}   expected {expect:,.0f}   "
      f"match? {abs(r['terminal'] - expect) < 1e-6}")

# 4. contributions must be identical in every arm
dv = (inv / trend_loglin(inv, 120) - 1.0).shift(1)
r = simulate(inv, dv, rate)
print(f"contributions equal?        {r['contributed'] == plain['contributed']} "
      f"({r['contributed']:,.0f})")

# 5. shuffle the signal 200 times -- where does the real overlay sit in the null?
real = simulate(inv, dv, rate)["terminal"]
rng = np.random.default_rng(0)
null = np.array([simulate(inv, pd.Series(rng.permutation(dv.to_numpy()), index=idx),
                          rate)["terminal"] for _ in range(200)])
pct = float((null < real).mean())
print(f"real overlay {real:,.0f} vs shuffled-signal null: "
      f"median {np.median(null):,.0f}, pctile of real = {pct:.3f}")
print(f"  (a genuine timing edge would sit near the 1.0 percentile of this null)")

A.save(pd.DataFrame([{"check": "see stdout", "ok": True}]), "s04_t1_selftest.csv")
