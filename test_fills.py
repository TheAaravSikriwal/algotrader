"""Fill-quality measurement.

The point of this log is to catch a backtest assumption that does not survive
contact with a real venue, so these tests use cases where the true answer is
known by construction and check the measurement against it.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from core.fills import FillLog, FillRecord, mid, summarise, verdict


def _rec(**kw):
    base = dict(symbol="SPY", side="buy", qty=100.0, reference_price=100.0)
    base.update(kw)
    return FillRecord(**base)


# -- sign convention ------------------------------------------------------

def test_buying_above_the_reference_is_a_cost():
    r = _rec(side="buy", filled_qty=100, filled_price=100.10)
    assert r.slippage_bps == pytest.approx(10.0)


def test_selling_below_the_reference_is_the_same_cost():
    """Both sides must report adverse as positive.

    Without the flip a book of buys and sells nets to roughly zero and the
    strategy looks free to trade.
    """
    r = _rec(side="sell", filled_qty=100, filled_price=99.90)
    assert r.slippage_bps == pytest.approx(10.0)


def test_price_improvement_is_negative_on_both_sides():
    buy = _rec(side="buy", filled_qty=100, filled_price=99.95)
    sell = _rec(side="sell", filled_qty=100, filled_price=100.05)
    assert buy.slippage_bps == pytest.approx(-5.0)
    assert sell.slippage_bps == pytest.approx(-5.0)


def test_mixed_book_does_not_cancel_to_zero():
    """The failure this convention exists to prevent.

    A buy 10 bps too high and a sell 10 bps too low both cost 10 bps. On raw
    price differences they would average to zero.
    """
    df = pd.DataFrame([
        {**vars(_rec(side="buy", filled_qty=100, filled_price=100.10)),
         "slippage_bps": _rec(side="buy", filled_qty=100, filled_price=100.10).slippage_bps,
         "fill_ratio": 1.0, "notional": 10010.0},
        {**vars(_rec(side="sell", filled_qty=100, filled_price=99.90)),
         "slippage_bps": _rec(side="sell", filled_qty=100, filled_price=99.90).slippage_bps,
         "fill_ratio": 1.0, "notional": 9990.0},
    ])
    assert summarise(df)["slippage_bps_mean"] == pytest.approx(10.0)


def test_effective_spread_is_double_the_one_way_slip():
    r = _rec(filled_qty=100, filled_price=100.02)
    assert r.slippage_bps == pytest.approx(2.0)
    assert r.effective_spread_bps == pytest.approx(4.0)


# -- fills that did not happen -------------------------------------------

def test_an_unfilled_order_has_no_slippage_rather_than_zero():
    """Zero would read as a free trade and drag the average down.

    Unfilled limits are the common case for a resting order, so scoring them
    as costless would make a passive strategy look cheaper the worse its fill
    rate got.
    """
    assert _rec(order_type="limit", limit_price=99.0).slippage_bps is None


def test_unfilled_orders_still_count_against_the_fill_rate():
    log_rows = [
        _rec(filled_qty=100, filled_price=100.01),
        _rec(order_type="limit", limit_price=99.0),
        _rec(order_type="limit", limit_price=99.0),
    ]
    df = pd.DataFrame([{**vars(r), "slippage_bps": r.slippage_bps,
                        "fill_ratio": r.fill_ratio, "notional": r.notional}
                       for r in log_rows])
    s = summarise(df)
    assert s["orders"] == 3
    assert s["filled_orders"] == 1
    assert s["fill_rate"] == pytest.approx(1 / 3)


def test_partial_fills_are_scored_as_partial():
    r = _rec(qty=100, filled_qty=40, filled_price=100.05)
    assert r.fill_ratio == pytest.approx(0.4)


def test_overfill_cannot_exceed_one():
    assert _rec(qty=100, filled_qty=110, filled_price=100.0).fill_ratio == 1.0


# -- weighting ------------------------------------------------------------

def test_notional_weighting_exposes_bad_big_orders():
    """Equal weighting is the flattering number when size fills worst.

    A small clean fill and a large ugly one average to something mild per
    order, but the money was spent at the ugly price.
    """
    small = _rec(qty=10, filled_qty=10, filled_price=100.00)     # 0 bps
    big = _rec(qty=1000, filled_qty=1000, filled_price=100.20)   # 20 bps
    df = pd.DataFrame([{**vars(r), "slippage_bps": r.slippage_bps,
                        "fill_ratio": r.fill_ratio, "notional": r.notional}
                       for r in (small, big)])
    s = summarise(df)
    assert s["slippage_bps_mean"] == pytest.approx(10.0)
    assert s["slippage_bps_notional_weighted"] > 19.0


# -- quotes ---------------------------------------------------------------

def test_mid_of_a_normal_quote():
    assert mid(99.98, 100.02) == pytest.approx(100.0)


@pytest.mark.parametrize("bid,ask", [
    (None, 100.0), (100.0, None),      # one side missing
    (100.02, 99.98),                   # crossed
    (0.0, 100.0),                      # zero bid
])
def test_mid_refuses_unusable_quotes(bid, ask):
    """A crossed or half-missing quote must not produce a confident midpoint."""
    assert mid(bid, ask) is None


# -- validation -----------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"side": "long"},                  # position language, not an order side
    {"reference_price": 0.0},
    {"reference_price": -5.0},
    {"qty": 0.0},
])
def test_nonsense_records_are_rejected_at_construction(kw):
    with pytest.raises(ValueError):
        _rec(**kw)


# -- round trip -----------------------------------------------------------

def test_log_round_trips_through_disk(tmp_path):
    log = FillLog(tmp_path / "fills.jsonl")
    log.record(_rec(filled_qty=100, filled_price=100.10, status="filled"))
    log.record(_rec(side="sell", filled_qty=50, filled_price=99.90, status="partial"))
    df = log.frame()
    assert len(df) == 2
    assert df["slippage_bps"].tolist() == pytest.approx([10.0, 10.0])


def test_derived_columns_are_recomputed_not_trusted(tmp_path):
    """A tampered slippage column in the file must not reach the analysis.

    The file is plain text. If `frame` read the stored number, anyone could
    make a losing strategy look profitable by editing it.
    """
    path = tmp_path / "fills.jsonl"
    log = FillLog(path)
    log.record(_rec(filled_qty=100, filled_price=100.10))

    raw = path.read_text(encoding="utf-8")
    doctored, n = re.subn(r'"slippage_bps": [-\d.eE+]+',
                          '"slippage_bps": -999.0', raw)
    # Guard the tampering itself. The stored value is 9.999999999999432, not
    # 10.0, so an exact-string replace silently matches nothing and leaves the
    # test asserting on untouched data -- it would pass even if `frame` did
    # trust the file.
    assert n == 1, "the doctoring did not take; this test would prove nothing"
    path.write_text(doctored, encoding="utf-8")

    assert FillLog(path).frame()["slippage_bps"].iloc[0] == pytest.approx(10.0)


def test_a_truncated_final_line_is_skipped(tmp_path):
    """A killed process leaves half a line. One bad row must not lose the file."""
    path = tmp_path / "fills.jsonl"
    log = FillLog(path)
    log.record(_rec(filled_qty=100, filled_price=100.10))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"symbol": "SPY", "side": "bu')
    assert len(FillLog(path).frame()) == 1


def test_empty_log_is_empty_not_an_error(tmp_path):
    assert FillLog(tmp_path / "nothing.jsonl").frame().empty
    assert summarise(pd.DataFrame()) == {"orders": 0}


# -- the verdict ----------------------------------------------------------

def _many(n, **kw):
    rows = []
    for _ in range(n):
        r = _rec(**kw)
        rows.append({**vars(r), "slippage_bps": r.slippage_bps,
                     "fill_ratio": r.fill_ratio, "notional": r.notional})
    return pd.DataFrame(rows)


def test_verdict_holds_when_costs_come_in_under_the_assumption():
    df = _many(40, filled_qty=100, filled_price=100.005)   # 0.5 bps -> 1 bps round trip
    v = verdict(df, assumed_cost_bps=3.0)
    assert v["verdict"] == "holds"


def test_verdict_fails_when_measured_cost_exceeds_the_assumption():
    df = _many(40, filled_qty=100, filled_price=100.05)    # 5 bps -> 10 bps round trip
    v = verdict(df, assumed_cost_bps=3.0)
    assert v["verdict"] == "fails"
    assert any("exceeds" in r for r in v["reasons"])


def test_verdict_fails_on_a_poor_fill_rate_even_when_the_fills_are_cheap():
    """The trap for a passive strategy.

    Resting limits that do fill can fill beautifully. The question is how many
    never filled at all -- those are the trades the backtest counted and the
    venue did not give you.
    """
    good = _many(10, filled_qty=100, filled_price=100.0)
    missed = _many(30, order_type="limit", limit_price=99.0)
    v = verdict(pd.concat([good, missed], ignore_index=True), assumed_cost_bps=10.0)
    assert v["verdict"] == "fails"
    assert any("fill rate" in r for r in v["reasons"])


def test_small_samples_are_called_insufficient_rather_than_judged():
    """Twenty orders cannot settle a 3 bps question, and should not pretend to."""
    v = verdict(_many(5, filled_qty=100, filled_price=100.05), assumed_cost_bps=3.0)
    assert v["verdict"] == "insufficient data"
