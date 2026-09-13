"""Turn a percentage into money, because percentages hide their own size.

"+0.0083% per trade" and "−0.92 basis points" are both easy to read past.
"On $1,000 that is 8 cents a trade" is not. Every percentage the app shows
gets a line like the second one, so the scale of an edge is visible without
doing arithmetic in your head.

The worked amount is an illustration, not a forecast. It answers "what would
this rate do to a round number over this many periods", which is a question
about the rate, not a claim about the future. `caveat()` supplies the wording
that says so wherever the number is shown.
"""
from __future__ import annotations

import math

DEFAULT_BASIS = 1_000.0


def amount(pct: float, basis: float = DEFAULT_BASIS) -> float:
    """What `pct` percent of `basis` comes to, in currency."""
    return basis * pct / 100.0


def fmt(value: float) -> str:
    """Money, at a precision that suits its size.

    Cents matter under ten dollars -- that is exactly the range where a thin
    edge lives, and rounding it to the nearest dollar would erase the point
    being made.
    """
    v = float(value)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v < 10:
        return f"{sign}${v:,.2f}"
    if v < 10_000:
        # Half away from zero, not Python's default half-to-even. $1,234.50
        # formatting as "$1,234" reads as a truncation bug to anyone checking
        # the arithmetic, and money is the one place people do check.
        return f"{sign}${math.floor(v + 0.5):,.0f}"
    if v < 1_000_000:
        return f"{sign}${v/1_000:,.1f}k"
    return f"{sign}${v/1_000_000:,.2f}m"


def once(pct: float, basis: float = DEFAULT_BASIS) -> str:
    """A single application of the rate. "On $1,000 that is $10." """
    return f"On {fmt(basis)} that is {fmt(amount(pct, basis))}"


def grow(pct: float, periods: int = 1, basis: float = DEFAULT_BASIS) -> float:
    """`basis` compounded at `pct` for `periods`.

    Compounded rather than multiplied, because repeated percentages do not
    add: losing 50% then gaining 50% is not break-even, and a table built on
    simple multiplication would say it was.
    """
    rate = 1.0 + pct / 100.0
    if rate <= 0:
        return 0.0            # a loss of 100% or worse ends at nothing
    return basis * rate ** periods


def over(pct: float, periods: int, period_name: str,
         basis: float = DEFAULT_BASIS) -> str:
    """The full sentence: start, end, and the gain between them.

    `period_name` is the singular noun -- "year", "trade", "month".
    """
    end = grow(pct, periods, basis)
    delta = end - basis
    unit = period_name if periods == 1 else f"{period_name}s"
    when = f"after {periods:,} {unit}"
    return (f"{fmt(basis)} becomes {fmt(end)} {when} "
            f"({'+' if delta >= 0 else ''}{fmt(delta)})")


def per_trade(pct: float, trades: int = 250,
              basis: float = DEFAULT_BASIS) -> str:
    """A per-trade edge, shown once and compounded over a year of trading.

    250 trades is roughly one a day for a year, which is the cadence a
    short-term rule actually runs at.
    """
    single = amount(pct, basis)
    end = grow(pct, trades, basis)
    return (f"{fmt(single)} a trade on {fmt(basis)} -- "
            f"{fmt(end)} after {trades:,} trades "
            f"({'+' if end >= basis else ''}{fmt(end - basis)})")


def annual(pct: float, years: int = 10, basis: float = DEFAULT_BASIS) -> str:
    """An annual rate, compounded over a realistic holding period."""
    return over(pct, years, "year", basis)


def bps(basis_points: float, basis: float = DEFAULT_BASIS) -> str:
    """Basis points, which are the easiest unit of all to read past.

    One basis point is a hundredth of a percent; on $1,000 it is ten cents.
    """
    return f"{fmt(amount(basis_points / 100.0, basis))} per {fmt(basis)} traded"


def caveat() -> str:
    return ("Illustration only, at a constant rate. Real returns arrive "
            "unevenly and this ignores tax.")


def describe(pct: float, kind: str = "annual", basis: float = DEFAULT_BASIS,
             periods: int | None = None) -> str:
    """One entry point the UI can call for any percentage it displays.

    `kind` is "annual", "trade", "bps" or "once".
    """
    if kind == "annual":
        return annual(pct, periods or 10, basis)
    if kind == "trade":
        return per_trade(pct, periods or 250, basis)
    if kind == "bps":
        return bps(pct, basis)
    return once(pct, basis)

def brief(pct: float, kind: str = "annual", basis: float = DEFAULT_BASIS,
          periods: int | None = None) -> str:
    """The same illustration, short enough for a narrow stat tile.

    The long sentences read well in body text and wrap to five lines inside a
    six-column tile, where they cost more legibility than they add.
    """
    if kind == "annual":
        years = periods or 10
        return f"{fmt(basis)} -> {fmt(grow(pct, years, basis))} in {years}y"
    if kind == "trade":
        n = periods or 250
        return (f"{fmt(amount(pct, basis))}/trade -> "
                f"{fmt(grow(pct, n, basis))} in {n:,}")
    if kind == "bps":
        return f"{fmt(amount(pct / 100.0, basis))} per {fmt(basis)}"
    return f"{fmt(amount(pct, basis))} per {fmt(basis)}"


def md(text: str) -> str:
    """Escape dollar signs so Streamlit does not read them as maths.

    Streamlit's markdown treats `$...$` as LaTeX. A sentence containing two
    amounts silently becomes an equation and the text between them renders as
    italic symbols -- which is exactly what happened the first time this
    module's output reached a markdown block.
    """
    return text.replace("$", "\\$")
