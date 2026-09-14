"""Percentages rendered as money.

The point of these is scale: a reader who skims past "+0.0083% per trade"
cannot skim past "8 cents". So the tests check the small numbers as carefully
as the large ones, and check that compounding is really compounding.
"""
from __future__ import annotations

import pytest

from core.money import (
    amount,
    annual,
    bps,
    describe,
    fmt,
    md,
    grow,
    once,
    over,
    per_trade,
)


# -- the arithmetic -------------------------------------------------------

def test_one_percent_of_a_thousand():
    assert amount(1.0, 1_000) == pytest.approx(10.0)


def test_a_basis_point_of_a_thousand_is_ten_cents():
    assert amount(0.01, 1_000) == pytest.approx(0.10)


def test_growth_compounds_rather_than_multiplies():
    """10% for two years is 21%, not 20%."""
    assert grow(10.0, 2, 1_000) == pytest.approx(1_210.0)


def test_a_gain_does_not_undo_an_equal_loss():
    """Down 50% then up 50% leaves you at 75, and a table built on simple
    multiplication would claim break-even."""
    assert grow(-50.0, 1, 1_000) == pytest.approx(500.0)
    assert grow(50.0, 1, 500) == pytest.approx(750.0)


def test_total_loss_ends_at_nothing_rather_than_going_negative():
    assert grow(-100.0, 3, 1_000) == 0.0
    assert grow(-150.0, 1, 1_000) == 0.0


def test_zero_percent_leaves_the_money_alone():
    assert grow(0.0, 100, 1_000) == pytest.approx(1_000.0)


# -- formatting -----------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0.08, "$0.08"),        # cents survive -- this is where thin edges live
    (9.99, "$9.99"),
    (10.0, "$10"),
    (1_234.5, "$1,235"),
    (25_000.0, "$25.0k"),
    (3_400_000.0, "$3.40m"),
    (-0.55, "-$0.55"),
    (-2_500.0, "-$2,500"),
])
def test_money_formatting(value, expected):
    assert fmt(value) == expected


def test_small_amounts_keep_their_cents():
    """Rounding 8 cents to $0 would erase exactly the point being made."""
    assert fmt(amount(0.0083, 1_000)) == "$0.08"


# -- the sentences --------------------------------------------------------

def test_once_reads_as_a_plain_sentence():
    assert once(1.0, 1_000) == "On $1,000 that is $10"


def test_over_states_start_end_and_gain():
    s = over(10.0, 2, "year", 1_000)
    assert "$1,000 becomes $1,210" in s
    assert "after 2 years" in s
    assert "+$210" in s


def test_a_single_period_is_singular():
    assert "after 1 year" in over(5.0, 1, "year", 1_000)


def test_a_loss_shows_a_negative_delta():
    s = over(-10.0, 2, "year", 1_000)
    assert "-$190" in s


def test_per_trade_shows_both_the_single_trade_and_the_year():
    """The real day-trading edge: +0.0083% a trade."""
    s = per_trade(0.0083, trades=250, basis=1_000)
    assert "$0.08 a trade" in s
    assert "250 trades" in s


def test_a_negative_per_trade_edge_ends_below_where_it_started():
    s = per_trade(-0.0076, trades=250, basis=1_000)
    assert "-$" in s.split("(")[-1], "the delta must read as a loss"


def test_annual_compounds_over_the_stated_years():
    s = annual(10.0, years=10, basis=1_000)
    assert "$2,594" in s        # 1000 * 1.1^10


def test_bps_reads_per_amount_traded():
    assert bps(1.59, 1_000) == "$0.16 per $1,000 traded"


# -- the single entry point ----------------------------------------------

@pytest.mark.parametrize("kind,pct", [
    ("annual", 8.0), ("trade", 0.05), ("bps", 2.0), ("once", 1.0),
])
def test_describe_handles_every_kind(kind, pct):
    out = describe(pct, kind=kind, basis=1_000)
    assert isinstance(out, str) and "$" in out


def test_describe_defaults_to_annual():
    assert describe(10.0, periods=10) == annual(10.0, 10)


# -- the compact form ----------------------------------------------------

def test_brief_is_short_enough_for_a_narrow_tile():
    """The long sentence wraps to five lines inside a six-column layout."""
    from core.money import brief
    for kind, pct in [("annual", 3.2), ("trade", 0.0083), ("bps", 1.59),
                      ("once", 17.2)]:
        out = brief(pct, kind=kind, basis=1_000)
        assert len(out) <= 34, f"{kind}: {out!r} is {len(out)} chars"
        assert "$" in out


def test_brief_annual_agrees_with_the_long_form():
    """Shorter wording, same arithmetic."""
    from core.money import brief, grow
    assert "$1,370" in brief(3.2, "annual", 1_000, 10)
    assert grow(3.2, 10, 1_000) == pytest.approx(1_370.2, abs=0.5)


# -- what the UI layer accepts -------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("+0.67 bps", "$0.07 per $1,000"),
    ("-9.20 bps", "-$0.92 per $1,000"),
    ("1.59 bps", "$0.16 per $1,000"),
    ("+2.35%", "$24 per $1,000"),
])
def test_money_line_reads_percentages_and_basis_points(value, expected):
    """Basis points need this more than percentages do.

    "+0.67 bps" and "-9.2 bps" look equally small at a glance and differ in
    sign only, so a tile showing them is the one most in need of a figure in
    dollars.
    """
    from core.ui import money_line
    assert money_line(value, "once") == expected


@pytest.mark.parametrize("value", [
    "unreachable", "connected", "\u2014", "", "12 trades", "0.34", "$1,000",
])
def test_money_line_stays_quiet_on_anything_that_is_not_a_rate(value):
    """Callers pass values through without checking, so a non-rate must
    produce nothing rather than a confident piece of nonsense."""
    from core.ui import money_line
    assert money_line(value, "once") == ""


def test_a_frequency_is_still_parsed_but_should_not_be_asked_for():
    """A win rate is a percentage and will render if asked.

    The guard is editorial, not mechanical: money lines are only attached to
    tiles showing a return or a cost. This test records that the helper
    cannot tell the difference, so the call sites have to.
    """
    from core.ui import money_line
    assert money_line("50%", "once") == "$500 per $1,000"


def test_dollar_signs_are_escaped_for_markdown():
    """Streamlit reads `$...$` as LaTeX.

    A sentence with two amounts in it silently became an equation and the
    words between them rendered as italic maths symbols. Caught by looking at
    the page rather than by any assertion, which is why there is now one.
    """
    from core.money import md
    out = md("On a $1,000 trade this nets $0.16 a trade")
    assert out.count("\$") == 2
    assert "$1,000" not in out.replace("\$", "@")


def test_md_is_for_markdown_only_not_for_html_blocks():
    """A record of a bug, so the escaping is not applied in the wrong place.

    Inside `unsafe_allow_html` the backslash has no meaning and renders as a
    literal "\$43". `md` belongs on st.markdown/st.info prose, never on a
    string being dropped into an HTML div.
    """
    from core.money import brief, md
    raw = brief(4.3, "once", 1_000)
    assert raw.startswith("$"), "HTML blocks take the unescaped form"
    assert md(raw).startswith("\$"), "markdown blocks take the escaped form"


def test_unmd_recovers_a_string_that_went_to_the_wrong_place():
    """The same money string is correct in markdown and wrong in HTML.

    Three separate times in this project an escaped string reached an
    unsafe_allow_html block and rendered as a literal "\$0.75". The HTML
    helpers now strip it on the way in, so the call site does not have to
    remember which variant it needs.
    """
    from core.money import md, unmd
    plain = "$0.75 per $1,000"
    assert md(plain) != plain
    assert unmd(md(plain)) == plain
    assert unmd(plain) == plain, "already-plain text passes through untouched"


def test_md_formats_a_number_rather_than_crashing_on_it():
    """Every other function here takes an amount, so passing one to md() is
    the obvious mistake. It used to fail inside str.replace with a message
    naming neither this module nor the caller."""
    assert md(0.19) == md(fmt(0.19))
    assert md(-2.5).startswith("-")


def test_md_still_escapes_a_string_the_way_it_did():
    assert md("costs $5.00 today") == "costs " + chr(92) + "$5.00 today"
