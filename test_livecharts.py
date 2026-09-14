"""The cycle strip, and whether its legend tells the truth about it.

A legend that disagrees with the picture is worse than no legend: it does not
leave you guessing, it tells you the wrong thing confidently. These tests pin
the two together.
"""
from __future__ import annotations

import colorsys

from core.livecharts import CYCLE_KINDS, cycle_legend, cycle_strip
from core.theme import tokens

#: The hue band each legend emoji promises, in degrees. Grey is the exception:
#: it has no hue at all, so it is identified by having almost no saturation.
SWATCH_HUE = {
    "🟩": (90, 160),      # green square
    "🟥": (-15, 15),      # red square
    "🟦": (195, 250),     # blue square
    "🟨": (35, 70),       # yellow square
    "🟧": (16, 34),       # orange square
}
GREY = "⬜"
#: Below this, a colour reads as grey whatever its nominal hue.
GREY_SATURATION = 0.15


def _hls(hex_colour: str):
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, _light, sat = colorsys.rgb_to_hls(r, g, b)
    return hue * 360, sat


def _reads_as(hex_colour: str) -> str:
    """Which legend emoji this colour actually looks like."""
    hue, sat = _hls(hex_colour)
    if sat < GREY_SATURATION:
        return GREY
    for swatch, (lo, hi) in SWATCH_HUE.items():
        if lo <= (hue - 360 if hue > 180 and lo < 0 else hue) <= hi:
            return swatch
    return f"no swatch (hue {hue:.0f})"


def _bar_colours(cycles, mode="light"):
    fig = cycle_strip(cycles, mode)
    return list(fig.data[0].marker.color)


# -- the legend matches the picture ---------------------------------------

def test_every_block_is_the_colour_its_legend_emoji_shows():
    """The bug this exists for: 'watched, no setup' was captioned with a blue
    square and drawn in orange, so the commonest outcome on the strip looked
    like a different one entirely.

    Checked by hue rather than exact equality -- the theme colours are their
    own shades, and what matters is that someone glancing at the block and
    then at the legend reaches the same word.
    """
    for mode in ("light", "dark"):
        t = tokens(mode)
        for key, kind in CYCLE_KINDS.items():
            actual = t[kind.token]
            assert _reads_as(actual) == kind.swatch, (
                f"{mode}: {key!r} is captioned {kind.swatch} but is drawn "
                f"{actual}, which reads as {_reads_as(actual)}")


def test_no_two_outcomes_share_a_colour():
    """'Placed an order' and 'told to wait' were two near-identical greens.
    One means a trade happened and the other means nothing did."""
    for mode in ("light", "dark"):
        t = tokens(mode)
        seen: dict[str, str] = {}
        for key, kind in CYCLE_KINDS.items():
            reads = _reads_as(t[kind.token])
            assert reads not in seen, (
                f"{mode}: {key!r} and {seen[reads]!r} both look {reads}")
            seen[reads] = key


def test_closed_out_and_told_to_wait_are_not_both_reds():
    """A close is a real event and a throttle is not. In dark mode these were
    35 units apart in RGB, which is a shade difference, not a colour one."""
    for mode in ("light", "dark"):
        t = tokens(mode)
        closed, _ = _hls(t[CYCLE_KINDS["flatten"].token])
        waited, _ = _hls(t[CYCLE_KINDS["throttled"].token])
        assert abs(closed - waited) > 30


def test_the_legend_names_every_kind_the_strip_can_draw():
    legend = cycle_legend()
    for kind in CYCLE_KINDS.values():
        assert kind.swatch in legend
        assert kind.words in legend


# -- the strip itself -----------------------------------------------------

def test_each_cycle_is_drawn_in_its_own_kinds_colour():
    t = tokens("light")
    cycles = [{"kind": "watching"}, {"kind": "order"}, {"kind": "throttled"}]
    assert _bar_colours(cycles) == [t["series_1"], t["good"], t["pending"]]


def test_an_unrecognised_kind_falls_back_to_grey_rather_than_vanishing():
    """A kind added to the loop and not to the palette must still show a
    block -- a missing block reads as 'the loop was asleep'."""
    colours = _bar_colours([{"kind": "something_new"}])
    assert colours == [tokens("light")["muted"]]


def test_no_cycles_yet_says_so_instead_of_drawing_an_empty_axis():
    fig = cycle_strip([], "light")
    assert not fig.data
    assert "No cycles yet" in str(fig.layout.title.text)


def test_hovering_a_block_says_when_it_ran_and_what_it_decided():
    fig = cycle_strip([{"kind": "watching", "time": "14:57",
                        "label": "Watching — no setup"}], "light")
    assert "14:57" in fig.data[0].hovertext[0]
    assert "no setup" in fig.data[0].hovertext[0]
