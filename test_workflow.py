"""The workflow strip's HTML.

Rendering is easy to get subtly wrong in ways nobody notices until the page
is in front of them: two boxes lit, a tooltip that is empty, a dollar sign
that turns into LaTeX. Each test here is one of those.
"""
from __future__ import annotations

import re

from core.tradestage import ORDER_PLACED, describe
from core.workflow import render


class FakeOrder:
    symbol, qty, side, limit_price = "QQQ", 1, "buy", 712.00


def _html(**kw) -> str:
    return render(describe(**kw).steps)


def _boxes(markup: str) -> list[str]:
    return re.findall(r'<div class="wf-step wf-(\w+)"', markup)


# -- which box is lit -----------------------------------------------------

def test_exactly_one_box_is_lit():
    """Two lit boxes is worse than none: it reads as two things happening."""
    for kw in ({}, {"closed_today": True},
               {"order": FakeOrder(), "quote": (712.28, 712.34)}):
        assert _boxes(_html(**kw)).count("current") == 1


def test_the_lit_box_is_the_stage_the_account_is_actually_at():
    markup = _html(order=FakeOrder(), quote=(712.28, 712.34))
    states = _boxes(markup)
    assert states[1] == "current"
    assert states[0] == "done"
    assert states[2] == states[3] == "upcoming"


def test_finished_boxes_show_a_tick_instead_of_their_number():
    markup = _html(order=FakeOrder(), quote=(712.28, 712.34))
    assert markup.index("&#10003;") < markup.index(">2<")


# -- the tooltips ---------------------------------------------------------

def test_every_box_carries_a_tooltip_including_the_ones_not_reached():
    """The reason later boxes have text at all: hovering box 3 while still
    watching has to answer 'and then what?'."""
    markup = _html()
    assert markup.count('class="wf-tip"') == 4
    for body in re.findall(r'<div class="wf-tip">(.*?)</div>', markup):
        assert "What has to happen" in body
        assert "What the app does" in body


def test_a_tooltip_is_never_an_empty_shell():
    for body in re.findall(r"<h5>What has to happen</h5><p>(.*?)</p>", _html()):
        assert len(body) > 25


def test_the_tooltip_names_the_stage_so_it_is_clear_which_box_it_belongs_to():
    markup = _html()
    for n in (1, 2, 3, 4):
        assert f"Stage {n} &#183;" in markup


def test_the_boxes_appear_in_order_with_arrows_between_them():
    markup = _html()
    assert markup.count('class="wf-arrow"') == 3
    assert [int(m) for m in re.findall(r'class="wf-badge">(\d)<', markup)] == [1, 2, 3, 4]


# -- text that has to survive the page ------------------------------------

def test_dollar_amounts_do_not_turn_into_latex():
    """Streamlit renders $...$ as maths even inside unsafe_allow_html, and a
    backslash escape shows up literally as \\$. Twice now."""
    from core.tradestage import Step
    step = Step(1, "k", "T", "l", waiting_for="it costs $500.00 to do",
                plan="and $250.00 comes back", state="current")
    markup = render([step])
    assert "$" not in markup
    assert "&#36;500.00" in markup


def test_text_from_the_broker_cannot_inject_markup():
    from core.tradestage import Step
    step = Step(1, "k", "<b>T</b>", "l", waiting_for="<script>x</script>",
                plan="p" * 30, state="current")
    markup = render([step])
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_nothing_to_draw_renders_nothing_rather_than_an_empty_frame():
    assert render([]) == ""


# -- reachable without a mouse --------------------------------------------

def test_every_box_can_be_focused_so_the_text_is_not_mouse_only():
    markup = _html()
    assert markup.count('tabindex="0"') == 4
    assert markup.count("aria-label=") == 4


def test_focus_opens_the_tooltip_as_well_as_hover():
    """A tooltip that only answers to :hover is unreachable by keyboard, and
    cannot be verified by automation either -- a synthetic mouse move does
    not set :hover."""
    from core.workflow import CSS
    assert ".wf-step:focus .wf-tip" in CSS
    assert ".wf-step:hover .wf-tip" in CSS


def test_a_tooltip_is_not_faded_just_because_its_box_is():
    """Upcoming boxes are drawn at 50% on purpose. Inheriting that into the
    tooltip made the text you most need -- 'and then what?' -- the hardest to
    read on the page."""
    from core.workflow import CSS
    tip_rule = CSS[CSS.index(".wf-tip {"):CSS.index(".wf-step:hover")]
    assert "color:inherit" not in tip_rule.replace(" ", "")
    assert "--text-color" in tip_rule
