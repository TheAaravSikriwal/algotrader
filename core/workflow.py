"""The four stages drawn as a workflow, not as four coloured bars.

A row of solid blocks can show *where* a trade is. It cannot show what has to
happen next, and that was the question that kept coming back: "what am I
waiting for?". So every box carries its own trigger and its own plan, and
hovering any of them -- including the ones not reached yet -- answers "and
then what?" without having to get there first.

Three things this has to get right:

  * **One box lit, and it is the real one.** The lit box comes from the
    broker's own view of the account via `core.tradestage`, not from a
    separate guess made here.
  * **Every box has a tooltip.** A box that reveals nothing on hover is the
    unlabelled-grey-block problem in a new shape. It opens on keyboard focus
    as well as hover, so the text is reachable without a mouse -- and so it
    can be tested, since a synthetic mouse move does not set CSS `:hover`.
  * **No stray dollars.** Streamlit renders ``$...$`` as LaTeX, so any money
    in the text has to be neutralised before it reaches the page.
"""
from __future__ import annotations

import html

#: Colours are written as rgba literals rather than theme tokens because this
#: is injected as raw HTML, where CSS variables from the chart theme are not
#: in scope. They are deliberately the same greens and greys the rest of the
#: page uses.
CSS = """
<style>
.wf { display:flex; align-items:stretch; gap:0; margin:10px 0 18px;
      flex-wrap:nowrap; }
.wf-step { position:relative; flex:1 1 0; min-width:0; padding:10px 12px;
           border:1px solid rgba(128,128,128,.18); border-radius:9px;
           background:transparent; cursor:default; transition:all .15s ease; }
.wf-step .wf-title { font-size:13.5px; font-weight:600; line-height:1.25;
                     white-space:nowrap; overflow:hidden;
                     text-overflow:ellipsis; }
.wf-step .wf-label { font-size:11.5px; line-height:1.3; margin-top:2px;
                     opacity:.75; white-space:nowrap; overflow:hidden;
                     text-overflow:ellipsis; }
.wf-badge { display:inline-flex; align-items:center; justify-content:center;
            width:17px; height:17px; border-radius:50%; font-size:10.5px;
            font-weight:700; margin-bottom:6px;
            border:1px solid currentColor; }

.wf-upcoming { color:rgba(128,128,128,.5); }
.wf-done     { color:rgba(128,128,128,.85); }
.wf-done .wf-badge { background:rgba(128,128,128,.16); border-color:transparent; }
.wf-current  { color:#0ca30c; border-color:#0ca30c;
               background:rgba(12,163,12,.10);
               box-shadow:0 0 0 2px rgba(12,163,12,.10); }
.wf-current .wf-badge { background:#0ca30c; color:#fff; border-color:#0ca30c; }
.wf-current .wf-label { opacity:.95; }

.wf-arrow { display:flex; align-items:center; padding:0 6px; font-size:14px;
            color:rgba(128,128,128,.4); flex:0 0 auto; }

.wf-tip { visibility:hidden; opacity:0; position:absolute; z-index:60;
          left:0; top:calc(100% + 8px); width:290px; max-width:74vw;
          padding:11px 13px; border-radius:9px; text-align:left;
          border:1px solid rgba(128,128,128,.28);
          background:var(--background-color,#fff);
          box-shadow:0 8px 26px rgba(0,0,0,.22);
          transition:opacity .13s ease;
          /* Full contrast regardless of the box's own state. Inheriting meant
             an unreached box, which is deliberately faded to 50%, handed that
             fade to its tooltip -- so the boxes whose text you most need to
             read were the ones you could not. */
          color:var(--text-color, #0b0b0b); }
.wf-step:hover, .wf-step:focus, .wf-step:focus-within {
    border-color:rgba(128,128,128,.55); outline:none; }
.wf-step:hover .wf-tip,
.wf-step:focus .wf-tip,
.wf-step:focus-within .wf-tip { visibility:visible; opacity:1; }
.wf-step:last-child .wf-tip { left:auto; right:0; }
.wf-tip h5 { margin:0 0 3px; font-size:10.5px; letter-spacing:.07em;
             text-transform:uppercase; opacity:.62; font-weight:700; }
.wf-tip p { margin:0 0 9px; font-size:12.5px; line-height:1.5; opacity:.95; }
.wf-tip p:last-child { margin-bottom:0; }
.wf-tip .wf-tip-head { font-size:12.5px; font-weight:700; margin:0 0 7px;
                       opacity:1; }
</style>
"""


def _clean(text: str) -> str:
    """Escape for HTML, and defuse the dollar signs.

    Streamlit renders ``$...$`` as LaTeX even inside an ``unsafe_allow_html``
    block, and a backslash escape shows up literally as ``\\$``. Replacing the
    sign with its HTML entity renders as a dollar and never as maths.
    """
    return html.escape(str(text)).replace("$", "&#36;")


def render(steps, heading: str = "") -> str:
    """The workflow as one HTML string.

    `steps` are `core.tradestage.Step`s, already carrying their own state.
    Nothing here decides which box is lit.
    """
    if not steps:
        return ""
    out = [CSS, '<div class="wf">']
    for i, step in enumerate(steps):
        if i:
            out.append('<div class="wf-arrow">&#10230;</div>')
        badge = "&#10003;" if step.state == "done" else str(step.n)
        out.append(
            f'<div class="wf-step wf-{step.state}" tabindex="0" '
            f'role="button" aria-label="Stage {step.n}, {_clean(step.title)}">'
            f'<div class="wf-badge">{badge}</div>'
            f'<div class="wf-title">{_clean(step.title)}</div>'
            f'<div class="wf-label">{_clean(step.label)}</div>'
            f'<div class="wf-tip">'
            f'<p class="wf-tip-head">Stage {step.n} &#183; {_clean(step.title)}</p>'
            f'<h5>What has to happen</h5><p>{_clean(step.waiting_for)}</p>'
            f'<h5>What the app does</h5><p>{_clean(step.plan)}</p>'
            f'</div></div>')
    out.append("</div>")
    if heading:
        out.insert(1, f'<div style="font-size:11.5px;letter-spacing:.07em;'
                      f'text-transform:uppercase;opacity:.55;font-weight:700;'
                      f'margin-bottom:2px">{_clean(heading)}</div>')
    return "".join(out)
