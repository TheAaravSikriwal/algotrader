"""Shared Streamlit pieces.

Extracted so every page paints the same chrome. `param_control` is the reason
adding a strategy needs no UI work at all: a strategy declares its `Param`
list, and the right control appears wherever that strategy is selectable.
"""
from __future__ import annotations

import streamlit as st

from core.strategy import Param
from core.theme import FONT, tokens


def active_mode() -> str:
    """Match the charts to the chrome by reading the configured theme base.

    Deliberately `theme.base` from .streamlit/config.toml rather than
    `st.context.theme.type` -- the latter reports the *browser's* preferred
    colour scheme, which disagrees with the theme Streamlit actually paints
    whenever config.toml pins one.
    """
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:  # noqa: BLE001
        return "light"


def inject_css(mode: str):
    t = tokens(mode)
    ring = "rgba(255,255,255,0.10)" if mode == "dark" else "rgba(11,11,11,0.10)"
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{ font-family: {FONT}; }}
      .tile {{
        background: {t['surface']}; border: 1px solid {ring};
        border-radius: 10px; padding: 14px 16px; height: 100%;
      }}
      .tile .label {{ font-size: 12px; color: {t['muted']}; letter-spacing: .02em; }}
      .tile .value {{ font-size: 28px; line-height: 1.15; margin-top: 4px;
                      color: {t['text_primary']}; font-weight: 600;
                      white-space: nowrap; }}
      .tile .sub   {{ font-size: 12px; margin-top: 4px; color: {t['text_secondary']}; }}
      .tile .value.up   {{ color: {t['good']}; }}
      .tile .value.down {{ color: {t['critical']}; }}
      .tile .value.flat {{ color: {t['muted']}; }}
    </style>
    """, unsafe_allow_html=True)


def tile(col, label: str, value: str, sub: str = "", tone: str = ""):
    cls = f" {tone}" if tone else ""
    col.markdown(
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value{cls}">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:,.{digits}f}%"


def tone_of(x: float) -> str:
    return "up" if x > 0 else ("down" if x < 0 else "")


def param_control(p: Param, key: str):
    """Render the control a Param describes. Adding a strategy needs no UI edit."""
    if p.kind == "bool":
        return st.checkbox(p.label, value=bool(p.default), key=key, help=p.help or None)
    if p.kind == "choice":
        return st.selectbox(p.label, p.choices, index=p.choices.index(p.default),
                            key=key, help=p.help or None)
    if p.kind == "float":
        return st.slider(p.label, float(p.min), float(p.max), float(p.default),
                         float(p.step or 0.1), key=key, help=p.help or None)
    return st.slider(p.label, int(p.min), int(p.max), int(p.default),
                     int(p.step or 1), key=key, help=p.help or None)


def param_form(params, prefix: str) -> dict:
    """Controls for a whole strategy, returned as its kwargs."""
    return {p.name: param_control(p, f"{prefix}_{p.name}") for p in params}


def page_header(title: str, subtitle: str = "") -> str:
    mode = active_mode()
    inject_css(mode)
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    return mode


# ---------------------------------------------------------------------------
# plain-English helpers
# ---------------------------------------------------------------------------
def explained_tile(col, key: str, value: str, sub: str = "", tone: str = "",
                   label: str | None = None):
    """A stat tile that can explain itself on hover."""
    from core.explain import metric

    import html

    item = metric(key)
    title = label or (item.label if item else key.replace("_", " ").capitalize())
    cls = f" {tone}" if tone else ""

    # The tooltip is free prose with quotes and line breaks in it. Dropping that
    # straight into a title attribute terminates the attribute early and the
    # remainder renders as visible page text -- escape it, and encode the
    # newlines rather than emitting them raw.
    hint = ""
    if item:
        hint = html.escape(item.tooltip(), quote=True).replace("\n", "&#10;")

    col.markdown(
        f'<div class="tile" title="{hint}">'
        f'<div class="label">{html.escape(title)} '
        f'<span style="opacity:.45">&#9432;</span></div>'
        f'<div class="value{cls}">{html.escape(str(value))}</div>'
        f'<div class="sub">{html.escape(str(sub))}</div></div>',
        unsafe_allow_html=True)


def explain_row(keys: list[str], title: str = "What do these numbers mean?"):
    """An expander spelling out every metric in a row of tiles."""
    import streamlit as st
    from core.explain import METRICS

    with st.expander(title):
        for key in keys:
            item = METRICS.get(key)
            if item:
                st.markdown(item.markdown())
                st.write("")


def explain_chart(key: str):
    """A short note under a chart saying how to read it."""
    import streamlit as st
    from core.explain import chart

    item = chart(key)
    if not item:
        return
    text = f"**How to read this:** {item.plain}"
    if item.good:
        text += f"  \n**What you want to see:** {item.good}"
    if item.trap:
        text += f"  \n**What can fool you:** {item.trap}"
    st.caption(text)


def step(number: int, title: str, done: bool = False, active: bool = False):
    """A numbered step heading, so the order of operations is never in doubt."""
    import streamlit as st

    mark = "✓" if done else str(number)
    weight = "600" if active else "400"
    colour = "var(--good)" if done else "inherit"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0 2px">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:26px;height:26px;border-radius:50%;border:1px solid rgba(128,128,128,.4);'
        f'font-size:13px;color:{colour}">{mark}</span>'
        f'<span style="font-size:18px;font-weight:{weight}">{title}</span></div>',
        unsafe_allow_html=True)


def plain(text: str):
    """A short plain-English lead-in above a section."""
    import streamlit as st
    st.markdown(f'<div style="opacity:.75;margin:2px 0 10px">{text}</div>',
                unsafe_allow_html=True)
