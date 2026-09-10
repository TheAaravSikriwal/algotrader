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
