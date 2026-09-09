"""Color tokens and Plotly styling.

Two selected modes -- the dark values are their own steps from the same ramps,
not an automatic inversion of the light ones.
"""
from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # categorical slots 1 and 2 -- validated as a pair against this surface
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "series_3": "#1baf7a",
    "drawdown": "#e34948",
    "good": "#0ca30c",
    "critical": "#d03b3b",
    # diverging arms for the monthly-returns grid
    "div_neg": "#d03b3b",
    "div_mid": "#f0efec",
    "div_pos": "#2a78d6",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "series_3": "#199e70",
    "drawdown": "#e66767",
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "div_neg": "#d03b3b",
    "div_mid": "#383835",
    "div_pos": "#3987e5",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(mode: str) -> dict:
    return DARK if mode == "dark" else LIGHT


def apply_layout(fig, mode: str, title: str = "", height: int = 380, legend: bool = False):
    """Recessive chrome, one axis, crosshair hover.

    `legend=True` reserves the extra top margin a legend row needs; single-series
    charts carry no legend box, so they keep the tighter header.
    """
    t = tokens(mode)
    top = (72 if legend else 46) if title else 12
    fig.update_layout(
        # title pinned to the figure container so the legend can sit below it
        title=dict(text=title, font=dict(size=15, color=t["text_primary"]),
                   yref="container", y=1.0, yanchor="top", pad=dict(t=14, l=4)) if title else None,
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font=dict(family=FONT, size=12, color=t["text_secondary"]),
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"],
                        font=dict(family=FONT, size=12, color=t["text_primary"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(showgrid=False, linecolor=t["axis"], tickcolor=t["axis"],
                   tickfont=dict(color=t["muted"]), showspikes=True,
                   spikecolor=t["axis"], spikethickness=1, spikemode="across", spikedash="dot"),
        yaxis=dict(gridcolor=t["grid"], zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=t["muted"])),
    )
    return fig
