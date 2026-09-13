"""The research ledger.

Every test ever run, and the bar that rises each time one is recorded. This
page exists because the single most important number in the project — how many
things have been tried — was previously only visible by typing a command.

Search enough strategies and something clears any fixed threshold by luck. The
ledger is what stops that from feeling like a discovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.journal import NULL, SIGNIFICANT, SIGNIFICANT_NEGATIVE, Journal
from core.theme import apply_layout, tokens
from core.ui import page_header, tile

try:
    st.set_page_config(page_title="Research journal", layout="wide")
except Exception:
    pass  # the host page already configured it


def tstat_chart(frame: pd.DataFrame, bar: float, mode: str) -> go.Figure:
    """Every test against the corrected bar, in the order they were run."""
    t = tokens(mode)
    colours = [t["good"] if v == SIGNIFICANT else
               (t["critical"] if v == SIGNIFICANT_NEGATIVE else t["muted"])
               for v in frame["verdict"]]

    fig = go.Figure(go.Bar(
        x=list(range(1, len(frame) + 1)),
        y=frame["t"].fillna(0.0),
        marker=dict(color=colours, line=dict(color=t["surface"], width=2)),
        customdata=frame[["strategy", "verdict"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>t = %{y:.2f}  ·  %{customdata[1]}"
                      "<extra></extra>",
        showlegend=False))

    for level in (bar, -bar):
        fig.add_hline(y=level, line=dict(color=t["critical"], width=2))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.add_annotation(x=len(frame), y=bar, text=f"corrected bar {bar:.2f}",
                       showarrow=False, yshift=12, xanchor="right",
                       font=dict(size=12, color=t["critical"]))

    apply_layout(fig, mode, "Every test against the bar it had to clear", 380)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(title=None, tickmode="linear", dtick=1)
    fig.update_yaxes(title=None)
    return fig


def show_forward_tests(journal: Journal):
    try:
        tests = journal.forward_tests()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Forward tests unavailable: {exc}")
        return
    if not tests:
        return

    st.markdown("#### Forward tests")
    st.caption("Specifications locked before the data existed. These are the "
               "only results that were never fitted to anything.")

    rows = [{"id": t.id, "strategy": t.strategy, "symbols": ",".join(t.symbols)[:30],
             "locked": (t.registered or "")[:10], "days_elapsed": t.days_elapsed,
             "min_days": t.min_days, "status": t.status,
             "ready": "yes" if t.ready else "not yet"}
            for t in tests]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    ready = journal.ready_forward_tests()
    if ready:
        st.success(f"{len(ready)} forward test(s) have reached their scoring "
                   "date. Score them with `python research.py score`.")


def main():
    mode = page_header(
        "Research journal",
        "Every test ever run, and the bar that rises with each one.")

    journal = Journal()
    frame = journal.frame()

    if frame.empty:
        st.info("No tests recorded yet. Run one from the command line:\n\n"
                "`python research.py test --strategy \"SMA crossover\" "
                "--symbols SPY --hypothesis \"beats buy and hold\"`")
        return

    stats = journal.summary()

    cols = st.columns(5)
    tile(cols[0], "Tests run", f"{stats['tests_run']}", "each one raises the bar")
    tile(cols[1], "Corrected bar", f"{stats['corrected_bar']:.2f}",
         f"naive bar {stats['uncorrected_bar']:.2f}")
    tile(cols[2], "Found real", f"{stats['significant_positive']}",
         "cleared the corrected bar",
         "up" if stats["significant_positive"] else "flat")
    tile(cols[3], "Null", f"{stats['null']}", "indistinguishable from noise")
    tile(cols[4], "Expected by luck",
         f"{stats['expected_false_positives_uncorrected']:.1f}",
         "at the naive bar, by chance")

    st.write("")
    st.plotly_chart(tstat_chart(frame, stats["corrected_bar"], mode),
                    use_container_width=True)

    if stats["significant_positive"] == 0:
        st.info(
            f"Nothing has cleared the corrected bar in {stats['tests_run']} "
            "tests. That is the honest state of the search, not a bug — and at "
            "this many tests roughly "
            f"{stats['expected_false_positives_uncorrected']:.1f} would clear "
            "the naive bar by chance alone.")
    if stats["survives_fdr"]:
        st.caption(f"{stats['survives_fdr']} result(s) survive the "
                   "Benjamini-Hochberg correction, which is more permissive "
                   "than Bonferroni. Suggestive, not a finding.")

    st.markdown("#### Every test")
    display = frame.drop(columns=["id"], errors="ignore")
    st.dataframe(
        display, hide_index=True, use_container_width=True, height=420,
        column_config={
            "recorded": st.column_config.TextColumn("Date", width="small"),
            "strategy": st.column_config.TextColumn("Strategy", width="medium"),
            "universe": st.column_config.TextColumn("Universe", width="small"),
            "window": st.column_config.TextColumn("Window", width="small"),
            "return_%": st.column_config.NumberColumn("Return", format="%.2f%%"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_dd_%": st.column_config.NumberColumn("Max DD", format="%.2f%%"),
            "t": st.column_config.NumberColumn("t", format="%.2f"),
            "p": st.column_config.NumberColumn("p", format="%.4f"),
            "verdict": st.column_config.TextColumn("Verdict", width="small"),
            "hypothesis": st.column_config.TextColumn("Hypothesis", width="large"),
            "survives_fdr": st.column_config.CheckboxColumn("FDR"),
        })

    st.download_button("Download the ledger (CSV)",
                       frame.to_csv(index=False).encode(),
                       file_name="research_journal.csv", mime="text/csv")

    st.divider()
    show_forward_tests(journal)

    st.caption(
        "The bar rises because every additional test is another chance for "
        "noise to look like signal. A result that would have been convincing "
        "as the first thing tried is not convincing as the twentieth.")


if __name__ == "__main__":
    main()
