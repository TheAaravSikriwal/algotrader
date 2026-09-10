"""Run the whole book against a practice account.

Three buttons in a fixed order: see what it wants to do, do it, check what
happened. The order is enforced by layout rather than left to the reader,
because the failure mode here is clicking the wrong thing with real money on
the other side of it later.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import BrokerError
from core.env import load_env
from core.overlay import ledger_frame
from core.paneltrader import PanelLiveConfig, PanelTrader
from core.theme import apply_layout, tokens
from core.trader import Halted
from core.ui import explain_row, explained_tile, page_header, plain, step
from strategies.cross_sectional import available_xs, get_xs_strategy

load_env()

st.set_page_config(page_title="Practice trading", layout="wide")

BOOKS = {
    "Sectors and bonds (macro)": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD",
                                  "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
                                  "XLP", "XLU"],
    "Big US companies": ["NVDA", "GOOGL", "TSLA", "MSFT", "AMZN", "META",
                         "AAPL", "MU", "AMD", "INTC", "AVGO", "CRM", "JPM",
                         "NFLX"],
    "Just the market (simplest)": ["SPY", "QQQ", "TLT"],
}


def weight_chart(base: pd.Series, final: pd.Series, mode: str) -> go.Figure:
    t = tokens(mode)
    order = final.sort_values(ascending=False).index
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(order), y=[base[s] * 100 for s in order], name="Plan",
        marker=dict(color=t["muted"], line=dict(color=t["surface"], width=2)),
        hovertemplate="plan %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(
        x=list(order), y=[final[s] * 100 for s in order], name="After news",
        marker=dict(color=t["series_1"], line=dict(color=t["surface"], width=2)),
        hovertemplate="after news %{y:.2f}%<extra></extra>"))
    apply_layout(fig, mode, "What the book should look like today", 340)
    fig.update_layout(barmode="group", hovermode="x unified", bargap=0.25)
    fig.update_yaxes(ticksuffix="%")
    return fig


def connect(symbols, strategy, params, dry_run, use_overlay, gross, bucket):
    from brokers.alpaca import AlpacaBroker

    broker = AlpacaBroker(paper=True)
    cfg = PanelLiveConfig(
        symbols=list(symbols), strategy=strategy, params=params,
        dry_run=dry_run, use_overlay=use_overlay, gross_target=gross,
        overlay_bucket=bucket, require_market_open=False)
    return PanelTrader(broker, get_xs_strategy(strategy)(**params), cfg), broker


def main():
    mode = page_header("Practice trading")

    st.markdown(
        "This runs your strategy against a **practice account** — real prices, "
        "real timing, **fake money**. Nothing here can spend anything.")

    with st.sidebar:
        st.markdown("### Set up")
        book_name = st.selectbox("What to trade", list(BOOKS),
                                 help="A basket, not single stocks. Start with "
                                      "the simplest one if unsure.")
        raw = st.text_area("Tickers", ",".join(BOOKS[book_name]), height=80)
        symbols = tuple(s.strip().upper() for s in raw.split(",") if s.strip())

        names = available_xs()
        default = "Equal weight all"
        strategy = st.selectbox(
            "How to split the money", names,
            index=names.index(default) if default in names else 0,
            help="'Equal weight all' just splits evenly — the simplest and "
                 "hardest-to-beat starting point.")
        cls = get_xs_strategy(strategy)
        if cls.description:
            st.caption(cls.description)

        from core.ui import param_form
        params = param_form(cls.params, f"live_{strategy}")

        st.divider()
        gross = st.slider("How much of the account to use", 0.5, 1.0, 0.95, 0.05,
                          help="0.95 leaves a little cash so rounding never "
                               "leaves you short.")
        use_overlay = st.checkbox(
            "Apply this week's news views", True,
            help="Adjusts the plan using anything recorded on the news page "
                 "that has not expired.")
        bucket = st.selectbox("Which news views", ["", "macro", "company"],
                              format_func=lambda b: b or "all of them")

    if len(symbols) < 3:
        st.error("Pick at least three tickers — this is a basket, not a bet.")
        return

    ledger = ledger_frame()
    live_views = 0 if ledger is None or ledger.empty else int(ledger["active"].sum())

    # ---- step 1 ---------------------------------------------------------
    step(1, "See what it wants to do", active=True)
    plain("Nothing is sent. This works out today's target and compares it to "
          "what the practice account currently holds.")

    if st.button("Work out today's plan", type="primary"):
        try:
            trader, _ = connect(symbols, strategy, params, True, use_overlay,
                                gross, bucket)
            st.session_state["preview"] = trader.run_once()
        except BrokerError as exc:
            st.error(f"Could not reach the practice account: {exc}")
            st.info("Add your Alpaca paper keys to the `.env` file, then run "
                    "`python check_setup.py` to confirm they work.")
            return
        except Halted as exc:
            st.error(f"Trading stopped on purpose: {exc}")
            return

    preview = st.session_state.get("preview")
    if preview:
        cols = st.columns(4)
        explained_tile(cols[0], "gross_exposure",
                       f"${preview['equity']:,.0f}", "practice account value",
                       label="Account")
        explained_tile(cols[1], "turnover", f"{len(preview['intents'])}",
                       "trades to get there", label="Trades needed")
        explained_tile(cols[2], "precedents", f"{live_views}",
                       "news views still active", label="News views")
        explained_tile(cols[3], "gross_exposure",
                       f"${preview['buying_power']:,.0f}", "available to spend",
                       label="Buying power")

        if preview["status"] == "market closed":
            st.warning("The market is closed, so nothing would be sent right "
                       "now. The plan below is still what it would do at the "
                       "next open.")

        base, final = preview["base"], preview["final"]
        changed = not base.equals(final)
        st.plotly_chart(weight_chart(base, final, mode),
                        use_container_width=True)
        st.caption(
            "**How to read this:** grey is what the strategy alone wants. Blue "
            "is after this week's news views are applied. If they are identical, "
            "no news view is currently active — which is normal and fine."
            if changed else
            "**How to read this:** the two bars match, meaning no news views are "
            "currently affecting the plan. That is the normal state.")

        if preview["intents"]:
            rows = [{"ticker": i.symbol,
                     "buy or sell": "buy" if i.delta > 0 else "sell",
                     "shares": abs(round(i.delta)),
                     "roughly": f"${i.notional:,.0f}",
                     "holding now": f"{i.current_weight:.1%}",
                     "should hold": f"{i.target_weight:.1%}"}
                    for i in preview["intents"]]
            st.markdown("**What it would do**")
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        else:
            st.success("Nothing to do — the account already matches the plan. "
                       "This is what most days look like.")

        explain_row(["gross_exposure", "turnover", "slippage"])

    st.divider()

    # ---- step 2 ---------------------------------------------------------
    step(2, "Actually place the practice orders", active=bool(preview))
    if not preview:
        plain("Work out the plan first — the button above.")
    elif not preview["intents"]:
        plain("Nothing to place. The account already matches.")
    else:
        plain(f"This sends **{len(preview['intents'])} orders** to the practice "
              "account. Fake money. You can undo it by doing nothing — positions "
              "just sit there.")
        confirm = st.checkbox("I understand this places practice orders")
        if st.button("Place practice orders", type="primary", disabled=not confirm):
            try:
                trader, _ = connect(symbols, strategy, params, False,
                                    use_overlay, gross, bucket)
                result = trader.run_once()
                sent = [r for r in result["results"]
                        if r["action"] in ("buy", "sell")]
                st.success(f"Sent {len(sent)} orders to the practice account.")
                if sent:
                    st.dataframe(pd.DataFrame(sent), hide_index=True,
                                 use_container_width=True)
                skipped = [r for r in result["results"]
                           if r["action"] in ("skipped", "rejected")]
                if skipped:
                    st.warning("Some orders did not go through:")
                    st.dataframe(pd.DataFrame(skipped), hide_index=True,
                                 use_container_width=True)
                st.session_state.pop("preview", None)
            except Halted as exc:
                st.error(f"Trading stopped on purpose: {exc}")
            except BrokerError as exc:
                st.error(f"The broker refused: {exc}")

    st.divider()

    # ---- step 3 ---------------------------------------------------------
    step(3, "Check what you hold")
    if st.button("Refresh holdings"):
        try:
            from brokers.alpaca import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            account = broker.get_account()
            positions = broker.get_positions()

            cols = st.columns(3)
            explained_tile(cols[0], "gross_exposure", f"${account.equity:,.0f}",
                           "practice account value", label="Account value")
            explained_tile(cols[1], "total_return", f"{len(positions)}",
                           "things held right now", label="Positions")
            explained_tile(cols[2], "gross_exposure", f"${account.cash:,.0f}",
                           "not invested", label="Cash")

            if positions:
                rows = [{"ticker": p.symbol, "shares": round(p.qty, 2),
                         "worth": f"${p.market_value:,.0f}",
                         "up or down": f"${p.unrealized_pl:,.0f}"}
                        for p in positions.values()]
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)
            else:
                st.info("Nothing held yet.")
        except BrokerError as exc:
            st.error(f"Could not reach the practice account: {exc}")

    st.divider()
    with st.expander("What stops this doing something stupid?"):
        st.markdown(
            "Five things, all on by default:\n\n"
            "1. **It only ever uses the practice account.** Real trading needs "
            "different keys that are not set.\n"
            "2. **It waits for orders to fill.** It will not send a second order "
            "for a ticker while the first is still working — that is how people "
            "accidentally buy twice.\n"
            "3. **It checks you can afford it** before sending, rather than "
            "finding out through rejections.\n"
            "4. **It stops on a bad day.** If the account drops more than 3% in "
            "one day, it halts instead of trading through it.\n"
            "5. **It ignores tiny adjustments.** Rebalancing a fraction of a "
            "percent costs more in fees than it fixes.")


if __name__ == "__main__":
    main()
