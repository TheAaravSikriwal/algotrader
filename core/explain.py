"""Plain-English explanations for everything the app shows.

One place, so a number is described the same way wherever it appears, and so
nothing gets displayed that nobody can define.

Each entry answers four questions in order, because that is the order someone
new actually asks them: what is this, what counts as good, how was it worked
out, and what would fool me. The last one matters most -- a metric you cannot
be fooled by is rare, and pretending otherwise is how people lose money
confidently.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Explanation:
    label: str
    plain: str               # what it is, no jargon
    good: str = ""           # what counts as good
    formula: str = ""        # how it was worked out
    trap: str = ""           # what would mislead you

    def tooltip(self) -> str:
        bits = [self.plain]
        if self.good:
            bits.append(f"Good: {self.good}")
        if self.formula:
            bits.append(f"Worked out as: {self.formula}")
        if self.trap:
            bits.append(f"Careful: {self.trap}")
        return "\n\n".join(bits)

    def markdown(self) -> str:
        lines = [f"**{self.label}** — {self.plain}"]
        if self.good:
            lines.append(f"- *What's good:* {self.good}")
        if self.formula:
            lines.append(f"- *How it's worked out:* {self.formula}")
        if self.trap:
            lines.append(f"- *What can fool you:* {self.trap}")
        return "\n".join(lines)


METRICS: dict[str, Explanation] = {
    "total_return": Explanation(
        "Total return",
        "How much the account grew or shrank over the whole test, as a percentage.",
        good="Higher — but only compared to the benchmark underneath it.",
        formula="Money at the end divided by money at the start, minus one.",
        trap="Over ten years almost everything looks positive, because the market "
             "itself rose. The number beneath it is the one that matters."),
    "cagr": Explanation(
        "CAGR",
        "The steady yearly growth rate that would have got you to the same place.",
        good="Above what an index fund did over the same years.",
        formula="Total growth spread evenly across the years, compounded.",
        trap="It smooths everything. Two strategies with the same CAGR can feel "
             "completely different to live through."),
    "sharpe": Explanation(
        "Sharpe ratio",
        "Return earned per unit of nerve required. It divides gains by how bumpy "
        "the ride was.",
        good="Above 1 is respectable. Below 0.5 means you were paid little for "
             "the stress. Above 2 on a retail strategy usually means a mistake.",
        formula="Average yearly return divided by the yearly swing in returns.",
        trap="Rises if you simply trade less. A strategy that sits in cash most "
             "of the year can post a fine Sharpe while making almost nothing."),
    "max_drawdown": Explanation(
        "Max drawdown",
        "The worst peak-to-bottom fall along the way — the deepest hole you "
        "would have sat in.",
        good="Smaller. This is the number that decides whether you actually "
             "stick with a strategy.",
        formula="Largest drop from any previous high point in the equity curve.",
        trap="A backtest's drawdown is painless. The same 30% live, over eight "
             "months, is what makes people quit at the bottom."),
    "win_rate": Explanation(
        "Win rate",
        "The share of trades that made money.",
        good="Less important than it sounds.",
        formula="Winning trades divided by all trades.",
        trap="A 90% win rate can still lose overall if the 10% are catastrophic. "
             "Win rate without average win and loss beside it says nothing."),
    "time_in_market": Explanation(
        "Time in market",
        "The share of days you were actually holding something.",
        good="Depends entirely on the strategy's intent.",
        formula="Days with a position divided by all days.",
        trap="Near 100% means the strategy has quietly become buy-and-hold, "
             "whatever its name says."),
    "turnover": Explanation(
        "Turnover",
        "How many times a year the whole portfolio gets bought and sold.",
        good="Lower. Every rotation pays the spread.",
        formula="Total value traded in a year divided by portfolio value.",
        trap="At 70x a year, costs alone eat several percent — enough to turn a "
             "real edge into a loss. We have watched exactly that happen here."),
    "t_vs_ew": Explanation(
        "t vs equal weight",
        "Whether picking names beat simply holding all of them in equal amounts.",
        good="Above +2. Between -2 and +2 means the picking added nothing "
             "detectable.",
        formula="Average daily difference against the equal-weight basket, "
                "divided by how much that difference bounces around.",
        trap="A visibly better Sharpe can still score near zero here. That means "
             "the gap is within what a coin flip produces."),
    "t_vs_zero": Explanation(
        "t vs zero",
        "Whether the strategy made money at all, beyond what randomness explains.",
        good="Above +2.",
        formula="Average daily return divided by its own variability, scaled by "
                "the number of days.",
        trap="Long-only strategies clear this easily just by owning stocks in a "
             "rising market. It is not evidence of skill on its own."),
    "corrected_bar": Explanation(
        "Corrected bar",
        "The score a result must beat, raised every time you test something new.",
        good="You want a result above it. Nothing here has managed that.",
        formula="Bonferroni correction: the usual threshold made stricter in "
                "proportion to how many things have been tried.",
        trap="Test twenty ideas and one clears the ordinary bar by luck alone. "
             "This is the defence against believing that one."),
    "clustering": Explanation(
        "Clustering",
        "How much the events all happened on the same few days.",
        good="Low. Spread out means independent observations.",
        formula="Largest share of events falling on a single date.",
        trap="If most events share one day, they share one market move. The "
             "sample is smaller than the count suggests and the statistics flatter."),
    "event_day": Explanation(
        "Event day move",
        "What the stock did on the day of the event itself.",
        good="Not applicable — this is usually not tradeable.",
        formula="Return on day zero, measured against the benchmark.",
        trap="If you selected events by a price move, this just repeats your "
             "filter back at you. It will look enormous and mean nothing."),
    "drift": Explanation(
        "Drift after the event",
        "What the stock did in the days *after* the event, once the initial move "
        "had happened.",
        good="A consistent move with a t above 2. This is the part you could "
             "actually trade.",
        formula="Cumulative return from the next day onward, benchmark removed.",
        trap="Small drifts do not survive costs. Under about 0.3% there is "
             "nothing left after crossing the spread twice."),
    "slippage": Explanation(
        "Slippage",
        "The small loss on every trade from buying slightly high and selling "
        "slightly low.",
        good="Assume more than you think. 5 basis points is 0.05%.",
        formula="Applied to each fill, in both directions.",
        trap="Setting it to zero makes almost any high-turnover strategy look "
             "profitable. It is the single easiest way to fool yourself."),
    "gross_exposure": Explanation(
        "Gross exposure",
        "How much of the account is committed to positions in total.",
        good="100% means fully invested with no borrowing.",
        formula="Sum of all position sizes, ignoring direction.",
        trap="Above 100% means leverage — gains and losses both multiply."),
    "confidence": Explanation(
        "Confidence",
        "How strongly a reasoning call is held. It scales the size of the tilt.",
        good="Used honestly. A plausible idea with thin evidence belongs near 0.3.",
        formula="Multiplied by the requested tilt to get the actual position change.",
        trap="Everything at 0.9 means confidence is being used as enthusiasm."),
    "precedents": Explanation(
        "Precedents",
        "Past reasoning calls on similar events, and how they actually turned out.",
        good="More. One is an anecdote; a dozen starts to be evidence.",
        formula="Matched by event category, scored against the benchmark after "
                "the call expired.",
        trap="Every precedent was written before its outcome was known — that is "
             "what makes it worth reading. Judge it by count, not by story."),
}

CHARTS: dict[str, Explanation] = {
    "equity_curve": Explanation(
        "Equity curve",
        "Your account balance over time. One line is the strategy, the other is "
        "the simple alternative you are trying to beat.",
        good="The strategy line finishing above the other one.",
        trap="If the strategy line is below, the strategy lost — no matter how "
             "clever the rule sounds. That is the whole test."),
    "drawdown": Explanation(
        "Drawdown chart",
        "How far below its best-ever level the account was, on each day. Zero "
        "means at a new high; -20% means a fifth below the peak.",
        good="Shallow and short.",
        trap="Look at how *long* it stays underwater, not only how deep. Two "
             "years below the high is what breaks resolve."),
    "car": Explanation(
        "Event chart",
        "The average path of the stock around an event, with day zero being the "
        "event. Left of the line is before, right is after.",
        good="A clear step after the line that keeps going in one direction.",
        trap="A jump exactly at day zero often just reflects how the events were "
             "chosen, not something you could have traded."),
    "tstat_bars": Explanation(
        "Test results chart",
        "Every idea tested so far, as a bar. The red lines are the score a "
        "result must beat to count as real.",
        good="A bar poking past a red line. None has yet.",
        trap="Bars near zero are not failures, they are non-findings — the idea "
             "made no measurable difference either way."),
    "monthly": Explanation(
        "Monthly returns grid",
        "Every month coloured by whether it made or lost money. Blue is up, red "
        "is down.",
        good="Consistency. Scattered colour rather than one enormous month.",
        trap="One deep-blue month carrying the whole record is fragile — that "
             "month may never repeat."),
}

CONTROLS: dict[str, Explanation] = {
    "universe": Explanation(
        "Universe",
        "The list of things the strategy is allowed to buy. Change this and you "
        "change the answer."),
    "symbols": Explanation(
        "Symbols",
        "The specific tickers. SPY is the whole US market, QQQ is big tech, "
        "the XL- ones are single sectors, TLT is government bonds, GLD is gold."),
    "date_range": Explanation(
        "Date range",
        "The slice of history being tested. Longer is generally more trustworthy, "
        "because it covers more kinds of market."),
    "in_sample_split": Explanation(
        "In-sample split",
        "Cuts history in two. The strategy is judged on the second half, which it "
        "was never tuned on.",
        good="Leave it on. A strategy that leads in the first half and lags in "
             "the second was fitted to the past, not to the market."),
    "starting_cash": Explanation(
        "Starting cash",
        "The pretend account size. It changes the dollar figures but almost never "
        "the conclusion."),
    "max_tilt": Explanation(
        "Max total tilt",
        "A cap on how far one week's reasoning can move the portfolio.",
        good="Leave it low. One confident wrong call should cost a slice, not "
             "the book."),
}


def metric(key: str) -> Explanation | None:
    return METRICS.get(key)


def chart(key: str) -> Explanation | None:
    return CHARTS.get(key)


def control(key: str) -> Explanation | None:
    return CONTROLS.get(key)


def glossary_markdown(keys: list[str] | None = None) -> str:
    """A readable block explaining a set of metrics."""
    chosen = keys or list(METRICS)
    return "\n\n".join(METRICS[k].markdown() for k in chosen if k in METRICS)
