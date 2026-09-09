"""Strategy plug-in framework.

A strategy is a class that turns a bar DataFrame into a *target position*
series. Declaring ``params`` is what makes it show up in the UI with the right
controls -- you never touch the app code to add a strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class Param:
    """One tunable knob. The UI renders a control from this description."""
    name: str
    default: Any
    label: str = ""
    kind: str = "int"          # int | float | bool | choice
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list = field(default_factory=list)
    help: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()


class Strategy:
    """Base class. Subclasses override :meth:`generate_signals`."""

    name: str = "Unnamed"
    description: str = ""
    params: list[Param] = []

    #: Extra columns this strategy needs alongside open/high/low/close/volume.
    #: The tournament skips a strategy whose requirements a dataset cannot meet,
    #: rather than running it and collecting an exception per symbol.
    requires_features: list[str] = []

    @classmethod
    def can_run_on(cls, df) -> bool:
        return all(col in df.columns for col in cls.requires_features)

    def __init__(self, **kwargs):
        for p in self.params:
            setattr(self, p.name, kwargs.get(p.name, p.default))
        unknown = set(kwargs) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")

    # ---- to be implemented by subclasses -------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Target position for each bar: 1 long, 0 flat, -1 short.

        The value on bar *t* is decided from information available at the close
        of bar *t* and is executed by the engine at the open of bar *t+1*.
        Fractional values between -1 and 1 are allowed for partial sizing.
        """
        raise NotImplementedError

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Optional. Series to overlay on the price chart, keyed by label."""
        return {}

    # ---- helpers -------------------------------------------------------
    @property
    def settings(self) -> dict:
        return {p.name: getattr(self, p.name) for p in self.params}

    def __repr__(self):
        bits = ", ".join(f"{k}={v}" for k, v in self.settings.items())
        return f"{self.name}({bits})"


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator -- makes the strategy visible to the CLI and the UI."""
    REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    if name not in REGISTRY:
        raise KeyError(f"No strategy named {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available() -> list[str]:
    return sorted(REGISTRY)
