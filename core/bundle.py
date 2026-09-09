"""Save and reload a complete backtest run.

A bundle is one JSON file holding everything needed to *reproduce* a result:
the strategy and its parameters, the data window, the execution assumptions,
and the output (metrics, equity curve, trades).

That last distinction matters. Reloading a bundle re-runs the backtest from the
saved inputs and compares against the saved outputs, so a bundle you made last
month will tell you if something in the engine has drifted since.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FORMAT = "algotrader.bundle"
VERSION = 1


class BundleError(ValueError):
    pass


def build_bundle(*, symbol: str, start, end, timeframe: str, source: str,
                 strategy, result, stats: dict, notes: str = "") -> dict:
    equity = result.equity
    return {
        "format": FORMAT,
        "version": VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "run": {
            "symbol": symbol.upper(),
            "start": str(start)[:10],
            "end": str(end)[:10],
            "timeframe": timeframe,
            "source": source,
        },
        "strategy": {
            "name": strategy.name,
            "params": strategy.settings,
        },
        "execution": asdict(result.config),
        "metrics": {k: (None if isinstance(v, float) and pd.isna(v) else v)
                    for k, v in stats.items()},
        "equity": {
            "timestamps": [t.isoformat() for t in equity.index],
            "values": [float(v) for v in equity.to_numpy()],
        },
        "trades": json.loads(result.trades.to_json(orient="records", date_format="iso")),
    }


def save_bundle(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    return path


def load_bundle(source) -> dict:
    """Accepts a path, a JSON string, or any object with .read()."""
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        raw = Path(source).read_text(encoding="utf-8")
    else:
        raw = str(source)

    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BundleError(f"not valid JSON: {exc}") from exc

    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise BundleError("this file is not an algotrader bundle")
    if int(bundle.get("version", 0)) > VERSION:
        raise BundleError(
            f"bundle version {bundle['version']} is newer than this build "
            f"understands (v{VERSION}) -- update the project first")
    for key in ("run", "strategy", "execution"):
        if key not in bundle:
            raise BundleError(f"bundle is missing its '{key}' section")
    return bundle


def bundle_equity(bundle: dict) -> pd.Series:
    eq = bundle.get("equity") or {}
    if not eq.get("timestamps"):
        return pd.Series(dtype=float)
    return pd.Series(eq["values"], index=pd.to_datetime(eq["timestamps"]), name="equity")


def describe(bundle: dict) -> str:
    run, strat = bundle["run"], bundle["strategy"]
    params = ", ".join(f"{k}={v}" for k, v in strat.get("params", {}).items())
    total = bundle.get("metrics", {}).get("Total return")
    ret = f" | {total * 100:+.1f}%" if isinstance(total, (int, float)) else ""
    return (f"{run['symbol']} {run['start']}..{run['end']} | "
            f"{strat['name']}({params}){ret}")


def replay(bundle: dict, loader, engine, registry) -> tuple:
    """Re-run a bundle from its saved inputs.

    `loader`, `engine` and `registry` are injected so this module stays free of
    import cycles. Returns (result, stats, drift) where drift is the difference
    between the freshly computed final equity and the saved one.
    """
    run, strat = bundle["run"], bundle["strategy"]
    df = loader(run["symbol"], run["start"], run["end"], run["timeframe"], run["source"])

    cls = registry(strat["name"])
    known = {p.name for p in cls.params}
    params = {k: v for k, v in strat.get("params", {}).items() if k in known}
    dropped = set(strat.get("params", {})) - known
    if dropped:
        raise BundleError(
            f"'{strat['name']}' no longer accepts {sorted(dropped)} -- the strategy "
            "has changed since this bundle was saved")

    strategy = cls(**params)
    config = engine.BacktestConfig(**{
        k: v for k, v in bundle["execution"].items()
        if k in engine.BacktestConfig.__dataclass_fields__
    })
    result = engine.run_backtest(df, strategy.generate_signals(df), config)

    saved = bundle_equity(bundle)
    drift = None
    if not saved.empty:
        drift = float(result.equity.iloc[-1] - saved.iloc[-1])
    return result, strategy, drift
