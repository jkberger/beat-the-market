import math
from datetime import date

import numpy as np
import pandas as pd

from app.backtest.engine import EngineResult, Trade
from app.strategy.models import BacktestConfig

DAYS = 252
KEYS = [
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "win_rate",
    "avg_win",
    "avg_loss",
    "profit_factor",
    "num_trades",
    "time_in_market",
]


def _clean(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _ratio(num, den):
    if den is None or num is None or den == 0:
        return None
    return num / den


def compute_metrics(
    equity: pd.Series, trades: list[Trade], position: pd.Series
) -> dict[str, float | int | None]:
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    years = (len(eq) - 1) / DAYS
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else None
    std = rets.std() if len(rets) > 1 else 0.0
    vol = std * math.sqrt(DAYS) if len(rets) > 1 else None
    sharpe = _ratio(rets.mean() * math.sqrt(DAYS), std) if std > 1e-12 else None
    downside = np.sqrt((np.minimum(rets, 0) ** 2).mean()) if len(rets) else 0.0
    sortino = _ratio(rets.mean() * math.sqrt(DAYS), downside) if downside > 1e-12 else None
    max_dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    calmar = _ratio(cagr, abs(max_dd)) if max_dd < 0 else None

    r = np.array([t.return_pct for t in trades], dtype=float)
    wins, losses = r[r > 0], r[r < 0]
    m = {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": len(wins) / len(r) if len(r) else None,
        "avg_win": wins.mean() if len(wins) else None,
        "avg_loss": losses.mean() if len(losses) else None,
        "profit_factor": wins.sum() / -losses.sum() if len(losses) and len(wins) else None,
        "num_trades": len(trades),
        "time_in_market": float((position != 0).mean()) if len(position) else 0.0,
    }
    return {k: (m[k] if k == "num_trades" else _clean(m[k])) for k in KEYS}


def buy_and_hold(
    bars: pd.DataFrame,
    config: BacktestConfig,
    start: date | None = None,
    end: date | None = None,
) -> EngineResult:
    if start is not None:
        bars = bars[bars.index >= pd.Timestamp(start)]
    if end is not None:
        bars = bars[bars.index <= pd.Timestamp(end)]
    entry = float(bars["open"].iloc[0]) * (1 + config.slippage_bps / 1e4)
    shares = config.initial_equity / entry
    equity = bars["close"] * shares
    exit_price = float(bars["close"].iloc[-1])
    trade = Trade(
        "long",
        bars.index[0].date(),
        entry,
        bars.index[-1].date(),
        exit_price,
        exit_price / entry - 1,
        len(bars) - 1,
        "end",
    )
    return EngineResult(equity.astype(float), pd.Series(1, index=bars.index), [trade])


def objective_value(metrics: dict, objective: str) -> float:
    if objective == "sortino":
        s, dd = metrics.get("sortino"), metrics.get("max_drawdown")
        return float("-inf") if s is None else s - abs(dd or 0)
    value = metrics.get(objective)
    return float("-inf") if value is None else float(value)
