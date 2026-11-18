"""Walk-forward tuning and the locked final holdout.

Nothing in here may let tuning see data from a test window or from the holdout.
"""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal

import pandas as pd

from app.backtest.engine import Trade, backtest
from app.backtest.metrics import compute_metrics, objective_value
from app.strategy.grid import expand_grid
from app.strategy.models import BacktestConfig, RiskSettings, Strategy

MIN_TRADES = 5


class NotEnoughHistory(ValueError):
    pass


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class SegmentResult:
    label: Literal["in_sample", "out_of_sample", "holdout", "backtest"]
    equity: pd.Series
    trades: list[Trade]
    metrics: dict
    start: date
    end: date
    position: pd.Series | None = None


def split_dev(bars: pd.DataFrame, holdout_start: date) -> pd.DataFrame:
    return bars[bars.index < pd.Timestamp(holdout_start)]


def walk_forward_windows(
    index: pd.DatetimeIndex, train_years: int = 3, test_months: int = 6
) -> list[Window]:
    need = train_years + test_months / 12
    if len(index) == 0:
        raise NotEnoughHistory(
            f"Walk-forward needs at least {need:g} years before the holdout, and the chosen "
            "start date is on or after the holdout start."
        )
    first, last = index[0], index[-1]
    have = (last - first).days / 365.25
    if have < need:
        raise NotEnoughHistory(
            f"Walk-forward needs at least {need:g} years before the holdout; "
            f"this stock has {have:.1f}."
        )
    windows = []
    anchor = first
    while True:
        test_from = anchor + pd.DateOffset(years=train_years)
        test_to = test_from + pd.DateOffset(months=test_months)
        train = index[(index >= anchor) & (index < test_from)]
        test = index[(index >= test_from) & (index < test_to)]
        if len(test) == 0 or len(train) == 0:
            break
        windows.append(Window(train[0].date(), train[-1].date(), test[0].date(), test[-1].date()))
        if test[-1] >= last:
            break
        anchor = anchor + pd.DateOffset(months=test_months)
    return windows


def _segment(label, result, start, end) -> SegmentResult:
    return SegmentResult(
        label,
        result.equity,
        result.trades,
        compute_metrics(result.equity, result.trades, result.position),
        start,
        end,
        result.position,
    )


def tune(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date,
    end: date,
) -> tuple[Strategy, dict]:
    candidates = expand_grid(strategy)
    best, best_score, best_metrics = candidates[0], float("-inf"), None
    qualified = False
    for cand in candidates:
        r = backtest(bars, cand, risk, config, start=start, end=end)
        m = compute_metrics(r.equity, r.trades, r.position)
        if m["num_trades"] < MIN_TRADES:
            continue
        score = objective_value(m, risk.objective)
        if not qualified or score > best_score:
            best, best_score, best_metrics, qualified = cand, score, m, True
    return best, {
        "qualified": qualified,
        "candidates": len(candidates),
        "objective": risk.objective,
        "score": best_score if qualified else None,
        "train_metrics": best_metrics,
    }


def _chosen_values(strategy: Strategy, tuned: Strategy) -> dict:
    """The tuned values, keyed by a readable path, for display."""
    from app.strategy.grid import _find_ranges, _get

    raw, out = strategy.model_dump(), {}
    for path in _find_ranges(raw):
        out[".".join(str(p) for p in path)] = _get(tuned.model_dump(), path)
    return out


def run_walk_forward(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    holdout_start: date,
    train_years: int = 3,
    test_months: int = 6,
) -> tuple[SegmentResult, list[dict], Strategy]:
    dev = split_dev(bars, holdout_start)
    windows = walk_forward_windows(dev.index, train_years, test_months)
    equity_parts, position_parts, trades, info = [], [], [], []
    capital = config.initial_equity
    for w in windows:
        chosen, tinfo = tune(strategy, dev, risk, config, w.train_start, w.train_end)
        run_cfg = config.model_copy(update={"initial_equity": capital})
        r = backtest(dev, chosen, risk, run_cfg, start=w.test_start, end=w.test_end)
        equity_parts.append(r.equity)
        position_parts.append(r.position)
        trades.extend(r.trades)
        capital = float(r.equity.iloc[-1])
        info.append(
            {
                "window": asdict(w),
                "chosen": _chosen_values(strategy, chosen),
                "qualified": tinfo["qualified"],
                "test": compute_metrics(r.equity, r.trades, r.position),
            }
        )
    equity = pd.concat(equity_parts)
    position = pd.concat(position_parts)
    seg = SegmentResult(
        "out_of_sample",
        equity,
        trades,
        compute_metrics(equity, trades, position),
        windows[0].test_start,
        windows[-1].test_end,
        position,
    )
    # Final settings for the holdout: tuned on the most recent training span before it.
    last_from = (dev.index[-1] - pd.DateOffset(years=train_years)).date()
    final, _ = tune(strategy, dev, risk, config, last_from, dev.index[-1].date())
    return seg, info, final


def score_holdout(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    holdout_start: date,
) -> SegmentResult:
    if bars.index[-1] < pd.Timestamp(holdout_start):
        raise NotEnoughHistory("There is no data on or after the holdout start date yet.")
    r = backtest(bars, strategy, risk, config, start=holdout_start)
    return _segment("holdout", r, holdout_start, bars.index[-1].date())


def simple_segment(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date | None,
    end: date | None,
) -> SegmentResult:
    r = backtest(bars, strategy, risk, config, start=start, end=end)
    if len(r.equity) < 2:
        raise NotEnoughHistory(
            f"No price data between {start or 'the first day'} and {end or 'today'}."
        )
    return _segment("backtest", r, r.equity.index[0].date(), r.equity.index[-1].date())
